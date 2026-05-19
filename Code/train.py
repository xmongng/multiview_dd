import os
import gc
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


HIDDEN_DIM = 128
DROPOUT_RATE = 0.3
INITIAL_LR = 0.0005
MIN_LR = 0.00001
DECAY_RATE = 0.00001
NUM_EPOCHS = 100
NUM_RUNS = 10
PATIENCE = 10
BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_HEADS = 4
INPUT_DIM = 384


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class MultiViewGatedAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        num_heads=NUM_HEADS,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.feature_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            )
            for _ in range(num_heads)
        ])

        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])

        self.original_feat_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def normalize_salience(prob, eps=1e-8):
        row_sum = prob.sum(dim=1, keepdim=True)
        normalized = prob / (row_sum + eps)
        normalized = torch.where(row_sum > eps, normalized, torch.zeros_like(normalized))
        return normalized

    def forward(self, x, prob, original_emb):
        prob = self.normalize_salience(prob)

        view_outputs = []

        for i in range(self.num_heads):
            gate_values = self.gate_nets[i](x)
            features = self.feature_nets[i](x)
            gated_features = gate_values * features * prob.unsqueeze(-1)
            h_k = torch.sum(gated_features, dim=1)
            view_outputs.append(h_k)

        h_multi = torch.stack(view_outputs, dim=0).mean(dim=0)
        original_feat = self.original_feat_net(original_emb)
        fused = torch.cat([h_multi, original_feat], dim=1)
        fused_output = self.fusion_net(fused)

        return self.classifier(fused_output)


def stratified_split(labels, val_ratio=0.1, test_ratio=0.1):
    val_indices = []
    test_indices = []

    for cls in range(NUM_CLASSES):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)

        val_n = int(len(cls_indices) * val_ratio)
        test_n = int(len(cls_indices) * test_ratio)

        val_indices.extend(cls_indices[:val_n])
        test_indices.extend(cls_indices[val_n:val_n + test_n])

    val_indices = sorted(list(set(val_indices)))
    test_indices = sorted(list(set(test_indices) - set(val_indices)))

    train_indices = sorted(
        list(set(range(len(labels))) - set(val_indices) - set(test_indices))
    )

    return train_indices, val_indices, test_indices


def train_one_run(
    instance_emb,
    probabilities,
    original_emb,
    labels,
    seed,
    device,
):
    set_seed(seed)

    print(f"\n  Seed: {seed}")

    label_np = labels.cpu().numpy()
    train_idx, val_idx, test_idx = stratified_split(label_np)

    dataset = torch.utils.data.TensorDataset(
        instance_emb,
        probabilities,
        original_emb,
        labels,
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MultiViewGatedAttentionMIL().to(device)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    criterion = nn.CrossEntropyLoss()

    current_lr = INITIAL_LR
    best_val_f1 = 0.0
    best_model_state = None
    best_val_metrics = {}
    patience_counter = 0
    min_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for x_batch, prob_batch, orig_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            prob_batch = prob_batch.to(device)
            orig_batch = orig_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch, prob_batch, orig_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        current_lr = max(current_lr - DECAY_RATE, MIN_LR)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        model.eval()
        val_true = []
        val_pred = []
        val_loss_total = 0.0

        with torch.no_grad():
            for x_batch, prob_batch, orig_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                prob_batch = prob_batch.to(device)
                orig_batch = orig_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch, prob_batch, orig_batch)
                loss = criterion(output, y_batch)

                val_loss_total += loss.item()

                preds = torch.argmax(output, dim=1)
                val_pred.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        avg_train_loss = total_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss_total / max(len(val_loader), 1)

        _, _, val_f1_w, _ = precision_recall_fscore_support(
            val_true,
            val_pred,
            average="weighted",
            zero_division=0,
        )

        val_acc = accuracy_score(val_true, val_pred)

        val_prec_w, val_rec_w, _, _ = precision_recall_fscore_support(
            val_true,
            val_pred,
            average="weighted",
            zero_division=0,
        )

        val_prec_m, val_rec_m, val_f1_m, _ = precision_recall_fscore_support(
            val_true,
            val_pred,
            average="macro",
            zero_division=0,
        )

        print(
            f"    Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1-W: {val_f1_w:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_f1_w > best_val_f1:
            best_val_f1 = val_f1_w
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            best_val_metrics = {
                "acc": val_acc,
                "prec_w": val_prec_w,
                "rec_w": val_rec_w,
                "f1_w": val_f1_w,
                "prec_m": val_prec_m,
                "rec_m": val_rec_m,
                "f1_m": val_f1_m,
            }

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch + 1}.")
                break

    if best_model_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    test_true = []
    test_pred = []
    test_probs = []

    with torch.no_grad():
        for x_batch, prob_batch, orig_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            prob_batch = prob_batch.to(device)
            orig_batch = orig_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch, prob_batch, orig_batch)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)

            test_probs.extend(probs.cpu().numpy())
            test_pred.extend(preds.cpu().numpy())
            test_true.extend(y_batch.cpu().numpy())

    acc = accuracy_score(test_true, test_pred)

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        test_true,
        test_pred,
        average="weighted",
        zero_division=0,
    )

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        test_true,
        test_pred,
        average="macro",
        zero_division=0,
    )

    class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(
        test_true,
        test_pred,
        average=None,
        zero_division=0,
        labels=list(range(NUM_CLASSES)),
    )

    test_metrics = {
        "acc": acc,
        "prec_w": prec_w,
        "rec_w": rec_w,
        "f1_w": f1_w,
        "prec_m": prec_m,
        "rec_m": rec_m,
        "f1_m": f1_m,
        "class_prec": class_prec,
        "class_rec": class_rec,
        "class_f1": class_f1,
        "test_indices": test_idx,
        "true_labels": test_true,
        "predicted_labels": test_pred,
        "prediction_probs": test_probs,
    }

    return best_val_metrics, test_metrics, best_model_state


def run_experiment(
    mode,
    npz_path,
    output_dir=".",
    instance_start_idx=2,
    device=torch.device("cpu"),
):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {mode} | File: {npz_path}")
    print(f"{'=' * 60}")

    data = np.load(npz_path, allow_pickle=True)

    if "embeddings" not in data:
        raise KeyError("NPZ file must contain key: 'embeddings'")

    if "labels" not in data:
        raise KeyError("NPZ file must contain key: 'labels'")

    if "probabilities" not in data:
        raise KeyError("NPZ file must contain key: 'probabilities'")

    embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
    labels = torch.tensor(np.argmax(data["labels"], axis=1), dtype=torch.long)
    probabilities = torch.tensor(data["probabilities"], dtype=torch.float32)

    if embeddings.ndim != 3:
        raise ValueError(
            f"'embeddings' must be 3D. Got shape: {embeddings.shape}"
        )

    if probabilities.ndim != 2:
        raise ValueError(
            f"'probabilities' must be 2D. Got shape: {probabilities.shape}"
        )

    if embeddings.shape[2] != INPUT_DIM:
        raise ValueError(
            f"Embedding dimension mismatch. Expected {INPUT_DIM}, got {embeddings.shape[2]}"
        )

    original_emb = embeddings[:, 0, :]
    instance_emb = embeddings[:, instance_start_idx:, :]

    if instance_emb.shape[1] != probabilities.shape[1]:
        raise ValueError(
            "Number of instance embeddings and probabilities do not match.\n"
            f"instance_emb shape: {instance_emb.shape}\n"
            f"probabilities shape: {probabilities.shape}\n"
            "Try changing --instance_start_idx to 1 or 2."
        )

    print(f"Embeddings shape      : {embeddings.shape}")
    print(f"Original emb shape    : {original_emb.shape}")
    print(f"Instance emb shape    : {instance_emb.shape}")
    print(f"Probabilities shape   : {probabilities.shape}")
    print(f"Labels shape          : {labels.shape}")
    print(f"Instance start index  : {instance_start_idx}")
    print(f"Device                : {device}")

    seeds = random.sample(range(1000), NUM_RUNS)
    print(f"Random seeds for this experiment: {seeds}")

    val_metrics_all = []
    test_metrics_all = []

    for run in range(NUM_RUNS):
        print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")

        val_metrics, test_metrics, model_state = train_one_run(
            instance_emb=instance_emb,
            probabilities=probabilities,
            original_emb=original_emb,
            labels=labels,
            seed=seeds[run],
            device=device,
        )

        val_metrics_all.append(val_metrics)
        test_metrics_all.append(test_metrics)

        model_path = os.path.join(output_dir, f"model_{mode}_run{run + 1}.pt")
        torch.save(model_state, model_path)

        results_df = pd.DataFrame({
            "data_index": test_metrics["test_indices"],
            "true_label": test_metrics["true_labels"],
            "predicted_label": test_metrics["predicted_labels"],
            "correct": [
                1 if t == p else 0
                for t, p in zip(
                    test_metrics["true_labels"],
                    test_metrics["predicted_labels"],
                )
            ],
            "confidence": [
                float(np.max(prob))
                for prob in test_metrics["prediction_probs"]
            ],
            **{
                f"prob_class_{c}": [
                    float(prob[c])
                    for prob in test_metrics["prediction_probs"]
                ]
                for c in range(NUM_CLASSES)
            },
        })

        csv_path = os.path.join(output_dir, f"test_results_{mode}_run{run + 1}.csv")
        results_df.to_csv(csv_path, index=False)

    def mean_std(metric_key):
        values = [m[metric_key] for m in test_metrics_all]
        return float(np.mean(values)), float(np.std(values))

    print(f"\n{'=' * 60}")
    print(f"Summary — {mode} ({NUM_RUNS} runs)")
    print(f"{'=' * 60}")

    acc_mean, acc_std = mean_std("acc")
    prec_w_mean, prec_w_std = mean_std("prec_w")
    rec_w_mean, rec_w_std = mean_std("rec_w")
    f1_w_mean, f1_w_std = mean_std("f1_w")

    prec_m_mean, prec_m_std = mean_std("prec_m")
    rec_m_mean, rec_m_std = mean_std("rec_m")
    f1_m_mean, f1_m_std = mean_std("f1_m")

    print("[ Test — Weighted ]")
    print(f"  Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Precision: {prec_w_mean:.4f} ± {prec_w_std:.4f}")
    print(f"  Recall   : {rec_w_mean:.4f} ± {rec_w_std:.4f}")
    print(f"  F1       : {f1_w_mean:.4f} ± {f1_w_std:.4f}")

    print("[ Test — Macro ]")
    print(f"  Precision: {prec_m_mean:.4f} ± {prec_m_std:.4f}")
    print(f"  Recall   : {rec_m_mean:.4f} ± {rec_m_std:.4f}")
    print(f"  F1       : {f1_m_mean:.4f} ± {f1_m_std:.4f}")

    print("\n[ Per-class F1 ]")

    for c in range(NUM_CLASSES):
        f1_values = [m["class_f1"][c] for m in test_metrics_all]
        print(f"  Class {c}: {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")

    val_f1_values = [m["f1_w"] for m in val_metrics_all]

    print("[ Validation — Weighted ]")
    print(f"  F1: {np.mean(val_f1_values):.4f} ± {np.std(val_f1_values):.4f}")

    all_runs = []

    for run in range(NUM_RUNS):
        csv_path = os.path.join(output_dir, f"test_results_{mode}_run{run + 1}.csv")
        df = pd.read_csv(csv_path)
        df["run"] = run + 1
        all_runs.append(df)

    combined = pd.concat(all_runs, ignore_index=True)
    combined_path = os.path.join(output_dir, f"test_results_{mode}_all_runs.csv")
    combined.to_csv(combined_path, index=False)

    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Multi-View Gated Attention MIL for cognitive distortion detection"
    )

    parser.add_argument(
        "--elb",
        required=True,
        help="Path to ELB-processed .npz file",
    )

    parser.add_argument(
        "--no_elb",
        required=True,
        help="Path to non-ELB-processed .npz file",
    )

    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save results",
    )

    parser.add_argument(
        "--instance_start_idx",
        type=int,
        default=2,
        choices=[1, 2],
        help="Index where instance embeddings start in the embeddings array",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )

    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)

    file_paths = {
        "ELB": args.elb,
        "no_ELB": args.no_elb,
    }

    for mode, path in file_paths.items():
        run_experiment(
            mode=mode,
            npz_path=path,
            output_dir=args.output_dir,
            instance_start_idx=args.instance_start_idx,
            device=device,
        )
