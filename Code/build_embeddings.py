"""Convert inferred bags into the NPZ tensor format consumed by train.py."""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


LABELS = [
    "All-or-nothing thinking",
    "Overgeneralization",
    "Mental filter",
    "Emotional reasoning",
    "Labeling",
    "Magnification",
    "Should statements",
    "Fortune-telling",
    "Mind Reading",
    "Personalization",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L12-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-instances", type=int, help="Optional cap; highest-salience instances are retained")
    parser.add_argument("--uniform-salience", action="store_true", help="Ablation: assign equal non-padding weights")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("No input bags found")
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    for row in rows:
        if row["label"] not in label_to_id:
            raise ValueError(f"Unknown label: {row['label']}")
        row["instances"] = sorted(
            row["instances"], key=lambda item: float(item["salience"]), reverse=True
        )
        if args.max_instances:
            row["instances"] = row["instances"][:args.max_instances]
        if not row["instances"]:
            raise ValueError(f"Bag id={row['id']} has no instances")

    max_instances = max(len(row["instances"]) for row in rows)
    texts = []
    spans = []
    for row in rows:
        start = len(texts)
        texts.append(row["text"])
        texts.extend(
            f"{item['type']}: {item['relevant_text']}" for item in row["instances"]
        )
        spans.append((start, len(row["instances"])))

    encoder = SentenceTransformer(args.model)
    encoded = encoder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    if encoded.shape[1] != 384:
        raise ValueError(f"Expected 384-dimensional embeddings, got {encoded.shape[1]}")

    embeddings = np.zeros((len(rows), 1 + max_instances, 384), dtype=np.float32)
    probabilities = np.zeros((len(rows), max_instances), dtype=np.float32)
    labels = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    ids = []

    for index, (row, (start, count)) in enumerate(zip(rows, spans)):
        embeddings[index, 0] = encoded[start]
        embeddings[index, 1:1 + count] = encoded[start + 1:start + 1 + count]
        scores = np.asarray([float(item["salience"]) for item in row["instances"]], dtype=np.float32)
        if args.uniform_salience:
            scores[:] = 1.0
        scores /= scores.sum()
        probabilities[index, :count] = scores
        labels[index, label_to_id[row["label"]]] = 1.0
        ids.append(str(row["id"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        probabilities=probabilities,
        labels=labels,
        ids=np.asarray(ids),
        label_names=np.asarray(LABELS),
    )
    metadata_path = args.output + ".json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({
            "sentence_transformer": args.model,
            "num_examples": len(rows),
            "max_instances": max_instances,
            "labels": LABELS,
            "uniform_salience": args.uniform_salience,
        }, handle, indent=2)
    print(f"Saved {embeddings.shape=} {probabilities.shape=} {labels.shape=} to {args.output}")


if __name__ == "__main__":
    main()
