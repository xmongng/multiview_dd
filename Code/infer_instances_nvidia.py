"""Infer Therapist QA cognitive-distortion instances with three NVIDIA NIM models."""

import argparse
import json
import os
import time

from nvidia_llm import DEFAULT_MODELS, complete, make_client


ALLOWED_TYPES = [
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
TYPE_LOOKUP = {value.casefold(): value for value in ALLOWED_TYPES}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path):
    if not os.path.exists(path):
        return set()
    return {str(row["id"]) for row in read_jsonl(path)}


def build_prompt(template, row, use_elb=True):
    replacements = {
        "{sentence}": str(row["text"]),
        "{emotion_info}": f"Emotion: {row.get('emotion', 'Not applicable')}" if use_elb else "",
        "{logic_info}": f"Logic: {row.get('logic', 'Not applicable')}" if use_elb else "",
        "{behavior_info}": f"Behavior: {row.get('behavior', 'Not applicable')}" if use_elb else "",
    }
    prompt = template
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def parse_instances(text):
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("No JSON array in model response")
    values = json.loads(text[start:end + 1])
    if not isinstance(values, list):
        raise ValueError("Model response is not a JSON array")

    parsed = []
    for item in values:
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("type", "")).strip()
        distortion_type = TYPE_LOOKUP.get(raw_type.casefold())
        score = item.get("salience score", item.get("probability"))
        if not distortion_type or score is None:
            continue
        try:
            score = max(float(score), 0.0)
        except (TypeError, ValueError):
            continue
        parsed.append({
            "type": distortion_type,
            "salience": score,
            "relevant_text": str(item.get("relevant_text", "")).strip(),
        })

    if not parsed:
        raise ValueError("Response contained no valid instances")
    total = sum(item["salience"] for item in parsed)
    if total <= 0:
        uniform = 1.0 / len(parsed)
        for item in parsed:
            item["salience"] = uniform
    else:
        for item in parsed:
            item["salience"] /= total
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="ELB JSONL from extract_elb_nvidia.py")
    parser.add_argument("--prompt", default="Prompts/therapist_qa_instances.txt")
    parser.add_argument("--output", default="Artifacts/therapist_qa_instances_elb.jsonl")
    parser.add_argument("--env-file")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--without-elb", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    template = open(args.prompt, encoding="utf-8").read()
    client = make_client(args.env_file, timeout=args.timeout)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = completed_ids(args.output)

    with open(args.output, "a", encoding="utf-8") as output:
        for position, row in enumerate(rows, 1):
            item_id = str(row["id"])
            if item_id in done:
                continue
            prompt = build_prompt(template, row, use_elb=not args.without_elb)
            all_instances = []
            for model in args.models:
                last_error = None
                for attempt in range(args.retries):
                    try:
                        raw = complete(client, model, prompt)
                        instances = parse_instances(raw)
                        for instance in instances:
                            instance["model"] = model
                        all_instances.extend(instances)
                        break
                    except Exception as exc:
                        last_error = exc
                        time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Inference failed for id={item_id}, model={model}: {last_error}")
                time.sleep(args.delay)

            record = {
                "id": item_id,
                "text": row["text"],
                "label": row["label"],
                "instances": all_instances,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{position}/{len(rows)}] {item_id}: {len(all_instances)} instances")


if __name__ == "__main__":
    main()
