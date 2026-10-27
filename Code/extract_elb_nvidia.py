"""Extract ELB fields with an NVIDIA-hosted model and resumable JSONL output."""

import argparse
import json
import os
import time

import pandas as pd

from nvidia_llm import complete, make_client


def parse_object(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object in model response")
    value = json.loads(text[start:end + 1])
    return {key: str(value.get(key, "Not applicable")) for key in ("emotion", "logic", "behavior")}


def load_completed(path):
    completed = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed[str(row["id"])] = row
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared CSV with id,text,label")
    parser.add_argument("--prompt", default="Prompts/elb_extraction.txt")
    parser.add_argument("--output", default="Artifacts/therapist_qa_elb.jsonl")
    parser.add_argument("--env-file")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for column in ("id", "text", "label"):
        if column not in df:
            raise KeyError(f"Input is missing '{column}'")
    if args.limit:
        df = df.head(args.limit)

    template = open(args.prompt, encoding="utf-8").read()
    client = make_client(args.env_file, timeout=args.timeout)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    completed = load_completed(args.output)

    with open(args.output, "a", encoding="utf-8") as output:
        for position, row in df.iterrows():
            item_id = str(row["id"])
            if item_id in completed:
                continue
            prompt = template.replace("{sentence}", str(row["text"]))
            last_error = None
            for attempt in range(args.retries):
                try:
                    raw = complete(client, args.model, prompt)
                    elb = parse_object(raw)
                    record = {"id": item_id, "text": row["text"], "label": row["label"], **elb}
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"[{position + 1}/{len(df)}] {item_id}: OK")
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"ELB extraction failed for id={item_id}: {last_error}")
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
