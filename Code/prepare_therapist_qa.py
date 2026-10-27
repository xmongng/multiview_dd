"""Download and prepare the 1,597-example Therapist QA subset used by MVACD."""

import argparse
import io
import os
import urllib.request
import zipfile

import pandas as pd


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sagarikashreevastava/cognitive-distortion-detetction-dataset"
)
EXPECTED_COUNTS = {
    "Mind Reading": 239,
    "Overgeneralization": 239,
    "Magnification": 195,
    "Labeling": 165,
    "Personalization": 153,
    "Fortune-telling": 143,
    "Emotional Reasoning": 134,
    "Mental filter": 122,
    "Should statements": 107,
    "All-or-nothing thinking": 100,
}


def load_annotated_csv(input_path=None):
    if input_path:
        if input_path.lower().endswith(".zip"):
            with zipfile.ZipFile(input_path) as archive:
                return pd.read_csv(archive.open("Annotated_data.csv"))
        return pd.read_csv(input_path)

    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "MVACD-reproduction"})
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return pd.read_csv(archive.open("Annotated_data.csv"))


def prepare(df):
    required = {"Id_Number", "Patient Question", "Dominant Distortion"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing source columns: {sorted(missing)}")

    result = df.loc[df["Dominant Distortion"] != "No Distortion", [
        "Id_Number", "Patient Question", "Dominant Distortion"
    ]].copy()
    result.columns = ["id", "text", "label"]
    result["text"] = result["text"].astype(str).str.strip()
    result = result[result["text"] != ""].reset_index(drop=True)

    counts = result["label"].value_counts().to_dict()
    if len(result) != 1597 or counts != EXPECTED_COUNTS:
        raise ValueError(
            "Prepared data does not match the paper's 1,597-row distribution. "
            f"Observed rows={len(result)}, counts={counts}"
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Optional Annotated_data.csv or downloaded ZIP")
    parser.add_argument("--output", default="Data/therapist_qa.csv")
    args = parser.parse_args()

    result = prepare(load_annotated_csv(args.input))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} paper-matching examples to {args.output}")
    print(result["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
