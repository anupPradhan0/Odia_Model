from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "odia_dataset"
ODIA_CHAR_MIN = ord("\u0B00")
ODIA_CHAR_MAX = ord("\u0B7F")


@dataclass
class ValidationSummary:
    split: str
    rows: int
    missing_images: int
    empty_text: int
    duplicate_image_names: int
    duplicate_pairs: int
    invalid_unicode_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OCR label CSV files against the dataset images.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), help="Root folder containing odia_dataset.")
    parser.add_argument("--report-json", help="Optional path to write the validation report as JSON.")
    return parser.parse_args()


def contains_odia_text(text: str) -> bool:
    return any(ODIA_CHAR_MIN <= ord(char) <= ODIA_CHAR_MAX for char in text)


def validate_split(dataset_root: Path, split_name: str) -> ValidationSummary:
    csv_path = dataset_root / f"{split_name}_labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing labels file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"image_name", "text"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing_columns)}")

    image_names = df["image_name"].astype(str).str.strip()
    texts = df["text"].fillna("").astype(str).str.strip()

    missing_images = sum(not (dataset_root / image_name).exists() for image_name in image_names)
    empty_text = int((texts == "").sum())
    invalid_unicode_rows = sum(text != "" and not contains_odia_text(text) for text in texts)
    duplicate_image_names = sum(count - 1 for count in Counter(image_names).values() if count > 1)
    duplicate_pairs = int(df.duplicated(subset=["image_name", "text"]).sum())

    return ValidationSummary(
        split=split_name,
        rows=len(df),
        missing_images=missing_images,
        empty_text=empty_text,
        duplicate_image_names=duplicate_image_names,
        duplicate_pairs=duplicate_pairs,
        invalid_unicode_rows=invalid_unicode_rows,
    )


def compute_overlap(dataset_root: Path) -> dict[str, int]:
    train_df = pd.read_csv(dataset_root / "train_labels.csv")
    test_df = pd.read_csv(dataset_root / "test_labels.csv")
    train_texts = set(train_df["text"].fillna("").astype(str).str.strip())
    test_texts = set(test_df["text"].fillna("").astype(str).str.strip())
    overlap = train_texts.intersection(test_texts)
    return {
        "train_unique_texts": len(train_texts),
        "test_unique_texts": len(test_texts),
        "overlap_texts": len(overlap),
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()

    summaries = [validate_split(dataset_root, "train"), validate_split(dataset_root, "test")]
    overlap = compute_overlap(dataset_root)

    report = {
        "dataset_root": str(dataset_root),
        "summaries": [asdict(summary) for summary in summaries],
        "overlap": overlap,
    }

    for summary in summaries:
        print(f"[{summary.split}] rows={summary.rows}")
        print(
            "  missing_images={0} empty_text={1} duplicate_image_names={2} duplicate_pairs={3} invalid_unicode_rows={4}".format(
                summary.missing_images,
                summary.empty_text,
                summary.duplicate_image_names,
                summary.duplicate_pairs,
                summary.invalid_unicode_rows,
            )
        )
    print(
        "[overlap] train_unique_texts={0} test_unique_texts={1} overlap_texts={2}".format(
            overlap["train_unique_texts"],
            overlap["test_unique_texts"],
            overlap["overlap_texts"],
        )
    )

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {report_path}")


if __name__ == "__main__":
    main()