"""Create reproducible stratified train/test CSVs from the registered dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.validate_data import validate_data

TARGET = "ProdTaken"
RANDOM_STATE = 42


def prepare_data(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    frame = validate_data(data_path)
    # Drop only CSV-export index columns; CustomerID is omitted from predictors.
    frame = frame.loc[:, ~frame.columns.astype(str).str.match(r"^Unnamed")]
    frame = frame.drop(columns=["CustomerID"])
    features = frame.drop(columns=[TARGET])
    target = frame[TARGET].astype("int64")
    counts = target.value_counts()
    stratify = target if len(counts) > 1 and counts.min() >= 2 else None
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_x.assign(**{TARGET: train_y}).to_csv(train_path, index=False)
    test_x.assign(**{TARGET: test_y}).to_csv(test_path, index=False)
    print(f"Training split: {train_path} ({len(train_x):,} rows)")
    print(f"Testing split: {test_path} ({len(test_x):,} rows)")
    print(f"Training class balance:\n{train_y.value_counts(normalize=True).sort_index()}")
    return train_path, test_path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=root / "data" / "tourism.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts" / "splits")
    args = parser.parse_args()
    prepare_data(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
