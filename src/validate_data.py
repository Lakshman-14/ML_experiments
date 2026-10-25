"""Validate the tourism CSV schema and print a compact data-quality summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch",
}


def validate_data(data_path: Path) -> pd.DataFrame:
    """Load and validate required fields, target values, and basic quality."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    columns = {str(column) for column in frame.columns}
    missing = sorted(REQUIRED_COLUMNS - columns)
    unexpected = sorted(columns - REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset contains no records.")
    if frame["ProdTaken"].isna().any():
        raise ValueError("Target column ProdTaken contains missing values.")
    target_values = set(frame["ProdTaken"].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"ProdTaken must contain only 0/1 values; got {target_values}.")

    print(f"Dataset: {data_path}")
    print(f"Shape: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"Target counts:\n{frame['ProdTaken'].value_counts(dropna=False).sort_index()}")
    missing_counts = frame.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    print("Missing values by column:")
    print(missing_counts.to_string() if not missing_counts.empty else "None")
    print(f"Unexpected columns: {unexpected if unexpected else 'None'}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "tourism.csv",
    )
    args = parser.parse_args()
    validate_data(args.data_path)


if __name__ == "__main__":
    main()
