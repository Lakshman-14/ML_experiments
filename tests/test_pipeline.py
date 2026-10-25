"""Focused tests for the data pipeline and saved model contract."""

from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

from src.prepare_data import prepare_data
from src.validate_data import validate_data

ROOT = Path(__file__).resolve().parents[1]


class TourismPipelineTests(unittest.TestCase):
    def test_dataset_schema_and_split(self) -> None:
        frame = validate_data(ROOT / "data" / "tourism.csv")
        self.assertEqual(len(frame), 4128)
        with tempfile.TemporaryDirectory() as directory:
            train_path, test_path = prepare_data(
                ROOT / "data" / "tourism.csv", Path(directory) / "splits"
            )
            train, test = pd.read_csv(train_path), pd.read_csv(test_path)
            self.assertEqual(len(train) + len(test), len(frame))
            self.assertNotIn("CustomerID", train.columns)
            self.assertNotIn("Unnamed: 0", train.columns)

    def test_training_serialization_and_prediction(self) -> None:
        model_path = ROOT / "model" / "best_model.joblib"
        test_path = ROOT / "artifacts" / "splits" / "test.csv"
        if not model_path.is_file() or not test_path.is_file():
            self.skipTest("Run data preparation and model training before this integration check.")
        pipeline = joblib.load(model_path)
        features = pd.read_csv(test_path).drop(columns=["ProdTaken"]).iloc[:2]
        self.assertEqual(len(pipeline.predict(features)), 2)


if __name__ == "__main__":
    unittest.main()
