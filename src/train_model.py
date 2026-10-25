"""Tune candidate classifiers, evaluate once on held-out data, and save artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

TARGET = "ProdTaken"
RANDOM_STATE = 42


def make_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )


def train_model(train_path: Path, test_path: Path, output_dir: Path) -> dict:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if TARGET not in train or TARGET not in test:
        raise ValueError(f"Both split files must include the {TARGET} target.")
    if list(train.drop(columns=[TARGET]).columns) != list(test.drop(columns=[TARGET]).columns):
        raise ValueError("Training and testing feature columns do not match.")
    train_x, train_y = train.drop(columns=[TARGET]), train[TARGET].astype(int)
    test_x, test_y = test.drop(columns=[TARGET]), test[TARGET].astype(int)
    if train_y.nunique() < 2:
        raise ValueError("Training target must contain both classes.")

    pipeline = make_pipeline(train_x)
    parameter_grid = [
        {
            "model": [DecisionTreeClassifier(random_state=RANDOM_STATE)],
            "model__max_depth": [5, 10, None],
            "model__min_samples_leaf": [1, 5],
        },
        {
            "model": [RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)],
            "model__n_estimators": [150],
            "model__max_depth": [12, None],
            "model__min_samples_leaf": [1, 3],
        },
        {
            "model": [GradientBoostingClassifier(random_state=RANDOM_STATE)],
            "model__n_estimators": [100, 150],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    ]
    class_counts = train_y.value_counts()
    folds = min(3, int(class_counts.min()))
    if folds < 2:
        raise ValueError("Not enough examples in each class for stratified cross-validation.")
    cross_validation = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(train_x, train_y)

    predictions = search.predict(test_x)
    probabilities = search.predict_proba(test_x)[:, 1]
    metrics = {
        "selection_metric": "mean cross-validated F1",
        "best_cv_f1": float(search.best_score_),
        "best_parameters": {
            key: (value.__class__.__name__ if key == "model" else value)
            for key, value in search.best_params_.items()
        },
        "test_metrics": {
            "accuracy": float(accuracy_score(test_y, predictions)),
            "precision": float(precision_score(test_y, predictions, zero_division=0)),
            "recall": float(recall_score(test_y, predictions, zero_division=0)),
            "f1": float(f1_score(test_y, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_y, probabilities)),
            "confusion_matrix_labels_0_1": confusion_matrix(
                test_y, predictions, labels=[0, 1]
            ).tolist(),
        },
        "test_rows": int(len(test_y)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, output_dir / "best_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    results = pd.DataFrame(search.cv_results_)
    results["params"] = results["params"].map(
        lambda params: json.dumps(
            {key: (value.__class__.__name__ if key == "model" else value)
             for key, value in params.items()},
            sort_keys=True,
            default=str,
        )
    )
    results.to_csv(output_dir / "cv_results.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Tuning combinations logged: {len(results)}")
    print(f"Saved selected pipeline: {output_dir / 'best_model.joblib'}")
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=root / "artifacts/splits/train.csv")
    parser.add_argument("--test-path", type=Path, default=root / "artifacts/splits/test.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "model")
    args = parser.parse_args()
    train_model(args.train_path, args.test_path, args.output_dir)


if __name__ == "__main__":
    main()
