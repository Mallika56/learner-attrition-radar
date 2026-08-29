"""Ensemble model wrapper: train, persist, load, and score.

RandomForestClassifier is the ensemble (bonus: cross-validation reporting of
AUC-ROC / precision / recall accompanies every training run).
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from app.features import FEATURE_NAMES, generate_synthetic_dataset

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
BASELINE_PATH = os.path.join(MODEL_DIR, "baseline_distribution.json")

N_ESTIMATORS = 200
MAX_DEPTH = 8


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    model_version: str
    n_training_rows: int
    cv_auc_roc: float
    cv_precision: float
    cv_recall: float
    baseline_distribution: dict


def _model_version() -> str:
    return datetime.datetime.utcnow().strftime("rf-%Y%m%d%H%M%S")


def train(seed: int = 42, n_rows: int = 4000) -> TrainingResult:
    """Train the ensemble on freshly generated synthetic data and cross-validate it."""
    data = generate_synthetic_dataset(n_rows=n_rows, seed=seed)
    X = data[FEATURE_NAMES]
    y = data["dropped_out"]

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=seed
    )

    scoring = {"auc": "roc_auc", "precision": "precision", "recall": "recall"}
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

    model.fit(X, y)

    baseline_distribution = {
        name: {"mean": float(X[name].mean()), "std": float(X[name].std())}
        for name in FEATURE_NAMES
    }
    # Deciles of each feature over the training set — the reference bins PSI compares against.
    baseline_distribution["_bins"] = {
        name: [float(q) for q in np.quantile(X[name], np.linspace(0, 1, 11))]
        for name in FEATURE_NAMES
    }

    return TrainingResult(
        model=model,
        model_version=_model_version(),
        n_training_rows=len(data),
        cv_auc_roc=float(np.mean(cv_results["test_auc"])),
        cv_precision=float(np.mean(cv_results["test_precision"])),
        cv_recall=float(np.mean(cv_results["test_recall"])),
        baseline_distribution=baseline_distribution,
    )


def save(result: TrainingResult) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": result.model, "model_version": result.model_version}, MODEL_PATH)
    with open(BASELINE_PATH, "w") as fh:
        json.dump(result.baseline_distribution, fh, indent=2)


def load() -> tuple[RandomForestClassifier, str]:
    if not os.path.exists(MODEL_PATH):
        result = train()
        save(result)
        return result.model, result.model_version
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["model_version"]


def load_baseline_distribution() -> dict:
    if not os.path.exists(BASELINE_PATH):
        return {}
    with open(BASELINE_PATH) as fh:
        return json.load(fh)


def predict(model: RandomForestClassifier, features: pd.DataFrame) -> float:
    return float(model.predict_proba(features[FEATURE_NAMES])[0][1])
