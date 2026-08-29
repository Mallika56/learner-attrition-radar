"""Lightweight experiment tracking: every training/retraining run is a row in
`experiment_runs`, so model quality over time can be queried without a
separate tracking service.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.ml_model import TrainingResult
from app.models import ExperimentRun


def record_run(db: Session, result: TrainingResult, n_estimators: int, max_depth: int) -> ExperimentRun:
    run = ExperimentRun(
        model_version=result.model_version,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_training_rows=result.n_training_rows,
        cv_auc_roc=result.cv_auc_roc,
        cv_precision=result.cv_precision,
        cv_recall=result.cv_recall,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def latest_run(db: Session) -> ExperimentRun | None:
    return db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).first()
