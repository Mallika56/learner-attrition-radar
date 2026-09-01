"""CLI entry point: train the ensemble, save it, and log an experiment run.

    python -m scripts.train_model
"""

import logging

from app.database import Base, SessionLocal, engine
from app.experiment_tracking import record_run
from app.ml_model import MAX_DEPTH, N_ESTIMATORS, save, train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    Base.metadata.create_all(bind=engine)
    result = train()
    save(result)

    db = SessionLocal()
    try:
        record_run(db, result, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
    finally:
        db.close()

    logger.info(
        "Trained %s: AUC-ROC=%.3f precision=%.3f recall=%.3f",
        result.model_version,
        result.cv_auc_roc,
        result.cv_precision,
        result.cv_recall,
    )


if __name__ == "__main__":
    main()
