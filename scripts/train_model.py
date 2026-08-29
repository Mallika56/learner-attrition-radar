"""CLI entry point: train the ensemble, save it, and log an experiment run.

    python -m scripts.train_model
"""

from app.database import Base, SessionLocal, engine
from app.experiment_tracking import record_run
from app.ml_model import MAX_DEPTH, N_ESTIMATORS, save, train


def main() -> None:
    Base.metadata.create_all(bind=engine)
    result = train()
    save(result)

    db = SessionLocal()
    try:
        record_run(db, result, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
    finally:
        db.close()

    print(f"Trained {result.model_version}: AUC-ROC={result.cv_auc_roc:.3f} "
          f"precision={result.cv_precision:.3f} recall={result.cv_recall:.3f}")


if __name__ == "__main__":
    main()
