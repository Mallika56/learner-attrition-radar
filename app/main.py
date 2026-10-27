import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from app import ml_model
from app.database import Base, engine, get_db
from app.drift import compute_psi_report
from app.experiment_tracking import latest_run, record_run
from app.features import FEATURE_NAMES, validate_features
from app.ml_model import MAX_DEPTH, N_ESTIMATORS
from app.models import PredictionLog
from app.schemas import LearnerTelemetry, PredictionResponse, RetrainResponse

logger = logging.getLogger(__name__)

RISK_THRESHOLD_HIGH = 0.66
RISK_THRESHOLD_MEDIUM = 0.33

_state: dict = {"model": None, "model_version": None}


def _risk_label(score: float) -> str:
    if score >= RISK_THRESHOLD_HIGH:
        return "high"
    if score >= RISK_THRESHOLD_MEDIUM:
        return "medium"
    return "low"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup hook: create tables if missing and load (or train) the model into `_state`."""
    Base.metadata.create_all(bind=engine)
    model, version = ml_model.load()
    _state["model"] = model
    _state["model_version"] = version
    logger.info("Loaded model %s", version)
    yield


app = FastAPI(
    title="Learner Attrition Radar",
    description="Predicts dropout risk for self-paced online learners from behavioral telemetry.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Liveness check: reports whether a model is loaded and which version."""
    return {"status": "ok", "model_version": _state["model_version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: LearnerTelemetry, db: Session = Depends(get_db)) -> PredictionResponse:
    """Score one learner's telemetry, log the prediction, and return the risk label."""
    features = validate_features(payload.model_dump())
    score = ml_model.predict(_state["model"], features)
    label = _risk_label(score)

    log = PredictionLog(
        model_version=_state["model_version"],
        risk_score=score,
        risk_label=label,
        **payload.model_dump(),
    )
    db.add(log)
    db.commit()

    return PredictionResponse(
        risk_score=round(score, 4), risk_label=label, model_version=_state["model_version"]
    )


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)) -> dict:
    """Prediction volume plus the most recent training run's cross-validated metrics."""
    run = latest_run(db)
    prediction_count = db.query(PredictionLog).count()
    if run is None:
        return {"prediction_count": prediction_count, "latest_experiment": None}
    return {
        "prediction_count": prediction_count,
        "latest_experiment": {
            "model_version": run.model_version,
            "cv_auc_roc": run.cv_auc_roc,
            "cv_precision": run.cv_precision,
            "cv_recall": run.cv_recall,
            "n_training_rows": run.n_training_rows,
        },
    }


@app.get("/drift")
def drift(db: Session = Depends(get_db), window: int = 200) -> dict:
    """PSI drift report comparing the last `window` predictions' feature values
    against the training-time baseline distribution."""
    recent = (
        db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(window).all()
    )
    baseline = ml_model.load_baseline_distribution()
    if not recent or not baseline:
        return {"features": {}, "overall_psi": 0.0, "overall_status": "insufficient_data"}

    recent_features = {name: [getattr(row, name) for row in recent] for name in FEATURE_NAMES}
    return compute_psi_report(recent_features, baseline)


@app.post("/retrain", response_model=RetrainResponse)
def retrain(db: Session = Depends(get_db)) -> RetrainResponse:
    """Automated retraining: re-fit the ensemble on freshly generated synthetic
    data and promote it if training succeeds. In production this would train on
    accumulated labeled outcomes instead of synthetic data."""
    result = ml_model.train()
    ml_model.save(result)
    record_run(db, result, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)

    _state["model"] = result.model
    _state["model_version"] = result.model_version

    return RetrainResponse(
        model_version=result.model_version,
        n_training_rows=result.n_training_rows,
        cv_auc_roc=round(result.cv_auc_roc, 4),
        cv_precision=round(result.cv_precision, 4),
        cv_recall=round(result.cv_recall, 4),
    )
