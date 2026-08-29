import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class PredictionLog(Base):
    """One row per /predict call: inputs, output, and the model version that served it."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    model_version: Mapped[str] = mapped_column(String, nullable=False)

    avg_session_length_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    days_since_last_login: Mapped[float] = mapped_column(Float, nullable=False)
    quiz_avg_score: Mapped[float] = mapped_column(Float, nullable=False)
    video_completion_rate: Mapped[float] = mapped_column(Float, nullable=False)
    forum_posts_count: Mapped[float] = mapped_column(Float, nullable=False)
    assignment_submission_rate: Mapped[float] = mapped_column(Float, nullable=False)

    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_label: Mapped[str] = mapped_column(String, nullable=False)


class ExperimentRun(Base):
    """One row per training/retraining run: hyperparameters and cross-validated metrics."""

    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    n_estimators: Mapped[int] = mapped_column(Integer, nullable=False)
    max_depth: Mapped[int] = mapped_column(Integer, nullable=False)
    n_training_rows: Mapped[int] = mapped_column(Integer, nullable=False)

    cv_auc_roc: Mapped[float] = mapped_column(Float, nullable=False)
    cv_precision: Mapped[float] = mapped_column(Float, nullable=False)
    cv_recall: Mapped[float] = mapped_column(Float, nullable=False)
