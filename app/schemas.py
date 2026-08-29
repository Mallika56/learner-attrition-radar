from pydantic import BaseModel, Field


class LearnerTelemetry(BaseModel):
    avg_session_length_minutes: float = Field(..., ge=0, le=600)
    days_since_last_login: float = Field(..., ge=0, le=365)
    quiz_avg_score: float = Field(..., ge=0, le=100)
    video_completion_rate: float = Field(..., ge=0, le=1)
    forum_posts_count: float = Field(..., ge=0)
    assignment_submission_rate: float = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "avg_session_length_minutes": 18.5,
                "days_since_last_login": 3.0,
                "quiz_avg_score": 74.0,
                "video_completion_rate": 0.62,
                "forum_posts_count": 2,
                "assignment_submission_rate": 0.8,
            }
        }
    }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    risk_score: float
    risk_label: str
    model_version: str


class RetrainResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_version: str
    n_training_rows: int
    cv_auc_roc: float
    cv_precision: float
    cv_recall: float
