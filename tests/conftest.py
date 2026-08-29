import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import ml_model
from app.database import Base, get_db
from app.main import app
from app.ml_model import train

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def trained_model(monkeypatch):
    """Train a small model in-memory instead of touching disk, and point the
    app's module-level state at it so /predict etc. don't need a real model file."""
    result = train(n_rows=300)
    monkeypatch.setattr(ml_model, "load", lambda: (result.model, result.model_version))
    monkeypatch.setattr(ml_model, "load_baseline_distribution", lambda: result.baseline_distribution)
    return result


@pytest.fixture()
def client(db_session, trained_model):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


SAMPLE_TELEMETRY = {
    "avg_session_length_minutes": 18.5,
    "days_since_last_login": 3.0,
    "quiz_avg_score": 74.0,
    "video_completion_rate": 0.62,
    "forum_posts_count": 2,
    "assignment_submission_rate": 0.8,
}
