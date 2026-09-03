from tests.conftest import SAMPLE_TELEMETRY


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_risk_score_and_label(client):
    resp = client.post("/predict", json=SAMPLE_TELEMETRY)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_label"] in {"low", "medium", "high"}


def test_predict_rejects_invalid_payload(client):
    bad_payload = dict(SAMPLE_TELEMETRY, quiz_avg_score=150)
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_metrics_reports_prediction_count(client):
    client.post("/predict", json=SAMPLE_TELEMETRY)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.json()["prediction_count"] >= 1


def test_retrain_promotes_a_new_model_version(client):
    before = client.get("/health").json()["model_version"]
    resp = client.post("/retrain")
    assert resp.status_code == 200
    after = client.get("/health").json()["model_version"]
    assert after != before


def test_drift_with_no_predictions_reports_insufficient_data(client):
    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["features"] == {}
    assert body["overall_status"] == "insufficient_data"


def test_drift_after_predictions_reports_per_feature_psi(client):
    for _ in range(5):
        client.post("/predict", json=SAMPLE_TELEMETRY)
    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overall_status"] in {"stable", "moderate_shift", "significant_shift"}
    assert set(body["features"]) <= {
        "avg_session_length_minutes",
        "days_since_last_login",
        "quiz_avg_score",
        "video_completion_rate",
        "forum_posts_count",
        "assignment_submission_rate",
    }
    for feature_report in body["features"].values():
        assert feature_report["status"] in {"stable", "moderate_shift", "significant_shift"}
