from app.features import FEATURE_NAMES, validate_features
from app.ml_model import predict, train
from tests.conftest import SAMPLE_TELEMETRY


def test_train_returns_cross_validated_metrics():
    result = train(n_rows=300)
    assert 0.0 <= result.cv_auc_roc <= 1.0
    assert 0.0 <= result.cv_precision <= 1.0
    assert 0.0 <= result.cv_recall <= 1.0
    assert result.n_training_rows == 300


def test_train_baseline_distribution_covers_every_feature():
    result = train(n_rows=300)
    for name in FEATURE_NAMES:
        assert name in result.baseline_distribution
        assert "mean" in result.baseline_distribution[name]
    assert "_bins" in result.baseline_distribution


def test_predict_returns_probability_in_unit_interval():
    result = train(n_rows=300)
    features = validate_features(SAMPLE_TELEMETRY)
    score = predict(result.model, features)
    assert 0.0 <= score <= 1.0
