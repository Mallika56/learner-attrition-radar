import pandas as pd

from app.features import FEATURE_NAMES, generate_synthetic_dataset, validate_features
from tests.conftest import SAMPLE_TELEMETRY


def test_validate_features_returns_ordered_dataframe():
    df = validate_features(SAMPLE_TELEMETRY)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FEATURE_NAMES
    assert df.shape == (1, len(FEATURE_NAMES))


def test_synthetic_dataset_has_expected_columns_and_label():
    data = generate_synthetic_dataset(n_rows=500, seed=1)
    assert set(FEATURE_NAMES) <= set(data.columns)
    assert "dropped_out" in data.columns
    assert set(data["dropped_out"].unique()) <= {0, 1}
    assert len(data) == 500


def test_synthetic_dataset_feature_ranges_are_plausible():
    data = generate_synthetic_dataset(n_rows=1000, seed=2)
    assert (data["quiz_avg_score"] >= 0).all() and (data["quiz_avg_score"] <= 100).all()
    assert (data["video_completion_rate"] >= 0).all() and (data["video_completion_rate"] <= 1).all()
    assert (data["assignment_submission_rate"] >= 0).all()
