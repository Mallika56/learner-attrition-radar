import pytest
from pydantic import ValidationError

from app.schemas import LearnerTelemetry
from tests.conftest import SAMPLE_TELEMETRY


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("avg_session_length_minutes", -1),
        ("avg_session_length_minutes", 601),
        ("days_since_last_login", -1),
        ("days_since_last_login", 366),
        ("quiz_avg_score", -1),
        ("quiz_avg_score", 101),
        ("video_completion_rate", -0.01),
        ("video_completion_rate", 1.01),
        ("forum_posts_count", -1),
        ("assignment_submission_rate", -0.01),
        ("assignment_submission_rate", 1.01),
    ],
)
def test_out_of_range_field_rejected(field, bad_value):
    payload = dict(SAMPLE_TELEMETRY, **{field: bad_value})
    with pytest.raises(ValidationError):
        LearnerTelemetry(**payload)


def test_in_range_payload_accepted():
    telemetry = LearnerTelemetry(**SAMPLE_TELEMETRY)
    assert telemetry.quiz_avg_score == SAMPLE_TELEMETRY["quiz_avg_score"]
