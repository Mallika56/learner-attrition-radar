"""Feature pipeline for learner-attrition-radar.

Six behavioral-telemetry features drive the dropout-risk model. All of them are
things a self-paced course platform can log without any extra instrumentation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "avg_session_length_minutes",
    "days_since_last_login",
    "quiz_avg_score",
    "video_completion_rate",
    "forum_posts_count",
    "assignment_submission_rate",
]


def validate_features(payload: dict) -> pd.DataFrame:
    """Order and validate a raw feature dict into a one-row DataFrame the model expects."""
    row = {name: float(payload[name]) for name in FEATURE_NAMES}
    return pd.DataFrame([row], columns=FEATURE_NAMES)


def generate_synthetic_dataset(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic learner telemetry with a plausible dropout label.

    This is synthetic data, not real learner records — see the README's Model
    section. Higher engagement (longer sessions, recent logins, strong quiz
    scores, video/assignment completion, forum participation) pushes dropout
    probability down; the label is sampled with noise so the classes overlap
    the way a real behavioral dataset would.
    """
    rng = np.random.default_rng(seed)

    avg_session_length_minutes = rng.gamma(shape=2.0, scale=12.0, size=n_rows)
    days_since_last_login = rng.exponential(scale=6.0, size=n_rows)
    quiz_avg_score = np.clip(rng.normal(loc=72, scale=18, size=n_rows), 0, 100)
    video_completion_rate = np.clip(rng.beta(a=2.0, b=2.0, size=n_rows), 0, 1)
    forum_posts_count = rng.poisson(lam=1.5, size=n_rows)
    assignment_submission_rate = np.clip(rng.beta(a=2.5, b=1.8, size=n_rows), 0, 1)

    z = (
        -0.04 * avg_session_length_minutes
        + 0.12 * days_since_last_login
        - 0.025 * quiz_avg_score
        - 1.8 * video_completion_rate
        - 0.15 * forum_posts_count
        - 2.1 * assignment_submission_rate
        + 2.0
    )
    dropout_probability = 1 / (1 + np.exp(-z))
    label = rng.binomial(1, dropout_probability)

    return pd.DataFrame(
        {
            "avg_session_length_minutes": avg_session_length_minutes,
            "days_since_last_login": days_since_last_login,
            "quiz_avg_score": quiz_avg_score,
            "video_completion_rate": video_completion_rate,
            "forum_posts_count": forum_posts_count,
            "assignment_submission_rate": assignment_submission_rate,
            "dropped_out": label,
        }
    )
