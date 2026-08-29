"""Population Stability Index (PSI) drift tracking.

Compares the distribution of features seen in recent predictions against the
training-time baseline. PSI < 0.1 is stable, 0.1-0.25 is a moderate shift
worth watching, > 0.25 is a significant shift the model should be retrained
against.
"""

from __future__ import annotations

import numpy as np

PSI_STABLE = 0.1
PSI_MODERATE = 0.25


def _psi_for_feature(actual_values: list[float], bins: list[float]) -> float:
    bins = sorted(set(bins))
    if len(bins) < 3 or not actual_values:
        return 0.0

    expected_share = 1.0 / (len(bins) - 1)
    actual_counts, _ = np.histogram(actual_values, bins=bins)
    actual_share = actual_counts / max(len(actual_values), 1)

    # Floor both shares so a bucket with zero observations doesn't blow up log(0).
    eps = 1e-4
    expected = np.full_like(actual_share, expected_share, dtype=float)
    expected = np.clip(expected, eps, None)
    actual_share = np.clip(actual_share, eps, None)

    return float(np.sum((actual_share - expected) * np.log(actual_share / expected)))


def compute_psi_report(
    recent_features: dict[str, list[float]], baseline_distribution: dict
) -> dict:
    """Return per-feature PSI plus an overall status label."""
    report = {}
    for feature_name, values in recent_features.items():
        bins = baseline_distribution.get(feature_name, {})
        if feature_name == "_bins" or not values:
            continue
        feature_bins = baseline_distribution.get("_bins", {}).get(feature_name)
        if not feature_bins:
            continue
        psi = _psi_for_feature(values, feature_bins)
        if psi < PSI_STABLE:
            status = "stable"
        elif psi < PSI_MODERATE:
            status = "moderate_shift"
        else:
            status = "significant_shift"
        report[feature_name] = {"psi": round(psi, 4), "status": status}

    overall_psi = max((v["psi"] for v in report.values()), default=0.0)
    return {
        "features": report,
        "overall_psi": round(overall_psi, 4),
        "overall_status": (
            "stable"
            if overall_psi < PSI_STABLE
            else "moderate_shift"
            if overall_psi < PSI_MODERATE
            else "significant_shift"
        ),
    }
