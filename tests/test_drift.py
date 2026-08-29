from app.drift import compute_psi_report


def _baseline_with_bins(bins):
    return {"_bins": {"quiz_avg_score": bins}}


def test_psi_is_near_zero_for_identical_distribution():
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # One value landing in each of the 10 bins matches the uniform baseline exactly.
    recent = {"quiz_avg_score": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]}
    report = compute_psi_report(recent, _baseline_with_bins(bins))
    assert report["features"]["quiz_avg_score"]["status"] == "stable"
    assert report["overall_status"] == "stable"


def test_psi_flags_significant_shift_when_all_mass_in_one_bin():
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    recent = {"quiz_avg_score": [95] * 50}
    report = compute_psi_report(recent, _baseline_with_bins(bins))
    assert report["features"]["quiz_avg_score"]["psi"] > 0.25
    assert report["overall_status"] == "significant_shift"


def test_empty_recent_features_reports_insufficient_data_upstream():
    report = compute_psi_report({}, _baseline_with_bins([0, 50, 100]))
    assert report["overall_psi"] == 0.0
    assert report["overall_status"] == "stable"
