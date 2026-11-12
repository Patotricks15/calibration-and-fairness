import numpy as np
import pandas as pd
import pytest

from calibration_fairness.audit import (
    IsotonicCalibrator,
    PlattCalibrator,
    age_band,
    disparity,
    group_audit,
    split_fit_calibration,
)


def test_platt_recovers_known_miscalibration():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, size=20000)
    y = rng.uniform(size=20000) < true_p
    raw = np.clip(true_p ** 3 / (true_p ** 3 + (1 - true_p) ** 3), 1e-4, 1 - 1e-4)  # overconfident
    cal = PlattCalibrator().fit(raw, y.astype(int))
    fitted = cal.predict(raw)
    assert np.mean(np.abs(fitted - true_p)) < np.mean(np.abs(raw - true_p))


def test_isotonic_output_is_monotone_in_input():
    rng = np.random.default_rng(1)
    raw = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < raw).astype(int)
    iso = IsotonicCalibrator().fit(raw, y)
    grid = np.linspace(0, 1, 200)
    out = iso.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out >= 0) & (out <= 1))


def test_group_audit_counts_and_rates():
    y_good = np.array([1, 1, 0, 0, 1, 0])
    p_good = np.array([0.9, 0.8, 0.2, 0.3, 0.7, 0.1])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])
    audit = group_audit(y_good, p_good, groups, cutoff_pd=0.5)
    assert audit.loc["A", "n"] == 3
    assert audit.loc["A", "approval_rate"] == pytest.approx(2 / 3)
    assert audit.loc["B", "approval_rate"] == pytest.approx(1 / 3)


def test_disparity_flags_low_ratio():
    audit = pd.DataFrame({"approval_rate": [0.9, 0.5]}, index=["x", "y"])
    res = disparity(audit)
    assert res["min_max_ratio"] == pytest.approx(0.5 / 0.9)
    assert res["four_fifths_flag"] is True


def test_age_band_edges():
    bands = age_band(pd.Series([24, 25, 40, 41]))
    assert list(bands) == ["<25", "25-39", "40+", "40+"]


def test_split_fit_calibration_is_stratified_and_disjoint():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series([0, 1] * 50)
    Xf, Xc, yf, yc = split_fit_calibration(X, y, calib_size=0.3)
    assert len(Xc) == 30 and len(Xf) == 70
    assert set(Xf.index).isdisjoint(Xc.index)
