"""Calibration of a scorecard and group-level fairness audit."""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from credit_core import RANDOM_STATE

EPS = 1e-6


def split_fit_calibration(X: pd.DataFrame, y: pd.Series, calib_size: float = 0.3):
    """Stratified split of the training data into a model-fit part and a calibration part."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=calib_size, stratify=y, random_state=RANDOM_STATE
    )
    return X_fit, X_cal, y_fit, y_cal


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Logistic regression on the logit of the raw probability (fit on calibration data only)."""

    def fit(self, p_raw, y):
        self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(np.asarray(p_raw)).reshape(-1, 1), y)
        return self

    def predict(self, p_raw) -> np.ndarray:
        return self.model_.predict_proba(_logit(np.asarray(p_raw)).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone non-parametric map from raw probability to observed rate."""

    def fit(self, p_raw, y):
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(
            np.asarray(p_raw), np.asarray(y)
        )
        return self

    def predict(self, p_raw) -> np.ndarray:
        return self.model_.predict(np.asarray(p_raw))


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 25, 40, 200], labels=["<25", "25-39", "40+"], right=False).astype(str)


def group_audit(y_good: np.ndarray, p_good: np.ndarray, groups: pd.Series, cutoff_pd: float) -> pd.DataFrame:
    """Per-group approval rate, bad rate among approved, and true-positive rate for bad applicants.

    Approve if PD <= cutoff_pd. `y_good` is 1 for good, so bad = 1 - y_good.
    """
    df = pd.DataFrame(
        {
            "group": np.asarray(groups),
            "good": np.asarray(y_good).astype(int),
            "pd": 1.0 - np.asarray(p_good),
        }
    )
    df["approved"] = df["pd"] <= cutoff_pd
    df["bad"] = 1 - df["good"]
    rows = []
    for g, sub in df.groupby("group"):
        approved = sub[sub["approved"]]
        bads = sub[sub["bad"] == 1]
        rows.append(
            {
                "group": g,
                "n": len(sub),
                "bad_rate": sub["bad"].mean(),
                "approval_rate": sub["approved"].mean(),
                "bad_rate_approved": approved["bad"].mean() if len(approved) else np.nan,
                "bad_flagged_rate": (bads["pd"] > cutoff_pd).mean() if len(bads) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("group")


def disparity(audit: pd.DataFrame, col: str = "approval_rate") -> dict:
    """Min/max ratio for a rate column. Under the four-fifths rule, a ratio below 0.8 is a flag."""
    vals = audit[col].dropna()
    ratio = float(vals.min() / vals.max()) if vals.max() > 0 else np.nan
    return {"column": col, "min_max_ratio": ratio, "four_fifths_flag": bool(ratio < 0.8)}
