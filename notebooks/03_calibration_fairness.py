# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CR03 - Calibration and fairness audit (German Credit)
#
# Goal: (1) check whether the CR01 scorecard PDs are calibrated and whether Platt or isotonic
# calibration helps, and (2) audit approval and error rates by Sex and age band at a cutoff.
#
# Split: the CR01 holdout (30%, stratified, seed 42) is the test set. The 70% training part is
# split again into a model-fit part (70%) and a calibration part (30%). The calibrator never sees
# test data.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import brier_score_loss

from calibration_fairness.audit import (
    IsotonicCalibrator,
    PlattCalibrator,
    age_band,
    disparity,
    group_audit,
    split_fit_calibration,
)
from credit_core.metrics import expected_calibration_error, gini
from credit_core.plots import plot_reliability
from scorecard_woe.pipeline import FEATURES, GermanSplit, fit_scorecard, load_split

FIG = Path("../reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Fit, calibrate, test

# %%
split = load_split()
X_fit, X_cal, y_fit, y_cal = split_fit_calibration(split.X_train, split.y_train)
fit_split = GermanSplit(X_fit, split.X_test, y_fit, split.y_test)

sc, C = fit_scorecard(fit_split)
print(f"Tuned C = {C}, fit n = {len(X_fit)}, calibration n = {len(X_cal)}, test n = {len(split.X_test)}")

# Raw probability of good (y=1 means good) from the scorecard.
p_cal_raw = sc.predict_good(X_cal[FEATURES])
p_test_raw = sc.predict_good(split.X_test[FEATURES])

platt = PlattCalibrator().fit(p_cal_raw, y_cal)
iso = IsotonicCalibrator().fit(p_cal_raw, y_cal)

variants = {
    "raw": p_test_raw,
    "platt": platt.predict(p_test_raw),
    "isotonic": iso.predict(p_test_raw),
}

rows = []
for name, p in variants.items():
    rows.append(
        {
            "variant": name,
            "brier": brier_score_loss(split.y_test, p),
            "ece": expected_calibration_error(split.y_test, p, n_bins=10),
            "mean_pred_good": float(np.mean(p)),
            "observed_good": float(split.y_test.mean()),
            "gini": gini(split.y_test, p),
        }
    )
calib_table = pd.DataFrame(rows).set_index("variant").round(4)
calib_table

# %% [markdown]
# Gini is rank-based, so a monotone map (Platt) leaves it unchanged. Isotonic changes it because
# its steps create ties. On this test set the raw scorecard is already the best calibrated:
# both calibrators were fit on only 210 calibration rows and add noise rather than correction.
# Calibration is not applied to the audit's ranking below; the raw PD is used.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, p) in zip(axes, variants.items()):
    plot_reliability(split.y_test, p, n_bins=8, ax=ax)
    ax.set_title(f"Reliability on test: {name}")
fig.tight_layout()
fig.savefig(FIG / "cr03_reliability.png", dpi=120)
plt.show()

# %% [markdown]
# ## 2. Fairness audit by Sex and age band
#
# Approve when the PD is at or below a cutoff that approves the 70% of test applicants with the
# lowest PD. We use the raw scorecard PD, which was the best calibrated on the test set (section 1).

# %%
p_good = variants["raw"]
pd_test = 1.0 - p_good
cutoff = float(np.quantile(pd_test, 0.70))
print(f"PD cutoff (approve 70% of test): {cutoff:.4f}")

groups = split.X_test["Sex"].astype(str)
audit_sex = group_audit(split.y_test.values, p_good, groups, cutoff)
audit_sex

# %%
bands = age_band(split.X_test["Age"])
audit_age = group_audit(split.y_test.values, p_good, bands, cutoff)
audit_age

# %%
disp = pd.DataFrame(
    [
        {"attribute": "Sex", **disparity(audit_sex, "approval_rate")},
        {"attribute": "Age band", **disparity(audit_age, "approval_rate")},
        {"attribute": "Sex", **disparity(audit_sex, "bad_rate_approved")},
        {"attribute": "Age band", **disparity(audit_age, "bad_rate_approved")},
    ]
)
disp

# %% [markdown]
# `bad_rate_approved` is the bad rate among approved applicants. A large gap means the cutoff
# lets more bad applicants through in one group, which is a calibration/error-rate disparity.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, audit, title in [(axes[0], audit_sex, "by Sex"), (axes[1], audit_age, "by age band")]:
    x = np.arange(len(audit))
    w = 0.38
    ax.bar(x - w / 2, audit["approval_rate"], w, label="approval rate")
    ax.bar(x + w / 2, audit["bad_rate_approved"], w, label="bad rate among approved")
    ax.set_xticks(x)
    ax.set_xticklabels(audit.index)
    ax.set_ylim(0, 1)
    ax.set_title(f"Audit {title} (cutoff PD {cutoff:.2f})")
    ax.legend()
fig.tight_layout()
fig.savefig(FIG / "cr03_fairness_audit.png", dpi=120)
plt.show()

# %% [markdown]
# ## 3. Proxy check: is Housing associated with Sex?
#
# Sex is not in the model, but a feature can still carry it. Housing is a categorical feature in
# the scorecard, so we test its association with Sex (chi-squared and Cramér's V) on the full
# German data.

# %%
full = pd.concat([split.X_train, split.X_test])
ct = pd.crosstab(full["Housing"], full["Sex"])
chi2, p_val, dof, _ = chi2_contingency(ct)
n = ct.values.sum()
cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
proxy = pd.DataFrame([{"feature": "Housing", "chi2": chi2, "p_value": p_val, "cramers_v": cramers_v}])
proxy.round(4)

# %%
ct

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ct.div(ct.sum(axis=0), axis=1).plot(kind="bar", ax=ax)
ax.set_ylabel("share within Sex")
ax.set_title("Housing mix by Sex (full German data)")
fig.tight_layout()
fig.savefig(FIG / "cr03_proxy_housing.png", dpi=120)
plt.show()

# %% [markdown]
# ## 4. Summary
#
# - Calibration: the raw scorecard has the lowest ECE and the lowest Brier on the test set. Platt
#   and isotonic, fit on 210 rows, do not help here.
# - Fairness: approval rates by Sex are similar (ratio about 0.98). Approval by age band is not
#   (under 25 approved at 53%, against about 73% to 74% for the other bands). Bad rate among
#   approved applicants is higher for women (26%, against 17% for men) and for under-25s (32%).
#   Both bad-rate ratios fail the four-fifths screen. Age is a model input, so its approval gap is
#   expected from the score. The Sex gap in bad rate among approved is a warning to investigate.
#   With 300 test rows and 96 women, these ratios have wide uncertainty.
# - Proxy: Housing is strongly associated with Sex (Cramér's V about 0.23, p < 0.001). Housing
#   is a scorecard feature, so Sex can enter the model indirectly. This must be stated in the
#   README and checked in a larger dataset.

# %%
calib_table
