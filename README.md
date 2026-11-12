# CR03 - Calibration and fairness audit (German Credit)

Checks whether the CR01 scorecard's probabilities are well calibrated, tests Platt and isotonic
recalibration, and audits approval and error rates by Sex and age band. It also checks whether
a feature (Housing) carries Sex information.

Notebook: [notebooks/03_calibration_fairness.ipynb](notebooks/03_calibration_fairness.ipynb)
(source: `03_calibration_fairness.py`, percent format). Figures are in `reports/figures/`.

## Data and split

- German Credit (`kabure/german-credit-data-with-risk`), 1,000 rows, target `Risk` (1 = good).
- No date column, so the split is random and stratified (seed 42), not time-based.
- Test set: the CR01 holdout (30%, n = 300).
- The 70% training part is split again into a model-fit part (n = 490) and a calibration part
  (n = 210). The calibrators never see test data.

## Results

**Calibration (test set, n = 300)**

| variant  | Brier  | ECE    | mean predicted good | observed good | Gini   |
|----------|--------|--------|---------------------|---------------|--------|
| raw      | 0.1806 | 0.0249 | 0.714               | 0.70          | 0.472  |
| platt    | 0.1820 | 0.0625 | 0.712               | 0.70          | 0.472  |
| isotonic | 0.1885 | 0.0816 | 0.706               | 0.70          | 0.452  |

The raw scorecard is the best calibrated here. Platt and isotonic, fit on 210 rows, add noise
instead of correcting bias. Platt keeps the ranking, so its Gini is unchanged. Isotonic creates
ties and loses some Gini.

**Fairness audit (cutoff PD 0.407, approves 70% of test applicants, raw PD)**

| group          | n   | approval rate | bad rate among approved |
|----------------|-----|---------------|-------------------------|
| female         | 96  | 0.71          | 0.26                    |
| male           | 204 | 0.70          | 0.17                    |
| under 25       | 47  | 0.53          | 0.32                    |
| 25 to 39       | 160 | 0.73          | 0.20                    |
| 40 and over    | 93  | 0.74          | 0.16                    |

Four-fifths screen (min/max ratio, flag below 0.8):

| attribute | measure                  | ratio | flag  |
|-----------|--------------------------|-------|-------|
| Sex       | approval rate            | 0.98  | no    |
| Age band  | approval rate            | 0.72  | yes   |
| Sex       | bad rate among approved  | 0.64  | yes   |
| Age band  | bad rate among approved  | 0.50  | yes   |

Interpretation:
- Approval rates are similar by Sex. The age gap in approval is expected, since age is a model input
  and the score uses it directly.
- Bad rate among approved applicants is higher for women and for under-25s. This is a
  calibration/error-rate disparity, and the cutoff does not equalise it. It needs investigation.
  The sample is small (96 women, 47 under 25), so the ratios carry wide uncertainty.

**Proxy check.** Housing and Sex are associated on the full German data (chi-squared 54.0,
Cramér's V 0.23, p < 0.001). Housing is a scorecard feature, so Sex can enter the model
indirectly even though Sex is excluded.

## Limitations

- 1,000 rows and a 300-row test set. Group estimates are noisy, and a four-fifths flag on a
  group of 47 is a screening signal, not a finding.
- Four-fifths is a screening heuristic from US employment practice, not a legal test for credit.
- The cutoff is a policy choice (approve 70%). Results depend on it.
- The proxy check uses a single categorical feature. Other features could also carry Sex.
- German Credit is an old, small dataset from one market. Conclusions do not transfer to
  present-day portfolios. The large-scale LendingClub check is in CR02.

## Reproduce

```bash
pip install -e ../credit-scoring-core -e ../scorecard-woe-logistic -e .
pytest
make notebook
```

Dependencies are pinned in `pyproject.toml`. The core and CR01 packages come from their public
GitHub repos.
