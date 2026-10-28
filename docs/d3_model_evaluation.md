# D3: Model Evaluation

## Research question
How accurately can brand, item type, condition, and age predict the asking-price-to-estimated-retail-price
ratio of luxury goods from SouthPark Mall brands listed on Rebag, compared with a brand-level median
benchmark?

## Data and validation design
- 2,304 labeled SouthPark listings from `data/clean/train_val.csv` + `test_holdout.csv` (D2). This note uses
  **`train_val.csv` only** (1,843 rows) — the 461-row `test_holdout.csv` is reserved for D6 and is never
  opened here.
- **Validation**: the 5 `cv_fold` folds built in D2 (`scripts/prepare_data.py`) via `StratifiedGroupKFold`,
  grouped by `model_group` (brand + model, lower-cased) and stratified by `brand_group`. Grouping means no
  listing of the same item model ever appears in both the training and validation side of a fold — without
  it, near-duplicate listings of the same bag would leak across the split and overstate accuracy.
- **No leakage**: every preprocessing step (the custom rare-category grouper, the one-hot encoder, the
  feature scaler) is fit fresh inside each fold's training split only, inside the scikit-learn `Pipeline`,
  and applied to that fold's validation split. Nothing is fit on validation or test data at any point.

## Features
| Role | Columns |
|---|---|
| Categorical | `brand_group`, `item_group`, `product_type`, `material_group`, `age_band` |
| Numeric | `condition_rank`, `log_retail_price`, `n_extras`, `age_censored`, `has_dust_bag`, `has_strap`, `has_auth_card`, `has_box` |

All eight are flagged `Predictor` in the D2 data dictionary — known before a listing is priced, never derived
from `asking_price` itself. `exterior_color`, `hardware_color`, and `size` were excluded: each is missing on
2–22% of rows, high-cardinality, and inconsistently formatted, and preliminary checks showed no meaningful
MAE improvement from including them.

**Custom transformer** (`scripts/transformers.py::RareCategoryGrouper`): `product_type` has 25 raw values,
several appearing only 1–3 times in training (e.g. a single "Hats & Hair Accessories" row). Fit only on the
training fold, it folds any category seen fewer than 10 times — including a type never seen at all — into
"Other", so one-hot encoding stays stable and an unseen product type at serving time degrades gracefully
instead of erroring.

## Models compared
| Model | Notes |
|---|---|
| **Baseline** | Brand-group median `asking_price_ratio` from the training fold (the benchmark named in the project's research question); falls back to the overall training median for an unseen brand. |
| **Ridge regression** | `RareCategoryGrouper` → one-hot (categorical) / `StandardScaler` (numeric) → `Ridge`. Alpha chosen from {0.3, 1, 3, 10, 30} by mean 5-fold CV MAE *before* the final comparison (alpha = 10). |
| **HistGradientBoostingRegressor** | Same preprocessing pipeline; `max_depth=6, learning_rate=0.08, max_iter=300, l2_regularization=0.1` (not deeply tuned — see Limitations). |

## Results (5-fold CV, mean ± std)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (brand median) | 12.00 ± 0.52 | 14.88 ± 0.55 | −0.02 ± 0.03 |
| HistGradientBoosting | 10.34 ± 0.41 | 12.78 ± 0.41 | 0.24 ± 0.10 |
| **Ridge (α=10)** | **9.96 ± 0.45** | **12.17 ± 0.43** | **0.31 ± 0.09** |

**Selected model: Ridge regression.** It has the lowest mean MAE and is consistently ahead of
HistGradientBoosting in 4 of 5 folds (full per-fold numbers: `reports/d3_model_comparison.csv`). With under
1,900 training rows and several sparse categorical levels, the regularized linear model generalizes better
than the boosted trees, which likely need either more data or real hyperparameter tuning to pull ahead — a
reasonable result, not a surprising one, for a dataset this size. Both models beat the baseline by a clear,
consistent margin (≈2 points of MAE, R² moving from ~0 to 0.31), so **the research question's premise holds**:
brand + item type + condition + age predict the ratio noticeably better than a brand-median benchmark alone.

## Error analysis (out-of-fold predictions, `reports/d3_error_analysis.csv`)
- Ridge improves on the baseline for almost every brand, item group, age band, and condition — by 1–5 points
  of MAE in most buckets.
- **Two groups get worse, not better**: Balenciaga (n=72, MAE +0.5) and "Other SouthPark" (n=14, MAE +0.5).
  Both are small groups where the baseline's own brand-median is already a tight fit (little within-brand
  variance to explain), so a shared-coefficient model trades a bit of per-brand accuracy for the gains
  elsewhere. "Other SouthPark" at n=14 is also simply too small to trust either number much.
- The largest gains are in older items (age band 11-20: −3.2 MAE, 21+: −3.2 MAE) and "Good" condition
  (n=25, −5.1 MAE, though that n is small enough to treat as indicative rather than solid).
- Condition "Pristine" (n=41) sees almost no improvement (−0.06): Pristine listings already sit in a narrow,
  predictable price band close to their brand median, so there's little left for the model to explain.

## Limitations
- **No deep hyperparameter search.** Ridge's alpha came from a 5-value grid; HistGradientBoosting used
  reasonable defaults with light manual tuning, not a systematic search (e.g. `RandomizedSearchCV`). The
  gap between the two models could narrow or reverse with more tuning budget.
- **R² of 0.31 leaves most variance unexplained.** The target is Rebag's *advertised* ratio, not a completed
  sale — unobserved factors (exact wear condition beyond the 6-level grade, market timing, Rebag's own
  internal pricing algorithm) aren't in the data and can't be modeled from it.
- **Small subgroups are noisy.** Several brand/condition combinations above have n below 30; their MAE
  deltas are directional, not statistically solid.
- **Jewelry & Watch is thin** (n=43 in the error table) relative to Handbag (n=1,710) — per the D2 coverage
  analysis, conclusions mostly describe handbags, and that imbalance carries into D3's per-group numbers.

The selected Ridge pipeline is carried into D4 for the fitted, served pipeline.
