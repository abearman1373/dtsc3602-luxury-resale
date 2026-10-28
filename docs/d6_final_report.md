# D6 Final Report — Luxury Resale Value Retention for SouthPark Mall Brands

DTSC 3602, Fall 2026

## 1. Problem and domain rationale
Secondhand luxury goods ("resale") are a fast-growing, opaque market: a shopper deciding whether to buy new
or resale, or a seller deciding where to list an item, has no easy way to know whether a given asking price
is typical for the item's brand, condition, and age. Rebag is a large resale platform that publishes asking
prices and, for a subset of listings, its own estimate of current retail price — giving a public, computable
measure of how steeply an item is discounted relative to buying new. We chose SouthPark Mall (Charlotte, NC)
brands specifically so the analysis stays grounded in a concrete, locally relevant retail footprint rather
than an abstract "luxury goods" category.

**Who this serves**: a prospective buyer deciding whether a listed price looks fair, and a prospective
seller deciding where to price an item relative to comparable listings — both decisions this project's final
application supports directly.

## 2. Research question
> How accurately can brand, item type, condition, and age predict the asking-price-to-estimated-retail-price
> ratio of luxury goods from SouthPark Mall brands listed on Rebag, compared with a brand-level median
> benchmark?

The target (`asking_price_ratio = asking_price / estimated_retail_price × 100`) is built entirely from
information available **before** a listing is priced — nothing in the feature set is derived from the
asking price itself (see `docs/data_dictionary.md`'s Role column).

## 3. Data
- **Source**: Rebag's public Shopify product feed (`shop.rebag.com/products.json`), permitted by its
  `robots.txt`. 25,000 listings collected October 20, 2026 (`scripts/crawl_shopify.py`,
  `scripts/parse_rebag.py` — D1/D2).
- **Scope**: 15,888 listings from brands with a SouthPark Mall store; of those, 2,317 carry Rebag's estimated
  retail price and so have a usable target. After dropping 1 listing with an implausible ratio (<5%) and 12
  "Outlet"-condition listings, **2,304 rows** remain for modeling.
- **Split**: 1,843 train/validation rows and a **461-row reserved test set**, split by item model (so the
  same bag never appears in both sides) and stratified by brand (`scripts/prepare_data.py`, D2). The test set
  was not opened until §6 of this report.
- **Known limitation carried through every stage**: every single labeled row is a Rebag "dropship-partner"
  listing (0 non-partner listings have a retail price) — see §7.1.

Full column-by-column documentation: `docs/data_dictionary.md`.

## 4. Methods
### 4.1 Features (D3)
Eight predictors, all known before a listing is priced: `brand_group`, `item_group`, `product_type`,
`material_group`, `age_band` (categorical), and `condition_rank`, `log_retail_price`, `n_extras`,
`age_censored`, plus four boolean extras flags (numeric). `exterior_color`, `hardware_color`, and `size` were
excluded for missingness and inconsistent formatting (see `docs/d3_model_evaluation.md`).

### 4.2 Custom transformer
`scripts/transformers.py::RareCategoryGrouper` folds any categorical value seen fewer than 10 times in
training — including values never seen at all — into `"Other"`. Fit on training data only, in every CV fold
and in the final pipeline. Applied mainly to `product_type` (25 raw values, several with n≤3).

### 4.3 Validation design
Five `cv_fold` folds grouped by `model_group` (brand+model) and stratified by `brand_group`
(`StratifiedGroupKFold`, seed 3602, built in D2). Grouping prevents near-duplicate listings of the same item
from straddling a train/validation split; stratification keeps each fold's brand mix representative. All
preprocessing (the custom transformer, one-hot encoding, scaling) is fit fresh inside each fold's training
split only, inside a single `sklearn.Pipeline` — never on validation or test data.

### 4.4 Models compared (D3)
| Model | MAE (5-fold CV) | R² |
|---|---|---|
| Baseline: brand-group median | 12.00 ± 0.52 | −0.02 ± 0.03 |
| HistGradientBoostingRegressor | 10.34 ± 0.41 | 0.24 ± 0.10 |
| **Ridge (α=10, grid-selected)** | **9.96 ± 0.45** | **0.31 ± 0.09** |

Ridge was selected: lowest mean CV MAE, ahead of HistGradientBoosting in 4 of 5 folds. Full comparison and
per-group error analysis: `docs/d3_model_evaluation.md`.

### 4.5 Fitted pipeline and serving (D4)
The selected Ridge pipeline was refit on the **full** train_val set (1,843 rows; test still untouched) and
saved as `models/pipeline.joblib` with `models/metadata.json` (training provenance, feature list,
scikit-learn version). Served via FastAPI (`api/main.py`), loaded once per process, returning 422 on invalid
input and 503 if the artifact is unavailable — both verified. Deployed on Modal with scikit-learn/pandas/
numpy pinned to the exact fitting versions; local and deployed `/predict` return byte-identical output for
the same input (`docs/d4_pipeline_api.md`).

### 4.6 Application (D5)
A static frontend (`web/index.html`, deployed on Vercel) calls the live Modal API directly from the browser,
with working loading and error states. A Postman collection (`postman/`) exercises health, info, a valid
prediction, and two invalid-input cases against the deployed API — 10/10 assertions pass
(`docs/d5_application_tests.md`).

## 5. Interpretation guidance
MAE (mean absolute error, in percentage points of the ratio) is the primary metric because the eventual
use — "is this asking price roughly typical?" — needs an answer in the same units a person reasons in
("off by about X points"), not a squared-error unit. RMSE is reported alongside to surface whether errors
are dominated by a few large misses (RMSE ≫ MAE would indicate that; here RMSE/MAE ≈ 1.2, so errors are
fairly evenly sized, not dominated by outliers). R² indicates what share of ratio variation the model
explains, for context — not the decision metric. **Prediction does not establish causation**: a strong
"condition → ratio" relationship in this data describes Rebag's pricing pattern, not a general causal "being
in better condition causes a higher resale value" claim beyond this dataset.

## 6. Final results (test holdout, opened for the first time — `scripts/evaluate_final.py`)
| Model | n | MAE | RMSE | R² |
|---|---|---|---|---|
| Baseline (brand median) | 461 | 13.02 | 16.03 | −0.07 |
| **Ridge (frozen, D4 artifact)** | 461 | **10.99** | 13.50 | 0.245 |

**The model beats the baseline on held-out data it never influenced in any way** — a ~2-point MAE
improvement and R² moving from effectively zero to 0.245, consistent in direction and magnitude with the D3
cross-validation estimate (MAE 9.96). The CV-to-test gap (+1.03 MAE) is modest and in the expected direction
for a held-out set that's smaller and was never used for any tuning decision — not evidence of serious
overfitting.

**Answering the research question**: yes — brand, item type, condition, and age predict the SouthPark/Rebag
asking-price ratio meaningfully better than a brand-level median benchmark, on data the model never saw
during development. The improvement is real but modest (R² 0.245): most of the remaining variation is
plausibly driven by factors this dataset doesn't capture — exact wear beyond a 6-level grade, Rebag's
internal pricing logic, and day-to-day market timing.

## 7. Limitations, usefulness, and bias/fallacy checks

### 7.1 Selection bias: every label comes from one listing channel
100% of labeled rows (2,317 of 2,317) are Rebag "dropship-partner" listings; **zero** non-partner listings
carry a retail price (documented in D2's `reports/audit_summary.txt` and surfaced on the dashboard's
"Coverage & bias" tab). **Check performed**: this was computed directly from the data (not assumed) and
stated in every downstream document. **Consequence**: this model's predictions describe dropship-partner-
style listings specifically. Generalizing to Rebag's non-partner inventory, or to other resale platforms,
is not supported by this data and isn't claimed.

### 7.2 Small-subgroup fallacy ("law of small numbers")
The D6 by-group table shows some brand/item combinations with large, even *negative*, improvements over
baseline — e.g. Tiffany & Co. on the test set (n=8) got **worse** by 5.9 MAE points, and Jewelry & Watch
(n=11) worse by 3.9. **Check performed**: every group table in this project (D2 dashboard, D3, D6) reports
`n` alongside every metric specifically so a reader doesn't mistake one of these small, noisy subgroups for
a reliable finding. **Consequence for use**: treat per-brand or per-category claims below roughly n=30 as
indicative, not decisive — the headline ~2-point MAE improvement is the trustworthy number; a single small
subgroup moving the "wrong" way is expected noise, not a contradiction of it.

### 7.3 Snapshot/survivorship bias in the age-related finding
The source data is a **single-day inventory snapshot** of listings still for sale, not a record of
completed sales. D2's EDA found older listings priced *lower*, not higher — but an older listing still
present in an October 2026 snapshot is, by definition, one that **hasn't sold yet**; a fast-selling (likely
well-priced) older item would already be gone and invisible to this dataset. **Check performed**: this
caveat is stated directly on the dashboard figure it affects ("Older listings are priced lower, not higher")
and repeated here rather than treated as a settled causal finding.

### 7.4 Other limitations
- No deep hyperparameter search (a 5-value grid for Ridge; light manual tuning for HistGradientBoosting) —
  the gap between candidate models could shift with a larger search budget.
- Jewelry & Watch is thin throughout (11–43 rows depending on the split) relative to Handbag (>1,700);
  conclusions mostly describe handbags, which the D2 coverage analysis flags explicitly.
- The target is an **advertised asking price**, not a completed sale — this model predicts how items are
  *listed*, not what buyers actually pay.

### 7.5 Practical usefulness
Despite these limitations, the frozen model gives a buyer or seller a concrete, better-than-naive estimate —
roughly 11 percentage points of accuracy on an unseen item, versus roughly 13 for "just use the brand's
median" — translatable into a dollar ballpark via the live application (D5). That is a modest but genuine
and honestly-measured improvement, delivered end-to-end as a working, tested, publicly reachable
application.

## 8. Reproducibility
| Step | Command |
|---|---|
| Install | `uv sync` |
| D2 data prep + EDA | `uv run python scripts/prepare_data.py && uv run python scripts/make_plots.py` |
| D3 model evaluation | `uv run python scripts/evaluate_models.py` |
| D4 fit + save pipeline | `uv run python scripts/train_final_pipeline.py` |
| D4 API (local) | `uv run uvicorn api.main:app --port 8000` |
| D6 final test evaluation | `uv run python scripts/evaluate_final.py` |

Live URLs:
- Dashboard (D2): https://abearman1373--luxury-resale-dashboard-run.us-east.modal.run
- API (D4): https://abearman1373--luxury-resale-api-fastapi-app.us-east.modal.run (docs at `/docs`)
- Frontend (D5): https://web-fawn-nine-hzedu95igg.vercel.app
- Repository: https://github.com/abearman1373/dtsc3602-luxury-resale

No credentials are committed; `.env` (Supabase keys) and Modal secrets stay local/account-scoped per
`.gitignore`.

## 9. AI assistance and contributions disclosure
Data collection, cleaning, EDA, modeling, the API, the frontend, and this report were developed with Claude
(Anthropic), within the course AI policy, across D1–D6. *[Group members: replace this line with each
teammate's name and specific contribution — e.g. who reviewed/ran which code, who wrote which report
section — before submitting; this was generated in a single assisted session and the actual review/division
of work needs to be filled in by the team.]*
