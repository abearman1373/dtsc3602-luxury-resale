# Data dictionary — cleaned modeling data

Files: `data/clean/train_val.csv` (1,843 rows) and `data/clean/test_holdout.csv` (461 rows, reserved for D6).
Built by `scripts/prepare_data.py` from `data/rebag_all_listings.csv` (Rebag public feed, collected 2026-10-20).
One row = one Rebag resale listing from a SouthPark Mall brand that shows an estimated retail price.

**Role** says how the column may be used in D3–D4:
**Target** = what we predict · **Predictor** = known before listing, allowed as model input · **Group/ID** = used for splitting or joining only · **Audit** = used to check data quality or bias, never as an input (it reveals or depends on the outcome, or isn't known to an app user).

| Column | Type | Role | Description / derivation | Missing (train_val) |
| --- | --- | --- | --- | --- |
| `asking_price_ratio` | float | **Target** | `asking_price / estimated_retail_price × 100`. Rebag's advertised asking price as a % of its estimated retail price. Not a completed-sale value. Range 11.3–83.3 | 0 |
| `asking_price` | float | Audit | Advertised asking price, USD. **Never an input** (it is the numerator of the target) | 0 |
| `estimated_retail_price` | float | Predictor | Rebag's estimate of retail price, USD. Likely today's retail, not the original price paid | 0 |
| `log_retail_price` | float | Predictor | Natural log of `estimated_retail_price` | 0 |
| `brand` | str | Predictor | Designer, as Rebag spells it | 0 |
| `brand_group` | str | Predictor / Group | `brand`, with brands under 20 labeled rows (Dolce & Gabbana, David Yurman, Omega, Breitling) pooled as "Other SouthPark". Used for the baseline and to stratify the split | 0 |
| `item_type` | str | Predictor | Rebag top category: Handbag, Accessory, Jewelry, Shoes, Watch | 0 |
| `item_group` | str | Predictor | `item_type` with Jewelry and Watch combined (too few watches) | 0 |
| `product_type` | str | Predictor | Rebag sub-category (Shoulder bags, Totes, Wallets, …) | 0 |
| `condition` | str | Predictor | Rebag grade: Fair, Good, Very Good, Great, Excellent, Pristine | 0 |
| `condition_rank` | int | Predictor | Ordinal 1 (Fair) – 6 (Pristine) | 0 |
| `production_year` | float | Predictor | Decoded from Louis Vuitton date codes or Chanel serials | 954 (51.8%) |
| `item_age_years` | float | Predictor | 2026 − `production_year` | 954 (51.8%) |
| `age_band` | str | Predictor | 0-5, 6-10, 11-20, 21+, or "Not available" | 0 |
| `age_available` | bool | Predictor | True when age could be decoded (Louis Vuitton and Chanel only) | 0 |
| `age_censored` | bool | Predictor | True for microchipped items (2021 or later); their age is set to 5 but could be less | 0 |
| `year_source` | str | Audit | How the year was decoded (LV date code, Chanel serial, microchip) | 954 |
| `model` | str | Group/ID | Rebag model name, e.g. "Neverfull Tote Damier MM" | 0 |
| `model_group` | str | Group/ID | lower-case `brand | model`. All listings of the same model stay in one split (prevents near-duplicates leaking into test) | 0 |
| `exterior_material` | str | Predictor | e.g. Leather, Monogram Canvas | 43 (2.3%) |
| `material_group` | str | Predictor | Top 8 materials, rest = "Other", missing = "Unknown" | 0 |
| `exterior_color`, `hardware_color` | str | Predictor (optional) | Item details | 9 / 81 |
| `accessories` | str | Predictor | Extras listed by Rebag; "No Accessories" when none | 0 |
| `has_dust_bag`, `has_strap`, `has_auth_card`, `has_box` | bool | Predictor | Parsed from `accessories` | 0 |
| `n_extras` | int | Predictor | Number of extras listed | 0 |
| `size` | str | Predictor (optional) | Size tag (PM, MM, Small, …) | 398 (21.6%) |
| `listed_at` | date | Audit | Date Rebag first posted the listing | 0 |
| `days_listed` | int | Audit | Days between `listed_at` and 2026-10-20. Not an input: a shopper predicting before purchase has no listing date | 0 |
| `is_recent` | bool | Audit | Posted Sep 1 – Oct 20, 2026 | 0 |
| `available` | bool | Audit | Still for sale at collection. False may mean sold or on hold; Rebag doesn't say, so it is **not** treated as a sale | 0 |
| `dropship_partner` | bool | Audit | Partner listing. True for every labeled row (see data note) | 0 |
| `listing_id`, `url` | int / str | Group/ID | Rebag product ID and link | 0 |
| `split` | str | Group/ID | `train_val` or `test` (≈20%, grouped by `model_group`, stratified by `brand_group`, seed 3602) | 0 |
| `cv_fold` | int | Group/ID | 0–4 within train_val, same grouping and stratification; −1 in test | 0 |

`data/clean/southpark_all.csv` holds all 15,888 SouthPark-brand listings (labeled or not) with the raw columns plus `has_target`, `is_recent` and `days_listed`. It is used only to compare the labeled sample with the full inventory.
