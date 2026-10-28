"""D2 data preparation: audit, clean, and split the Rebag listings.

Usage:
    uv run python scripts/prepare_data.py

Reads  data/rebag_all_listings.csv  (raw snapshot collected 2026-10-20)
Writes data/clean/southpark_all.csv      every SouthPark-brand listing + audit flags
       data/clean/train_val.csv          modeling rows used for EDA, CV and model selection
       data/clean/test_holdout.csv       reserved final test rows (do not open until D6)
       reports/audit_summary.txt          counts behind every cleaning step

Target: asking_price_ratio = asking_price / estimated_retail_price x 100.
This is an ADVERTISED asking-price ratio, not a completed-sale value.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "rebag_all_listings.csv"
OUT = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
SEED = 3602

COLLECTION_DATE = "2026-10-20"
RECENT_START = "2026-09-01"          # listings first posted Sep-Oct 2026
MIN_PLAUSIBLE_RATIO = 5.0            # below 5% of retail = data-entry error
MIN_BRAND_ROWS = 20                  # smaller brands pooled into "Other SouthPark"

CONDITION_ORDER = ["Fair", "Good", "Very Good", "Great", "Excellent", "Pristine"]
ITEM_GROUPS = {"Handbag": "Handbag", "Accessory": "Accessory",
               "Jewelry": "Jewelry & Watch", "Watch": "Jewelry & Watch", "Shoes": "Shoes"}
EXTRAS = {"dust_bag": "Dust Bag", "strap": "Strap", "auth_card": "Authenticity Card", "box": "Box"}

log = []


def note(msg, n=None):
    line = f"{msg}: {n:,}" if n is not None else msg
    log.append(line)
    print(line)


def main():
    raw = pd.read_csv(RAW, encoding="utf-8-sig")
    note("Raw listings", len(raw))
    note("Duplicate listing_id", int(raw.listing_id.duplicated().sum()))

    sp = raw[raw.southpark_brand].copy()
    note("SouthPark-brand listings", len(sp))

    # ---- audit flags (kept for representativeness checks) ----
    sp["has_target"] = sp.estimated_retail_price.notna()
    sp["is_recent"] = sp.listed_at >= RECENT_START
    sp["days_listed"] = (pd.Timestamp(COLLECTION_DATE) - pd.to_datetime(sp.listed_at)).dt.days
    note("  with estimated retail price (labeled)", int(sp.has_target.sum()))
    note("  labeled rows that are dropship-partner listings",
         int((sp.has_target & sp.dropship_partner).sum()))
    note("  non-partner rows with a retail price", int((sp.has_target & ~sp.dropship_partner).sum()))

    sp.to_csv(OUT / "southpark_all.csv", index=False)

    # ---- modeling rows ----
    m = sp[sp.has_target].copy()

    bad = m.asking_price_ratio < MIN_PLAUSIBLE_RATIO
    note(f"Dropped: ratio below {MIN_PLAUSIBLE_RATIO}% (implausible retail price)", int(bad.sum()))
    m = m[~bad]

    outlet = m.condition == "Outlet"
    note("Dropped: condition 'Outlet' (not a wear grade)", int(outlet.sum()))
    m = m[~outlet]

    # recompute target from the two prices so it is traceable
    m["asking_price_ratio"] = (m.asking_price / m.estimated_retail_price * 100).round(2)

    # ---- derived features (all known before an item is listed) ----
    counts = m.brand.value_counts()
    m["brand_group"] = np.where(m.brand.map(counts) >= MIN_BRAND_ROWS, m.brand, "Other SouthPark")
    m["item_group"] = m.item_type.map(ITEM_GROUPS)
    m["condition_rank"] = m.condition.map({c: i + 1 for i, c in enumerate(CONDITION_ORDER)})
    m["age_available"] = m.item_age_years.notna()
    m["age_censored"] = m.year_source.fillna("").str.contains("2021 or later")
    m["age_band"] = pd.cut(m.item_age_years, [-1, 5, 10, 20, 100],
                           labels=["0-5", "6-10", "11-20", "21+"]).astype(object).fillna("Not available")
    acc = m.accessories.fillna("")
    for col, word in EXTRAS.items():
        m[f"has_{col}"] = acc.str.contains(word, case=False)
    m["n_extras"] = acc.apply(lambda s: 0 if s in ("", "No Accessories") else len(s.split(",")))
    m["log_retail_price"] = np.log(m.estimated_retail_price)
    m["material_group"] = m.exterior_material.fillna("Unknown")
    top_mat = m.material_group.value_counts().index[:8]
    m.loc[~m.material_group.isin(top_mat), "material_group"] = "Other"
    # near-identical items share a group so they never straddle train and test
    m["model_group"] = (m.brand + " | " + m.model).str.lower()

    note("Modeling rows after cleaning", len(m))
    note("Distinct brand+model groups", m.model_group.nunique())

    # ---- reserve final test data: ~20%, grouped by model, stratified by brand ----
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, test_idx = next(sgkf.split(m, m.brand_group, groups=m.model_group))
    m["split"] = "train_val"
    m.iloc[test_idx, m.columns.get_loc("split")] = "test"
    # 5 CV folds inside train_val, also grouped by model (used in D3)
    tv = m[m.split == "train_val"]
    m["cv_fold"] = -1
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                                .split(tv, tv.brand_group, groups=tv.model_group)):
        m.loc[tv.index[va], "cv_fold"] = k

    assert not set(m[m.split == "test"].model_group) & set(m[m.split == "train_val"].model_group)
    note("Train/validation rows", int((m.split == "train_val").sum()))
    note("Reserved test rows", int((m.split == "test").sum()))

    keep = ["listing_id", "brand", "brand_group", "model", "model_group", "item_type", "item_group",
            "product_type", "condition", "condition_rank", "production_year", "item_age_years",
            "age_band", "age_available", "age_censored", "year_source", "exterior_material",
            "material_group", "exterior_color", "hardware_color", "accessories", "has_dust_bag",
            "has_strap", "has_auth_card", "has_box", "n_extras", "size", "estimated_retail_price",
            "log_retail_price", "asking_price", "asking_price_ratio", "listed_at", "days_listed",
            "is_recent", "available", "dropship_partner", "url", "split", "cv_fold"]
    m = m[keep]
    m[m.split == "train_val"].to_csv(OUT / "train_val.csv", index=False)
    m[m.split == "test"].to_csv(OUT / "test_holdout.csv", index=False)
    (REPORTS / "audit_summary.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    main()
