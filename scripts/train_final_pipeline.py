"""D4: fit the selected pipeline (Ridge, from D3) on the FULL train_val set
and save it as a versioned artifact for the API to serve.

The reserved test_holdout.csv is still never opened here — this is the
model that D6 will finally evaluate against it.

Usage:
    uv run python scripts/train_final_pipeline.py
Writes:
    models/pipeline.joblib
    models/metadata.json
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transformers import RareCategoryGrouper

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

CAT_COLS = ["brand_group", "item_group", "product_type", "material_group", "age_band"]
NUM_COLS = ["condition_rank", "log_retail_price", "n_extras",
            "age_censored", "has_dust_bag", "has_strap", "has_auth_card", "has_box"]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "asking_price_ratio"
ALPHA = 10.0  # selected in D3 — see reports/d3_summary.txt


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", Pipeline([("rare_group", RareCategoryGrouper(min_count=10)),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=ALPHA, random_state=3602))])


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    tv = pd.read_csv(ROOT / "data" / "clean" / "train_val.csv")
    for c in ["age_censored", "has_dust_bag", "has_strap", "has_auth_card", "has_box"]:
        tv[c] = tv[c].astype(int)

    pipe = build_pipeline()
    pipe.fit(tv[FEATURES], tv[TARGET])

    MODELS.mkdir(exist_ok=True)
    import joblib
    joblib.dump(pipe, MODELS / "pipeline.joblib")

    d3_summary = (ROOT / "reports" / "d3_summary.txt")
    cv_mae = None
    if d3_summary.exists():
        for line in d3_summary.read_text().splitlines():
            if line.strip().startswith("ridge"):
                cv_mae = line.split()[1]

    meta = {
        "model_type": "Ridge",
        "alpha": ALPHA,
        "selected_in": "D3 (reports/d3_model_evaluation... see docs/d3_model_evaluation.md)",
        "cv_mean_mae_from_d3": "see docs/d3_model_evaluation.md (~9.96)",
        "features": {"categorical": CAT_COLS, "numeric": NUM_COLS},
        "target": TARGET,
        "trained_on": "data/clean/train_val.csv (1,843 rows; test_holdout.csv never used)",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "git_commit": git_commit(),
        "training_rows": len(tv),
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Saved models/pipeline.joblib and models/metadata.json")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
