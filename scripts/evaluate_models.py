"""D3 model evaluation: baseline vs. two candidate models.

Validation: the 5 grouped/stratified `cv_fold` folds already built into
train_val.csv by prepare_data.py (grouped by model_group, stratified by
brand_group) — so no listing of the same item model ever straddles a
train/validation split, and preprocessing (the custom transformer, the
scaler, the one-hot encoder) is fit fresh inside each fold's training
split only. The reserved test_holdout.csv is never opened here.

Usage:
    uv run python scripts/evaluate_models.py
Writes:
    reports/d3_model_comparison.csv   per-fold metrics, every model
    reports/d3_error_analysis.csv     out-of-fold MAE by group, baseline vs. selected model
    reports/d3_oof_predictions.csv    out-of-fold predictions (for the write-up's plots)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from transformers import RareCategoryGrouper

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

CAT_COLS = ["brand_group", "item_group", "product_type", "material_group", "age_band"]
NUM_COLS = ["condition_rank", "log_retail_price", "n_extras",
            "age_censored", "has_dust_bag", "has_strap", "has_auth_card", "has_box"]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "asking_price_ratio"
RIDGE_ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0]


def make_pipeline(model):
    pre = ColumnTransformer([
        ("cat", Pipeline([("rare_group", RareCategoryGrouper(min_count=10)),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "r2": r2_score(y_true, y_pred)}


def baseline_predict(train, val):
    """Brand-group median ratio, the benchmark named in the project README."""
    medians = train.groupby("brand_group")[TARGET].median()
    overall = train[TARGET].median()
    return val.brand_group.map(medians).fillna(overall).to_numpy()


def main():
    tv = pd.read_csv(ROOT / "data" / "clean" / "train_val.csv")
    for c in ["age_censored", "has_dust_bag", "has_strap", "has_auth_card", "has_box"]:
        tv[c] = tv[c].astype(int)

    folds = sorted(tv.cv_fold.unique())
    rows, oof = [], {}

    # ---- pick the Ridge alpha by mean CV MAE, same folds, before the final comparison ----
    print("Selecting Ridge alpha...")
    alpha_mae = {}
    for alpha in RIDGE_ALPHAS:
        maes = []
        for k in folds:
            tr, va = tv[tv.cv_fold != k], tv[tv.cv_fold == k]
            pipe = make_pipeline(Ridge(alpha=alpha, random_state=3602))
            pipe.fit(tr[FEATURES], tr[TARGET])
            maes.append(mean_absolute_error(va[TARGET], pipe.predict(va[FEATURES])))
        alpha_mae[alpha] = np.mean(maes)
        print(f"  alpha={alpha:>5}: mean CV MAE = {alpha_mae[alpha]:.3f}")
    best_alpha = min(alpha_mae, key=alpha_mae.get)
    print(f"Selected Ridge alpha = {best_alpha}\n")

    models = {
        "baseline_brand_median": None,
        "ridge": lambda: Ridge(alpha=best_alpha, random_state=3602),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.08, max_iter=300, l2_regularization=0.1,
            random_state=3602),
    }
    oof_pred = {name: np.full(len(tv), np.nan) for name in models}

    print("Running 5-fold grouped/stratified CV...")
    for k in folds:
        tr_idx = tv.cv_fold != k
        va_idx = tv.cv_fold == k
        tr, va = tv[tr_idx], tv[va_idx]

        preds = {"baseline_brand_median": baseline_predict(tr, va)}
        for name in ["ridge", "hist_gradient_boosting"]:
            pipe = make_pipeline(models[name]())
            pipe.fit(tr[FEATURES], tr[TARGET])
            preds[name] = pipe.predict(va[FEATURES])

        for name, pred in preds.items():
            oof_pred[name][va_idx.to_numpy()] = pred
            m = metrics(va[TARGET], pred)
            rows.append({"model": name, "fold": k, "n": len(va), **m})
            print(f"  fold {k}  {name:<24} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")

    cmp_df = pd.DataFrame(rows)
    summary = cmp_df.groupby("model")[["mae", "rmse", "r2"]].agg(["mean", "std"])
    print("\n=== 5-fold CV summary ===")
    print(summary.round(3).to_string())

    best_model = cmp_df[cmp_df.model != "baseline_brand_median"].groupby("model").mae.mean().idxmin()
    print(f"\nSelected model (lowest mean CV MAE, excluding baseline): {best_model}")

    # ---- error analysis: out-of-fold predictions, baseline vs. selected model ----
    tv["oof_baseline"] = oof_pred["baseline_brand_median"]
    tv["oof_selected"] = oof_pred[best_model]
    tv["abs_err_baseline"] = (tv[TARGET] - tv.oof_baseline).abs()
    tv["abs_err_selected"] = (tv[TARGET] - tv.oof_selected).abs()

    err_rows = []
    for group_col in ["brand_group", "item_group", "age_band", "condition"]:
        g = (tv.groupby(group_col)
             .agg(n=("listing_id", "size"),
                  mae_baseline=("abs_err_baseline", "mean"),
                  mae_selected=("abs_err_selected", "mean"))
             .reset_index().rename(columns={group_col: "value"}))
        g.insert(0, "group", group_col)
        err_rows.append(g)
    err_df = pd.concat(err_rows, ignore_index=True)
    err_df["improvement"] = err_df.mae_baseline - err_df.mae_selected

    REPORTS.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(REPORTS / "d3_model_comparison.csv", index=False)
    err_df.to_csv(REPORTS / "d3_error_analysis.csv", index=False)
    tv[["listing_id", "brand_group", "item_group", "age_band", "condition", TARGET,
        "oof_baseline", "oof_selected"]].to_csv(REPORTS / "d3_oof_predictions.csv", index=False)

    print("\n=== error analysis (out-of-fold MAE) ===")
    print(err_df.round(3).to_string())

    (REPORTS / "d3_summary.txt").write_text(
        f"Selected Ridge alpha: {best_alpha}\n"
        f"Selected model: {best_model}\n\n"
        f"5-fold CV summary (mean +/- std):\n{summary.round(3).to_string()}\n")


if __name__ == "__main__":
    main()
