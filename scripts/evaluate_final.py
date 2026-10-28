"""D6: final evaluation of the FROZEN pipeline and baseline on the reserved
test holdout. This is the first and only time test_holdout.csv is opened —
no model, feature, or hyperparameter choice was made by looking at it.

Usage:
    uv run python scripts/evaluate_final.py
Writes:
    reports/d6_final_test_results.csv     baseline vs. model, overall + by group
    reports/d6_final_summary.txt
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

CAT_COLS = ["brand_group", "item_group", "product_type", "material_group", "age_band"]
NUM_COLS = ["condition_rank", "log_retail_price", "n_extras",
            "age_censored", "has_dust_bag", "has_strap", "has_auth_card", "has_box"]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "asking_price_ratio"


def metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "r2": r2_score(y_true, y_pred)}


def main():
    tv = pd.read_csv(ROOT / "data" / "clean" / "train_val.csv")
    test = pd.read_csv(ROOT / "data" / "clean" / "test_holdout.csv")
    for df in (tv, test):
        for c in ["age_censored", "has_dust_bag", "has_strap", "has_auth_card", "has_box"]:
            df[c] = df[c].astype(int)

    assert not set(tv.model_group) & set(test.model_group), "train/test model_group overlap — leakage!"
    print(f"Train/val: {len(tv)} rows | Test holdout: {len(test)} rows (opened for the first time)")

    pipe = joblib.load(ROOT / "models" / "pipeline.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    print(f"Loaded frozen pipeline: {meta['model_type']} (alpha={meta['alpha']}), "
          f"trained on {meta['training_rows']} rows at {meta['trained_at_utc']}")

    # Baseline: brand-group median from the FULL train_val set (same data the frozen
    # model was fit on) — never touches test for fitting, only for scoring.
    brand_medians = tv.groupby("brand_group")[TARGET].median()
    overall_median = tv[TARGET].median()
    test["pred_baseline"] = test.brand_group.map(brand_medians).fillna(overall_median)
    test["pred_model"] = pipe.predict(test[FEATURES])

    overall_rows = []
    for name, col in [("baseline_brand_median", "pred_baseline"), ("ridge_final", "pred_model")]:
        m = metrics(test[TARGET], test[col])
        overall_rows.append({"model": name, "n": len(test), **m})
    overall_df = pd.DataFrame(overall_rows)
    print("\n=== Final test results (overall) ===")
    print(overall_df.round(3).to_string(index=False))

    test["abs_err_baseline"] = (test[TARGET] - test.pred_baseline).abs()
    test["abs_err_model"] = (test[TARGET] - test.pred_model).abs()

    group_rows = []
    for group_col in ["brand_group", "item_group", "age_band", "condition"]:
        g = (test.groupby(group_col)
             .agg(n=("listing_id", "size"),
                  mae_baseline=("abs_err_baseline", "mean"),
                  mae_model=("abs_err_model", "mean"))
             .reset_index().rename(columns={group_col: "value"}))
        g.insert(0, "group", group_col)
        group_rows.append(g)
    group_df = pd.concat(group_rows, ignore_index=True)
    group_df["improvement"] = group_df.mae_baseline - group_df.mae_model

    print("\n=== Final test results (by group) ===")
    print(group_df.round(3).to_string(index=False))

    reports = ROOT / "reports"
    overall_df.to_csv(reports / "d6_final_test_overall.csv", index=False)
    group_df.to_csv(reports / "d6_final_test_by_group.csv", index=False)
    test[["listing_id", "brand_group", "item_group", "age_band", "condition", TARGET,
          "pred_baseline", "pred_model"]].to_csv(reports / "d6_final_test_predictions.csv", index=False)

    cv_mae_from_d3 = 9.96  # docs/d3_model_evaluation.md
    test_mae = overall_df.loc[overall_df.model == "ridge_final", "mae"].iloc[0]
    (reports / "d6_final_summary.txt").write_text(
        f"Test holdout: {len(test)} rows, opened for the first time in this script.\n"
        f"D3 CV MAE (train_val, 5-fold): {cv_mae_from_d3:.2f}\n"
        f"D6 final test MAE (ridge_final): {test_mae:.2f}\n"
        f"Gap (test - CV): {test_mae - cv_mae_from_d3:+.2f}\n\n"
        f"{overall_df.round(3).to_string(index=False)}\n"
    )
    print(f"\nCV MAE (D3) was {cv_mae_from_d3:.2f}; final test MAE is {test_mae:.2f} "
          f"(gap {test_mae - cv_mae_from_d3:+.2f}).")


if __name__ == "__main__":
    main()
