"""Upload cleaned data to Supabase (run after prepare_data.py and sql/create_tables.sql).

Needs two environment variables (put them in a .env file, never commit it):
    SUPABASE_URL          https://<project>.supabase.co
    SUPABASE_SERVICE_KEY  service_role key (write access) - used only here, never in the dashboard

Usage:
    uv run python scripts/load_supabase.py
"""
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
TRAIN_COLS = ["listing_id", "brand", "brand_group", "model", "item_type", "item_group", "product_type",
              "condition", "condition_rank", "item_age_years", "age_band", "age_available", "age_censored",
              "material_group", "n_extras", "has_dust_bag", "has_strap", "estimated_retail_price",
              "asking_price", "asking_price_ratio", "listed_at", "days_listed", "is_recent", "available",
              "cv_fold"]
INV_COLS = ["listing_id", "brand", "item_type", "condition", "has_target", "dropship_partner",
            "is_recent", "listed_at"]


def records(df):
    # to_json turns NaN into null and numpy types into plain JSON
    return json.loads(df.to_json(orient="records", date_format="iso"))


def upload(client, table, df, batch=500):
    rows = records(df)
    for i in range(0, len(rows), batch):
        client.table(table).upsert(rows[i:i + batch]).execute()
    print(f"{table}: {len(rows):,} rows")


def main():
    load_dotenv(ROOT / ".env")
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    tv = pd.read_csv(ROOT / "data" / "clean" / "train_val.csv")
    inv = pd.read_csv(ROOT / "data" / "clean" / "southpark_all.csv")
    assert (tv.split == "train_val").all(), "test rows must never be uploaded"
    upload(client, "listings_train", tv[TRAIN_COLS])
    upload(client, "southpark_inventory", inv[INV_COLS])


if __name__ == "__main__":
    main()
