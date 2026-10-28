# Luxury Goods: Asking Prices vs Retail for SouthPark Mall Brands on Rebag

DTSC 3602 (Fall 2026) group project.

**Research question:** How accurately can brand, item type, condition, and age predict the asking-price-to-estimated-retail-price ratio of luxury goods from SouthPark Mall brands listed on Rebag, compared with a brand-level median benchmark?

The target is based on Rebag's **advertised asking prices**, not completed sales, so it describes how items are listed, not what buyers paid.

## Dataset

- **Preview in the browser:** [`data/sample_200_rows.csv`](data/sample_200_rows.csv), 200 SouthPark-brand rows that GitHub shows as a table.
- **Full dataset (6.8 MB):** [download `rebag_all_listings.csv`](https://raw.githubusercontent.com/ebut11/dtsc3602-luxury-resale/main/data/rebag_all_listings.csv). GitHub can't preview a file this large, so it downloads instead. It opens in Excel.

[`data/rebag_all_listings.csv`](data/rebag_all_listings.csv) has 25,000 resale listings collected on **October 20, 2026** from Rebag's public product feed ([shop.rebag.com/products.json](https://shop.rebag.com/products.json)). The site's `robots.txt` does not block this feed. The file is a snapshot of every listing for sale that day. About 88% were first posted in September–October 2026, and the rest were posted between August 2024 and August 2026.

| Count | Listings |
| --- | --- |
| All listings | 25,000 |
| Brands sold at SouthPark (`southpark_brand = True`) | 15,888 |
| SouthPark brands with an estimated retail price | 2,317 |
| SouthPark brands with a decoded production year | 5,197 |

### Columns

| Column | Description |
| --- | --- |
| `listing_id` | Rebag product ID |
| `brand` | Designer |
| `southpark_brand` | True if the brand has a store at SouthPark Mall |
| `model` | Item model name |
| `item_type` | Handbag, Jewelry, Accessory, Shoes, Apparel, Watch |
| `product_type` | Rebag's finer category (Totes, Wallets, ...) |
| `condition` | Rebag grade: Fair, Good, Very Good, Great, Excellent, Pristine |
| `asking_price` | Rebag's advertised asking price, USD (not a completed sale) |
| `estimated_retail_price` | Rebag's estimated retail price, USD (blank when not listed) |
| `asking_price_ratio` | **Target:** asking_price / estimated_retail_price x 100, as a percent (e.g. 65.0 = listed at 65% of estimated retail) |
| `production_year` | Decoded from Louis Vuitton date codes, Chanel serial numbers ([Rebag's chart](https://rebag.com/magazine/chanel-101-serial-codes)) or Hermès stamps |
| `item_age_years` | 2026 minus production_year |
| `year_source` | How the year was decoded. "2021 or later" means a microchipped item, with 2021 used as its year |
| `brand_code` | Raw date code or serial from the listing |
| `exterior_material`, `exterior_color`, `hardware_color` | Item details |
| `accessories` | Extras included (dust bag, strap, authenticity card, ...). Blank means unknown, not none |
| `size` | Size tag, when given |
| `available` | Whether the item was still for sale at collection |
| `listed_at` | Date Rebag first posted the listing |
| `dropship_partner` | True for partner listings, which have shorter descriptions |
| `url` | Link to the listing |

### Known limitations

- Prices are asking prices, not final sale prices. Older listings may include markdowns.
- `estimated_retail_price` is Rebag's estimate and likely reflects today's retail price, not the original purchase price.
- Production year is only available for Louis Vuitton, Chanel and Hermès.
- Data is used for coursework only. Rebag is the source of all listing data.

## Reproducing the data

Requires Python 3 with pandas and xlsxwriter.

```
python scripts/crawl_shopify.py https://shop.rebag.com raw/rebag
python scripts/parse_rebag.py raw/rebag data
```

`crawl_shopify.py` downloads the feed 250 listings per page, waiting 1.5 seconds between requests. Shopify stops at page 100, so the crawl returns up to 25,000 listings. `parse_rebag.py` turns the raw pages into the CSV. A new crawl will return different listings, since Rebag's inventory changes daily.

## D2: data preparation, EDA and dashboard

| File | Purpose |
| --- | --- |
| `scripts/prepare_data.py` | Audit, clean, derive features, reserve the test set. Writes `data/clean/` and `reports/audit_summary.txt` |
| `scripts/make_plots.py` | The five EDA figures in `reports/figures/` (train/validation rows only) |
| `docs/data_dictionary.md` | Every cleaned column, its role (target / predictor / group / audit) and missingness |
| `sql/create_tables.sql` | Supabase tables + read-only policies |
| `scripts/load_supabase.py` | Uploads training rows and the SouthPark inventory (test rows are never uploaded) |
| `dashboard/app.py` | Streamlit dashboard reading from Supabase |
| `dashboard/modal_app.py` | Deploys the dashboard on Modal |

**Reserved test set:** `data/clean/test_holdout.csv` (461 rows, ~20%) is split by item model so the same model never appears in both sets, and stratified by brand. Nobody opens it until the D6 final evaluation.

### Run it (Windows PowerShell)

```powershell
uv sync
uv run python scripts/prepare_data.py
uv run python scripts/make_plots.py

# Supabase: run sql/create_tables.sql in the SQL editor, then
Copy-Item .env.example .env      # fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_ANON_KEY
uv run python scripts/load_supabase.py

# Dashboard locally
$env:SUPABASE_URL="https://<project>.supabase.co"; $env:SUPABASE_ANON_KEY="<anon key>"
uv run streamlit run dashboard/app.py

# Dashboard on Modal
uv run modal setup
uv run modal secret create supabase-dashboard SUPABASE_URL=https://<project>.supabase.co SUPABASE_ANON_KEY=<anon key>
uv run modal deploy dashboard/modal_app.py
```

## AI disclosure

The data collection, cleaning, D2 preparation, plotting and dashboard code were written with Claude (Anthropic), within the course AI policy. The group reviewed and ran all code.
