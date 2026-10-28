"""Turn raw Rebag products.json pages into one tidy row per listing."""
import sys, os, re, json, glob, html
import pandas as pd

raw, out = sys.argv[1], sys.argv[2]

# Brands with a store at SouthPark (from the mall's Luxury page), as Rebag spells them
SOUTHPARK = {"Louis Vuitton", "Gucci", "Chanel", "Saint Laurent", "Balenciaga", "Burberry",
             "Dolce & Gabbana", "Golden Goose", "Tiffany & Co.", "David Yurman", "Rolex", "Omega",
             "TAG Heuer", "Tudor", "Breitling", "BOSS", "Hugo Boss", "St. John"}
FIELDS = ["Estimated Retail Price", "Condition", "Accessories", "Designer", "Model", "Exterior Material",
          "Exterior Color", "Interior Material", "Interior Color", "Hardware Color", "Item Number", "Brand Code"]

def body_fields(body):
    text = html.unescape(re.sub(r"<br\s*/?>|</p>", "\n", body))
    text = re.sub(r"<[^>]+>", "", text)
    found = {}
    for f in FIELDS:
        m = re.search(re.escape(f) + r":\s*(.*?)(?=\n|$|" + "|".join(re.escape(g) + ":" for g in FIELDS if g != f) + ")", text)
        if m:
            found[f] = m.group(1).strip(" .|")
    return found

# Chanel serial series -> first year of production (Rebag's chart: rebag.com/magazine/chanel-101-serial-codes)
CHANEL_7 = {0: 1986, 1: 1989, 2: 1991, 3: 1994, 4: 1996, 5: 1997, 6: 2000, 7: 2002, 8: 2003, 9: 2004}
CHANEL_8 = {10: 2005, 11: 2006, 12: 2008, 13: 2009, 14: 2010, 15: 2011, 16: 2012, 17: 2012, 18: 2013, 19: 2014,
            20: 2014, 21: 2015, 22: 2016, 23: 2017, 24: 2017, 25: 2018, 26: 2018, 27: 2019, 28: 2019, 29: 2019,
            30: 2020, 31: 2021}
CURRENT_YEAR = 2026

def production_year(brand, code):
    """Return (year, how) decoded from the brand's date code / serial, or (None, None)."""
    if not code:
        return None, None
    code = code.strip()
    if brand == "Louis Vuitton":
        if "NFC" in code.upper():
            return 2021, "LV microchip (2021 or later)"
        m = re.fullmatch(r"[A-Z]{2}\s?(\d)(\d)(\d)(\d)", code.upper())
        if m:
            d = m.groups()
            yy, period = int(d[1] + d[3]), int(d[0] + d[2])
            year = 1900 + yy if yy >= 90 else 2000 + yy
            if year <= 2021 and 1 <= period <= 53:
                return year, "LV date code"
    elif brand == "Chanel":
        if re.fullmatch(r"\d{8}", code):
            year = CHANEL_8.get(int(code[:2]))
            return (year, "Chanel serial") if year else (None, None)
        if re.fullmatch(r"\d{7}", code):
            return CHANEL_7[int(code[0])], "Chanel serial"
        if re.fullmatch(r"[A-Z0-9]{8}", code.upper()) and re.search(r"[A-Z]", code.upper()):
            return 2021, "Chanel microchip (2021 or later)"
    elif brand == "Hermes":
        m = re.search(r"\((\d{4})\)", code)
        if m:
            return int(m.group(1)), "Hermes blind stamp"
    return None, None

CONDITIONS = {"Pristine", "Excellent", "Great", "Very Good", "Good", "Fair"}

def money(s):
    m = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)", s or "")
    return float(m.group(1).replace(",", "")) if m else None

def tag(tags, prefix):
    hits = [t[len(prefix):] for t in tags if t.startswith(prefix)]
    return hits[0].replace("-", " ").title() if hits else None

rows = []
for path in sorted(glob.glob(os.path.join(raw, "page_*.json"))):
    for p in json.load(open(path, encoding="utf-8"))["products"]:
        b = body_fields(p.get("body_html") or "")
        v = p["variants"][0]
        price, retail = float(v["price"]), money(b.get("Estimated Retail Price"))
        cond = (b.get("Condition") or "").split(".")[0].strip()
        if not cond:
            cond = next((t[10:] for t in p["tags"] if t[10:] in CONDITIONS and t.startswith("bc-filter-")), "")
        cond = cond.title() or None
        brand = p["vendor"]
        year, how = production_year(brand, b.get("Brand Code"))
        rows.append({
            "listing_id": p["id"],
            "brand": brand,
            "southpark_brand": brand in SOUTHPARK,
            "model": b.get("Model") or p["title"],
            "item_type": tag(p["tags"], "item-type-"),
            "product_type": p["product_type"],
            "condition": cond,
            "asking_price": price,
            "estimated_retail_price": retail,
            "asking_price_ratio": round(100 * price / retail, 1) if retail else None,
            "production_year": year,
            "item_age_years": CURRENT_YEAR - year if year else None,
            "year_source": how,
            "brand_code": b.get("Brand Code"),
            "exterior_material": b.get("Exterior Material"),
            "exterior_color": b.get("Exterior Color"),
            "hardware_color": b.get("Hardware Color"),
            "accessories": b.get("Accessories"),
            "size": tag(p["tags"], "meta-size-"),
            "available": v["available"],
            "listed_at": p["created_at"][:10],
            "dropship_partner": "reflaunt_dropship" in p["tags"],
            "url": "https://shop.rebag.com/products/" + p["handle"],
        })

df = pd.DataFrame(rows).drop_duplicates("listing_id")
os.makedirs(out, exist_ok=True)
df.to_csv(os.path.join(out, "rebag_all_listings.csv"), index=False, encoding="utf-8-sig")
sp = df[df.southpark_brand]
sp.to_csv(os.path.join(out, "rebag_southpark_brands.csv"), index=False, encoding="utf-8-sig")

print(f"all listings: {len(df)} | SouthPark brands: {len(sp)} | with retail price: {sp.estimated_retail_price.notna().sum()}")
print(sp.groupby("brand").agg(listings=("listing_id", "size"), with_retail=("estimated_retail_price", "count"),
      median_asking=("asking_price", "median"), median_ratio=("asking_price_ratio", "median"))
      .sort_values("listings", ascending=False).to_string())
print("with production year:", sp.production_year.notna().sum(), "| year AND retail:", (sp.production_year.notna() & sp.estimated_retail_price.notna()).sum())
print(sp.groupby("brand").production_year.agg(["count", "min", "median", "max"]).query("count > 0").to_string())
print(sp.item_type.value_counts(dropna=False).to_string())
print(sp.condition.value_counts(dropna=False).head(10).to_string())

# Readable Excel copy: SouthPark brands, plus a per-brand summary
summary = (sp.groupby(["brand", "item_type"], dropna=False)
             .agg(listings=("listing_id", "size"), with_retail_price=("estimated_retail_price", "count"),
                  median_asking_price=("asking_price", "median"), median_retail_price=("estimated_retail_price", "median"),
                  median_asking_price_ratio=("asking_price_ratio", "median"))
             .reset_index().sort_values(["brand", "listings"], ascending=[True, False]))
aged = sp[sp.production_year.notna()].copy()
aged["age_band"] = pd.cut(aged.item_age_years, [-1, 5, 10, 20, 100], labels=["5 or less", "6-10", "11-20", "Over 20"])
over_time = (aged.groupby(["brand", "item_type", "age_band"], observed=True)
                 .agg(listings=("listing_id", "size"), median_asking_price=("asking_price", "median"),
                      listings_with_retail=("asking_price_ratio", "count"),
                      median_asking_price_ratio=("asking_price_ratio", "median"))
                 .reset_index())
with pd.ExcelWriter(os.path.join(out, "rebag_southpark_brands.xlsx"), engine="xlsxwriter") as w:
    for name, d in (("Summary", summary), ("Value Over Time", over_time), ("Listings", sp.drop(columns=["southpark_brand"]))):
        d = d.copy()
        d.columns = [c.replace("_", " ").title().replace("Pct", "%").replace("Id", "ID").replace("Url", "URL") for c in d.columns]
        d.to_excel(w, sheet_name=name, index=False, startrow=1, header=False)
        ws = w.sheets[name]
        ws.add_table(0, 0, len(d), len(d.columns) - 1,
                     {"columns": [{"header": c} for c in d.columns], "style": "Table Style Medium 2"})
        for i, c in enumerate(d.columns):
            width = max(len(c), d[c].astype(str).str.len().quantile(0.9)) + 2
            ws.set_column(i, i, min(width, 45))
        ws.freeze_panes(1, 1)
