"""D2 EDA dashboard: Rebag asking-price ratios for SouthPark Mall brands.

Reads from Supabase (tables listings_train and southpark_inventory).
Env vars: SUPABASE_URL, SUPABASE_ANON_KEY  (read-only key; the service key is never used here).
Local preview without Supabase: set DASHBOARD_LOCAL_CSV=1 to read data/clean/*.csv instead.

Run locally:  uv run streamlit run dashboard/app.py
"""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Luxury Resale EDA", layout="wide")
COND = ["Fair", "Good", "Very Good", "Great", "Excellent", "Pristine"]
BLUE, ORANGE = "#2a78d6", "#eb6834"


def fetch_all(client, table):
    """Supabase returns at most 1,000 rows per request, so page through."""
    rows, start, step = [], 0, 1000
    while True:
        chunk = client.table(table).select("*").range(start, start + step - 1).execute().data
        rows += chunk
        if len(chunk) < step:
            return pd.DataFrame(rows)
        start += step


@st.cache_data(ttl=3600, show_spinner="Loading data from Supabase…")
def load():
    if os.environ.get("DASHBOARD_LOCAL_CSV"):
        root = Path(__file__).resolve().parents[1] / "data" / "clean"
        return pd.read_csv(root / "train_val.csv"), pd.read_csv(root / "southpark_all.csv"), "local CSV"
    from supabase import create_client
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    return fetch_all(client, "listings_train"), fetch_all(client, "southpark_inventory"), "Supabase"


try:
    tv, inv, source = load()
except Exception as e:  # show a readable error instead of a stack trace
    st.error(f"Could not load data from Supabase: {e}")
    st.stop()

st.title("Asking prices vs. retail for SouthPark Mall brands on Rebag")
st.caption(
    "Target: **asking-price ratio** = Rebag's advertised asking price ÷ Rebag's estimated retail price × 100. "
    "These are listing prices, not completed sales. Snapshot collected Oct 20, 2026. "
    f"Training/validation rows only — the reserved test set is not loaded. Data source: {source}."
)

# ---- filters (one row) ----
c1, c2, c3, c4 = st.columns(4)
brands = c1.multiselect("Brand", sorted(tv.brand_group.unique()))
items = c2.multiselect("Item group", sorted(tv.item_group.unique()))
conds = c3.multiselect("Condition", COND)
recency = c4.radio("Listing posted", ["All", "Sep–Oct 2026", "Before Sep 2026"], horizontal=True)

f = tv.copy()
if brands: f = f[f.brand_group.isin(brands)]
if items: f = f[f.item_group.isin(items)]
if conds: f = f[f.condition.isin(conds)]
if recency != "All": f = f[f.is_recent == (recency == "Sep–Oct 2026")]

if f.empty:
    st.warning("No listings match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Listings", f"{len(f):,}")
k2.metric("Median asking-price ratio", f"{f.asking_price_ratio.median():.1f}%")
k3.metric("Middle 50%", f"{f.asking_price_ratio.quantile(.25):.0f}–{f.asking_price_ratio.quantile(.75):.0f}%")
k4.metric("Age available", f"{f.age_available.mean():.0%}")

tab1, tab2, tab3, tab4 = st.tabs(["Target", "By feature", "Coverage & bias", "Data table"])

with tab1:
    hist = alt.Chart(f).mark_bar(color=BLUE).encode(
        alt.X("asking_price_ratio:Q", bin=alt.Bin(step=2), title="Asking-price ratio (%)"),
        alt.Y("count():Q", title="Listings"), tooltip=["count():Q"])
    med = alt.Chart(pd.DataFrame({"m": [f.asking_price_ratio.median()]})).mark_rule(
        strokeDash=[4, 3], color="black").encode(x="m:Q")
    st.altair_chart(hist + med, use_container_width=True)
    st.markdown("No listing is priced above **83.3%** of estimated retail, and the distribution is skewed "
                "toward lower ratios. Dashed line = median.")

with tab2:
    feature = st.selectbox("Group by", ["condition", "brand_group", "item_group", "age_band", "material_group",
                                        "n_extras"])
    g = (f.groupby(feature).asking_price_ratio
         .agg(median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), n="count")
         .reset_index())
    order = COND if feature == "condition" else (["0-5", "6-10", "11-20", "21+", "Not available"]
                                                 if feature == "age_band" else None)
    base = alt.Chart(g).encode(alt.X(f"{feature}:N", sort=order, title=None))
    chart = (base.mark_rule(color=BLUE, strokeWidth=3).encode(y=alt.Y("q1:Q", title="Asking-price ratio (%)",
                                                                      scale=alt.Scale(zero=False)), y2="q3:Q")
             + base.mark_point(filled=True, size=90, color=BLUE).encode(
                 y="median:Q", tooltip=[feature, alt.Tooltip("median:Q", format=".1f"), "n:Q"]))
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(g.round(1), hide_index=True, use_container_width=True)
    st.caption("Dot = median, bar = middle 50%. Groups under ~30 listings are noisy.")

with tab3:
    st.subheader("Which listings have a retail price (and so a target)?")
    cov = (inv.groupby("item_type").has_target.agg(total="count", labeled="sum").reset_index())
    cov["share_labeled"] = cov.labeled / cov.total
    st.dataframe(cov.style.format({"share_labeled": "{:.0%}"}), hide_index=True, use_container_width=True)
    st.markdown(
        f"- Only **{inv.has_target.mean():.0%}** of {len(inv):,} SouthPark listings show a retail price.\n"
        f"- Every labeled listing is a **dropship-partner** listing "
        f"({int((inv.has_target & ~inv.dropship_partner).sum())} non-partner listings have one).\n"
        "- Jewelry is 18% of inventory but 2% of the labeled sample, so results mostly describe handbags.")
    rec = (tv.groupby(["brand_group", "is_recent"]).asking_price_ratio.median().reset_index())
    rec["posted"] = rec.is_recent.map({True: "Sep–Oct 2026", False: "Before Sep 2026"})
    st.subheader("Older listings are priced lower, not higher")
    st.altair_chart(alt.Chart(rec).mark_point(filled=True, size=90).encode(
        x=alt.X("asking_price_ratio:Q", title="Median asking-price ratio (%)", scale=alt.Scale(zero=False)),
        y=alt.Y("brand_group:N", title=None),
        color=alt.Color("posted:N", scale=alt.Scale(range=[ORANGE, BLUE]), title="Posted"),
        tooltip=["brand_group", "posted", alt.Tooltip("asking_price_ratio:Q", format=".1f")]),
        use_container_width=True)

with tab4:
    st.dataframe(f.drop(columns=["cv_fold"], errors="ignore"), hide_index=True, use_container_width=True)
