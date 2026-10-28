"""D2 EDA dashboard: Rebag asking-price ratios for SouthPark Mall brands.

Reads from Supabase (tables listings_train and southpark_inventory).
Env vars: SUPABASE_URL, SUPABASE_ANON_KEY  (read-only key; the service key is never used here).
Local preview without Supabase: set DASHBOARD_LOCAL_CSV=1 to read data/clean/*.csv instead.

Run locally:  uv run streamlit run dashboard/app.py

Colors follow the project's validated categorical/sequential palette (see the
dataviz design notes in the repo) — fixed hue-per-entity, never repainted by
filters, with a 6-8 ΔE CVD floor and legends/tooltips carrying identity.
"""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Luxury Resale EDA", page_icon="💎", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens — validated categorical palette (8 slots, fixed order) +
# sequential blue ramp for ordinal features. See dataviz skill / palette.md.
# ---------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BORDER = "#e1e0d9", "#c3c2b7", "rgba(11,11,11,0.10)"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# Fixed entity -> color. Never re-derived from the current (filtered) data,
# so dropping a brand never repaints the ones that remain.
BRAND_ORDER = ["Louis Vuitton", "Gucci", "Chanel", "Saint Laurent", "Balenciaga",
               "Tiffany & Co.", "Burberry", "Other SouthPark"]
BRAND_COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, MUTED]

ITEM_ORDER = ["Handbag", "Accessory", "Jewelry & Watch", "Shoes"]
ITEM_COLORS = [BLUE, ORANGE, AQUA, YELLOW]

COND = ["Fair", "Good", "Very Good", "Great", "Excellent", "Pristine"]
COND_COLORS = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]  # ordinal ramp, step>=250

AGE_ORDER = ["0-5", "6-10", "11-20", "21+", "Not available"]
AGE_COLORS = ["#86b6ef", "#5598e7", "#256abf", "#104281", MUTED]

ORD_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]


def ordinal_colors(n):
    """n evenly spaced steps from the sequential ramp, light -> dark."""
    if n <= 1:
        return [ORD_RAMP[len(ORD_RAMP) // 2]]
    step = (len(ORD_RAMP) - 1) / (n - 1)
    return [ORD_RAMP[round(i * step)] for i in range(n)]


def fixed_scale(order, colors):
    return alt.Scale(domain=order, range=colors)


# ---------------------------------------------------------------------------
# Chrome: a light "card" visual language on top of Streamlit's own theme.
# ---------------------------------------------------------------------------
_CSS = """
<style>
  [data-testid="stMetric"] {
      background: var(--background-color, #fff);
      border: 1px solid @BORDER@;
      border-radius: 12px;
      padding: 14px 16px 10px;
      box-shadow: 0 1px 2px rgba(11,11,11,0.04);
      transition: box-shadow .15s ease, transform .15s ease;
  }
  [data-testid="stMetric"]:hover {
      box-shadow: 0 4px 14px rgba(11,11,11,0.08);
      transform: translateY(-1px);
  }
  [data-testid="stMetricLabel"] { color: @INK2@; }
  .hero {
      background: linear-gradient(135deg, #1c5cab 0%, #2a78d6 55%, #1baf7a 130%);
      border-radius: 16px;
      padding: 28px 32px;
      color: #ffffff;
      margin-bottom: 18px;
      box-shadow: 0 6px 24px rgba(28,92,171,0.25);
  }
  .hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; font-weight: 700; }
  .hero p { margin: 0; opacity: 0.92; font-size: 0.95rem; max-width: 70ch; }
  .result-pill {
      display: inline-block;
      background: @BLUE@1a;
      color: @BLUE@;
      border-radius: 999px;
      padding: 3px 12px;
      font-weight: 600;
      font-size: 0.85rem;
  }
  div[data-baseweb="tab-list"] { gap: 4px; }
</style>
""".replace("@BORDER@", BORDER).replace("@INK2@", INK2).replace("@BLUE@", BLUE)
st.markdown(_CSS, unsafe_allow_html=True)


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

st.markdown(f"""
<div class="hero">
  <h1>💎 Asking prices vs. retail for SouthPark Mall brands on Rebag</h1>
  <p>Target: <strong>asking-price ratio</strong> = Rebag's advertised asking price ÷ Rebag's estimated
  retail price × 100. These are listing prices, not completed sales. Snapshot collected Oct 20, 2026.
  Training/validation rows only — the reserved test set is never loaded. Data source: {source}.</p>
</div>
""", unsafe_allow_html=True)

# ---- filters (one row, above every chart below) ----
c1, c2, c3, c4 = st.columns(4)
brands = c1.multiselect("Brand", [b for b in BRAND_ORDER if b in tv.brand_group.unique()])
items = c2.multiselect("Item group", [i for i in ITEM_ORDER if i in tv.item_group.unique()])
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

# ---------------------------------------------------------------------------
# Interactive estimator — a live lookup (not a model): pick a few traits and
# see the typical asking-price ratio for similar SouthPark listings, with an
# optional retail price to translate that into a ballpark asking price.
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("### 🔎 Estimate a listing's asking-price ratio")
    st.caption("Pick what you know. If there aren't enough similar listings, we widen the group automatically "
               "and tell you what changed.")
    e1, e2, e3, e4, e5 = st.columns([1.3, 1.1, 1.3, 1.1, 1])
    e_brand = e1.selectbox("Brand", ["Any"] + [b for b in BRAND_ORDER if b in tv.brand_group.unique()])
    e_item = e2.selectbox("Item group", ["Any"] + [i for i in ITEM_ORDER if i in tv.item_group.unique()])
    e_cond = e3.selectbox("Condition", ["Any"] + COND, index=0)
    e_age = e4.selectbox("Age", ["Any"] + [a for a in AGE_ORDER if a != "Not available"])
    e_retail = e5.number_input("Retail price ($, optional)", min_value=0, value=0, step=100)

    MIN_N = 15
    dims = [("brand_group", e_brand), ("item_group", e_item), ("condition", e_cond), ("age_band", e_age)]
    active = [(c, v) for c, v in dims if v != "Any"]
    relaxed = []
    d = tv
    # Try the full match first, then relax the least-recently-added dimension until n is large enough.
    for i in range(len(active), -1, -1):
        trial = active[:i]
        d = tv
        for col, val in trial:
            d = d[d[col] == val]
        if len(d) >= MIN_N or i == 0:
            relaxed = [c for c, _ in active[i:]]
            break

    if d.empty:
        st.warning("No listings match that combination at all — try a different brand or item group.")
    else:
        med, q1, q3, n = d.asking_price_ratio.median(), d.asking_price_ratio.quantile(.25), \
            d.asking_price_ratio.quantile(.75), len(d)
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"<span class='result-pill'>n = {n:,} similar listings</span>", unsafe_allow_html=True)
            st.markdown(f"## {med:.0f}%&nbsp;&nbsp;<span style='font-size:1rem;color:{INK2};font-weight:400'>"
                        f"typical ratio</span>", unsafe_allow_html=True)
            st.caption(f"Middle 50% of comparable listings: {q1:.0f}–{q3:.0f}%")
            if e_retail > 0:
                st.markdown(f"**Ballpark asking price:** ${e_retail * med / 100:,.0f} "
                            f"(range ${e_retail * q1 / 100:,.0f}–${e_retail * q3 / 100:,.0f})")
            if relaxed:
                st.caption(f"⚠️ Too few exact matches, so this ignores: {', '.join(relaxed)}.")
        with r2:
            chart = alt.Chart(d).mark_bar(color=BLUE, opacity=0.85, cornerRadiusTopLeft=4,
                                           cornerRadiusTopRight=4).encode(
                alt.X("asking_price_ratio:Q", bin=alt.Bin(step=3), title="Asking-price ratio (%)"),
                alt.Y("count():Q", title="Listings"),
                tooltip=[alt.Tooltip("count():Q", title="Listings")])
            rule = alt.Chart(pd.DataFrame({"m": [med]})).mark_rule(
                strokeDash=[4, 3], color=INK, size=1.5).encode(x="m:Q")
            st.altair_chart((chart + rule).properties(height=180), use_container_width=True)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Target", "By feature", "Explore", "Coverage & bias", "Data table"])

with tab1:
    hist = alt.Chart(f).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        alt.X("asking_price_ratio:Q", bin=alt.Bin(step=2), title="Asking-price ratio (%)"),
        alt.Y("count():Q", title="Listings"),
        tooltip=[alt.Tooltip("count():Q", title="Listings")])
    med = alt.Chart(pd.DataFrame({"m": [f.asking_price_ratio.median()]})).mark_rule(
        strokeDash=[4, 3], color=INK, size=1.5).encode(x="m:Q")
    st.altair_chart((hist + med).properties(height=340), use_container_width=True)
    st.caption("No listing is priced above **83.3%** of estimated retail, and the distribution is skewed "
               "toward lower ratios. Dashed line = median. Hover a bar for its exact count.")

with tab2:
    feature = st.selectbox("Group by", ["condition", "brand_group", "item_group", "age_band",
                                         "material_group", "n_extras"])
    g = (f.groupby(feature).asking_price_ratio
         .agg(median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75), n="count")
         .reset_index())

    if feature == "condition":
        order, colors = COND, COND_COLORS
    elif feature == "age_band":
        order, colors = [a for a in AGE_ORDER if a in g[feature].values], \
            AGE_COLORS[:len(g)] if len(g) <= len(AGE_COLORS) else ordinal_colors(len(g))
    elif feature == "brand_group":
        order, colors = [b for b in BRAND_ORDER if b in g[feature].values], None
    elif feature == "item_group":
        order, colors = [i for i in ITEM_ORDER if i in g[feature].values], None
    else:
        order = sorted(g[feature].unique(), key=lambda x: (isinstance(x, str), x))
        colors = ordinal_colors(len(order))

    if colors is None:
        colors = [dict(zip(BRAND_ORDER, BRAND_COLORS)).get(b, MUTED) for b in order] if feature == "brand_group" \
            else [dict(zip(ITEM_ORDER, ITEM_COLORS)).get(i, MUTED) for i in order]

    scale = fixed_scale(order, colors)
    base = alt.Chart(g).encode(alt.X(f"{feature}:N", sort=order, title=None))
    bars = base.mark_rule(strokeWidth=4).encode(
        y=alt.Y("q1:Q", title="Asking-price ratio (%)", scale=alt.Scale(zero=False)), y2="q3:Q",
        color=alt.Color(f"{feature}:N", scale=scale, legend=None))
    dots = base.mark_point(filled=True, size=110).encode(
        y="median:Q", color=alt.Color(f"{feature}:N", scale=scale, legend=None),
        tooltip=[alt.Tooltip(f"{feature}:N", title=feature.replace('_', ' ')),
                 alt.Tooltip("median:Q", format=".1f", title="median"),
                 alt.Tooltip("q1:Q", format=".0f", title="p25"), alt.Tooltip("q3:Q", format=".0f", title="p75"),
                 alt.Tooltip("n:Q", title="listings")])
    labels = base.mark_text(dy=-14, color=INK, fontSize=11, fontWeight=600).encode(
        y="median:Q", text=alt.Text("median:Q", format=".0f"))
    st.altair_chart((bars + dots + labels).properties(height=360), use_container_width=True)
    st.dataframe(g.round(1), hide_index=True, use_container_width=True)
    st.caption("Dot = median (labeled), bar = middle 50%. Groups under ~30 listings are noisy.")

with tab3:
    st.caption("Retail price vs. asking-price ratio. Color = condition (darker = better). "
               "Scroll to zoom, drag to pan, hover a point for details.")
    cscale = fixed_scale(COND, COND_COLORS)
    scatter = alt.Chart(f).mark_circle(opacity=0.65, stroke="white", strokeWidth=0.6).encode(
        x=alt.X("estimated_retail_price:Q", title="Estimated retail price ($)", scale=alt.Scale(type="log")),
        y=alt.Y("asking_price_ratio:Q", title="Asking-price ratio (%)", scale=alt.Scale(zero=False)),
        size=alt.Size("n_extras:Q", title="Extras included", scale=alt.Scale(range=[20, 220])),
        color=alt.Color("condition:N", scale=cscale, sort=COND, title="Condition"),
        tooltip=[alt.Tooltip("brand:N", title="Brand"), alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("item_group:N", title="Item"), alt.Tooltip("condition:N", title="Condition"),
                 alt.Tooltip("estimated_retail_price:Q", format="$,.0f", title="Retail"),
                 alt.Tooltip("asking_price:Q", format="$,.0f", title="Asking"),
                 alt.Tooltip("asking_price_ratio:Q", format=".1f", title="Ratio %")]
    ).properties(height=440).interactive()
    st.altair_chart(scatter, use_container_width=True)
    st.caption("Point size = number of extras included (dust bag, strap, card, box). "
               "Bigger, darker points toward the top are the best-preserved deals relative to retail.")

with tab4:
    st.subheader("Which listings have a retail price (and so a target)?")
    cov = (inv.groupby("item_type").has_target.agg(total="count", labeled="sum").reset_index())
    cov["share_labeled"] = cov.labeled / cov.total
    cov_chart = alt.Chart(cov).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("item_type:N", sort="-y", title=None),
        y=alt.Y("share_labeled:Q", title="Share with a retail price", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("item_type:N", title="Item type"), alt.Tooltip("total:Q", title="Total listings"),
                 alt.Tooltip("labeled:Q", title="With retail price"),
                 alt.Tooltip("share_labeled:Q", format=".0%", title="Share")])
    st.altair_chart(cov_chart.properties(height=260), use_container_width=True)
    st.dataframe(cov.style.format({"share_labeled": "{:.0%}"}), hide_index=True, use_container_width=True)
    st.markdown(
        f"- Only **{inv.has_target.mean():.0%}** of {len(inv):,} SouthPark listings show a retail price.\n"
        f"- Every labeled listing is a **dropship-partner** listing "
        f"({int((inv.has_target & ~inv.dropship_partner).sum())} non-partner listings have one).\n"
        "- Jewelry is 18% of inventory but 2% of the labeled sample, so results mostly describe handbags.")
    rec = (tv.groupby(["brand_group", "is_recent"]).asking_price_ratio.median().reset_index())
    rec["posted"] = rec.is_recent.map({True: "Sep–Oct 2026", False: "Before Sep 2026"})
    st.subheader("Older listings are priced lower, not higher")
    brand_order = [b for b in BRAND_ORDER if b in rec.brand_group.values]
    st.altair_chart(alt.Chart(rec).mark_point(filled=True, size=110, stroke="white", strokeWidth=0.8).encode(
        x=alt.X("asking_price_ratio:Q", title="Median asking-price ratio (%)", scale=alt.Scale(zero=False)),
        y=alt.Y("brand_group:N", title=None, sort=brand_order),
        color=alt.Color("posted:N", scale=alt.Scale(domain=["Sep–Oct 2026", "Before Sep 2026"],
                                                      range=[BLUE, ORANGE]), title="Posted"),
        tooltip=["brand_group", "posted", alt.Tooltip("asking_price_ratio:Q", format=".1f")]
    ).properties(height=320), use_container_width=True)

with tab5:
    st.dataframe(f.drop(columns=["cv_fold"], errors="ignore"), hide_index=True, use_container_width=True)
