"""D2 EDA plots. Target-based plots use train_val rows only; the test holdout is never read.

Usage:
    uv run python scripts/make_plots.py
Writes reports/figures/fig1..fig5 PNGs.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND = ["Fair", "Good", "Very Good", "Great", "Excellent", "Pristine"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def fig1_coverage(sp):
    """Which listings carry a retail price? Item-type mix, labeled vs all SouthPark."""
    order = ["Handbag", "Jewelry", "Accessory", "Apparel", "Shoes", "Watch"]
    allp = sp.item_type.value_counts(normalize=True).reindex(order).fillna(0) * 100
    lab = sp[sp.has_target].item_type.value_counts(normalize=True).reindex(order).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = range(len(order)); w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], allp, w, color=ORANGE, label=f"All SouthPark listings (n={len(sp):,})")
    ax.bar([i + w / 2 + 0.01 for i in x], lab, w, color=BLUE, label=f"Listings with a retail price (n={sp.has_target.sum():,})")
    for i, (a, b) in enumerate(zip(allp, lab)):
        ax.text(i - w / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8, color=INK2)
        ax.text(i + w / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(list(x), order); ax.set_ylabel("Share of listings (%)"); ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_title("Fig 1. The labeled sample is almost all handbags")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    save(fig, "fig1_coverage_item_type.png")


def fig2_target(tv):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(tv.asking_price_ratio, bins=range(10, 88, 2), color=BLUE, edgecolor="white", linewidth=1)
    med = tv.asking_price_ratio.median(); mx = tv.asking_price_ratio.max()
    ax.axvline(med, color=INK, lw=1.2, ls="--")
    ax.text(med - 1, ax.get_ylim()[1] * 0.92, f"median {med:.1f}%", ha="right", fontsize=8.5, color=INK)
    ax.axvline(mx, color=INK2, lw=1.2, ls=":")
    ax.annotate(f"max {mx:.1f}%", (mx, ax.get_ylim()[1] * 0.97), xytext=(mx + 1.5, ax.get_ylim()[1] * 0.97),
                fontsize=8.5, color=INK2, va="center")
    ax.set_xlim(8, 92)
    ax.set_xlabel("Asking price as % of Rebag's estimated retail price")
    ax.set_ylabel("Listings"); ax.grid(axis="x", visible=False)
    ax.set_title(f"Fig 2. Target is left-skewed and capped near 83% (n={len(tv):,})")
    save(fig, "fig2_target_distribution.png")


def iqr_dots(ax, groups, labels, color):
    for i, g in enumerate(groups):
        q1, q2, q3 = g.quantile([.25, .5, .75])
        ax.plot([i, i], [q1, q3], color=color, lw=2, solid_capstyle="round")
        ax.plot(i, q2, "o", ms=8, color=color, mec="white", mew=2)
        ax.text(i + 0.12, q2, f"{q2:.0f}%", va="center", fontsize=8.5, color=INK)
        ax.text(i, ax.get_ylim()[0] + 1, f"n={len(g)}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(labels)), labels); ax.grid(axis="x", visible=False)


def fig3_condition(tv):
    groups = [tv[tv.condition == c].asking_price_ratio for c in COND]
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax.set_ylim(25, 85)
    iqr_dots(ax, groups, COND, BLUE)
    ax.set_ylabel("Asking-price ratio (%)")
    ax.set_title("Fig 3. Ratio rises from Very Good up; Fair/Good overlap (median, middle 50%)")
    save(fig, "fig3_ratio_by_condition.png")


def fig4_age(tv):
    a = tv[tv.age_available]
    bands = ["0-5", "6-10", "11-20", "21+"]
    groups = [a[a.age_band == b].asking_price_ratio for b in bands]
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax.set_ylim(25, 85)
    iqr_dots(ax, groups, [f"{b} yrs" for b in bands], BLUE)
    ax.set_ylabel("Asking-price ratio (%)")
    ax.set_title(f"Fig 4. Ratio falls with age for Louis Vuitton and Chanel (n={len(a):,})")
    save(fig, "fig4_ratio_by_age.png")


def fig5_brand_recency(tv):
    brands = tv.brand_group.value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, (flag, color, lab) in enumerate([(True, BLUE, "Posted Sep–Oct 2026"),
                                            (False, ORANGE, "Posted before Sep 2026")]):
        sub = tv[tv.is_recent == flag].groupby("brand_group").asking_price_ratio.agg(["median", "count"])
        sub = sub.reindex(brands)
        ys = [i + (0.15 if j else -0.15) for i in range(len(brands))]
        ax.scatter(sub["median"], ys, s=60, color=color, edgecolor="white", linewidth=1.5, zorder=3,
                   label=lab)
    bm = tv.groupby("brand_group").asking_price_ratio.median().reindex(brands)
    ax.scatter(bm, range(len(brands)), marker="|", s=260, color=INK, linewidth=2, zorder=4,
               label="All listings (baseline)")
    nr = tv[tv.is_recent].brand_group.value_counts().reindex(brands).fillna(0).astype(int)
    no = tv[~tv.is_recent].brand_group.value_counts().reindex(brands).fillna(0).astype(int)
    ax.set_yticks(range(len(brands)), [f"{b} ({r} / {o})" for b, r, o in zip(brands, nr, no)])
    ax.set_xlim(38, 72)
    ax.invert_yaxis(); ax.set_xlabel("Median asking-price ratio (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Fig 5. Brand medians (the baseline) and recent vs older listings")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=3)
    ax.text(0, -0.36, "Labels show n recent / n older. Older medians for brands with < 15 older listings are unstable.",
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    save(fig, "fig5_brand_recency.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(ROOT / "data" / "clean" / "southpark_all.csv")
    tv = pd.read_csv(ROOT / "data" / "clean" / "train_val.csv")
    fig1_coverage(sp); fig2_target(tv); fig3_condition(tv); fig4_age(tv); fig5_brand_recency(tv)
