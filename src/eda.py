"""
eda.py
------
Stage 3: describe the customer base in numbers, then in four pictures.

matplotlib only, with the Agg backend so it runs on a server, in Docker and in
Colab without trying to open a window.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

SEGMENT_COLOUR = {
    "Champions": "#1b6b3a",
    "Loyal Customers": "#3f9d4f",
    "Potential Loyalist": "#7bbf6a",
    "Promising": "#a8d08d",
    "New Customers": "#4a90c2",
    "Needs Attention": "#e0a53a",
    "At Risk": "#e07b39",
    "Cannot Lose Them": "#c0392b",
    "Hibernating": "#9b7b8c",
    "Lost": "#7a7a7a",
    "Others": "#bdbdbd",
}
BAND_COLOUR = {"LOW": "#3f9d4f", "MEDIUM": "#e0a53a", "HIGH": "#c0392b"}


def print_summary(df):
    print("\n" + "=" * 72)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 72)

    print(f"Customers            : {len(df)}")
    print(f"Total revenue        : Rs.{df['monetary'].sum():,}")
    print(f"Average order value  : Rs.{df['avg_order_value'].mean():,.0f}")
    print(f"Median recency       : {df['recency_days'].median():.0f} days")
    print(f"Total projected CLV  : Rs.{df['clv'].sum():,.0f} of gross margin")

    print("\n-- Segment breakdown --")
    grp = (df.groupby("segment")
             .agg(customers=("customer_id", "size"),
                  revenue=("monetary", "sum"),
                  avg_recency=("recency_days", "mean"),
                  avg_churn=("churn_score", "mean"),
                  clv=("clv", "sum"))
             .sort_values("revenue", ascending=False))
    total_rev = df["monetary"].sum()
    for seg, r in grp.iterrows():
        print(f"  {seg:<19} {int(r['customers']):2d} cust  "
              f"Rs.{r['revenue']:>9,.0f} ({r['revenue'] / total_rev:5.1%})  "
              f"recency {r['avg_recency']:5.0f}d  churn {r['avg_churn']:4.0f}  "
              f"CLV Rs.{r['clv']:>9,.0f}")

    print("\n-- Churn risk bands --")
    for band in ["LOW", "MEDIUM", "HIGH"]:
        sub = df[df["churn_band"] == band]
        if len(sub):
            print(f"  {band:<7} {len(sub):2d} customers   "
                  f"Rs.{sub['monetary'].sum():>9,.0f} of historic revenue at stake")

    print("\n-- The Pareto check --")
    ranked = df.sort_values("monetary", ascending=False)
    top20 = max(1, int(len(df) * 0.2))
    share = ranked.head(top20)["monetary"].sum() / total_rev
    print(f"  Top {top20} customers ({top20 / len(df):.0%}) generate "
          f"{share:.1%} of all revenue")

    print("\n-- Correlation with total spend --")
    corr = df[["recency_days", "frequency", "avg_order_value", "tenure_days",
               "return_rate", "monetary"]].corr(numeric_only=True)["monetary"]
    for name, v in corr.drop("monetary").sort_values(ascending=False).items():
        print(f"  {name:<18} {v:+.3f}")


def _save(fig, name):
    path = config.OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[eda] saved {path.name}")


def build_charts(df):
    # 1. Customers and revenue per segment ----------------------------------
    grp = (df.groupby("segment")
             .agg(customers=("customer_id", "size"), revenue=("monetary", "sum"))
             .sort_values("revenue"))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(grp.index, grp["revenue"] / 1000,
                   color=[SEGMENT_COLOUR.get(s, "#999") for s in grp.index])
    for bar, n in zip(bars, grp["customers"]):
        ax.text(bar.get_width() + 12, bar.get_y() + bar.get_height() / 2,
                f"{n} cust", va="center", fontsize=8)
    ax.set_xlabel("Historic revenue (thousand rupees)")
    ax.set_title("Revenue and customer count by RFM segment")
    ax.set_xlim(0, grp["revenue"].max() / 1000 * 1.22)
    _save(fig, "01_segment_revenue.png")

    # 2. The RFM map ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for seg in df["segment"].unique():
        sub = df[df["segment"] == seg]
        ax.scatter(sub["recency_days"], sub["monetary"] / 1000,
                   s=sub["frequency"] * 6 + 30,
                   color=SEGMENT_COLOUR.get(seg, "#999"),
                   alpha=0.85, edgecolors="white", label=seg)
    for _, r in df.iterrows():
        ax.annotate(r["customer_id"][-3:], (r["recency_days"], r["monetary"] / 1000),
                    fontsize=6, ha="center", va="center", color="white")
    ax.set_xlabel("Recency (days since last purchase) - lower is better")
    ax.set_ylabel("Monetary value (thousand rupees)")
    ax.set_title("The RFM map (bubble size = purchase frequency)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    _save(fig, "02_rfm_map.png")

    # 3. Churn risk vs value -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for band in ["LOW", "MEDIUM", "HIGH"]:
        sub = df[df["churn_band"] == band]
        if not len(sub):
            continue
        ax.scatter(sub["churn_score"], sub["clv"] / 1000, s=90,
                   color=BAND_COLOUR[band], alpha=0.85, edgecolors="white",
                   label=f"{band} risk")
    ax.axvline(50, color="#666", ls="--", lw=1)
    ax.axhline(config.HIGH_VALUE_CLV / 1000, color="#666", ls="--", lw=1)
    ax.text(52, config.HIGH_VALUE_CLV / 1000 * 1.05, "escalation zone",
            fontsize=8, color="#c0392b")
    ax.set_xlabel("Churn risk score (0-100)")
    ax.set_ylabel("Projected CLV (thousand rupees of margin)")
    ax.set_title("Who is worth saving: risk against value")
    ax.legend(fontsize=8)
    _save(fig, "03_churn_vs_value.png")

    # 4. Average RFM scores per segment --------------------------------------
    grp = (df.groupby("segment")[["r_score", "f_score", "m_score"]]
             .mean().sort_values("m_score"))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = range(len(grp))
    h = 0.26
    ax.barh([i + h for i in y], grp["r_score"], height=h, label="R (recency)",
            color="#4a90c2")
    ax.barh(list(y), grp["f_score"], height=h, label="F (frequency)",
            color="#3f9d4f")
    ax.barh([i - h for i in y], grp["m_score"], height=h, label="M (monetary)",
            color="#e0a53a")
    ax.set_yticks(list(y))
    ax.set_yticklabels(grp.index, fontsize=8)
    ax.set_xlabel("Average score (1 = worst, 5 = best)")
    ax.set_xlim(0, 5.6)
    ax.set_title("What each segment actually looks like")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "04_segment_profiles.png")


def run(df):
    print_summary(df)
    build_charts(df)


if __name__ == "__main__":
    import data_loader
    import rfm_engine

    scored, _ = rfm_engine.score_customers(data_loader.get_customers())
    run(scored)
