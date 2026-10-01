"""
main.py
-------
Entry point. Runs the five stages in order:

    ingest -> RFM scoring -> EDA -> clustering -> real-time event stream

Usage:
    python main.py                # full run, 1 second between events
    python main.py --fast         # no delay
    python main.py --no-charts    # console only, skip matplotlib
"""

import argparse
from datetime import datetime

import config
import data_loader
import eda
import model as model_module
import rfm_engine
import stream_simulator


def banner(step, title):
    print("\n" + "#" * 72)
    print(f"# STEP {step}: {title}")
    print("#" * 72)


def write_summary(scored, metrics, k_table, profile, kpi):
    lines = [
        "LOYALSENSE - CUSTOMER LOYALTY ANALYTICS - RUN SUMMARY",
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"As-of date: {config.AS_OF_DATE}",
        "",
        "1. CUSTOMER BASE",
        f"   Customers          : {len(scored)}",
        f"   Historic revenue   : Rs.{scored['monetary'].sum():,}",
        f"   Projected CLV      : Rs.{scored['clv'].sum():,.0f} of gross margin",
        f"   Segment mix        : {scored['segment'].value_counts().to_dict()}",
        f"   Churn bands        : {scored['churn_band'].value_counts().to_dict()}",
        "",
        "2. SEGMENTATION MODEL (unsupervised)",
        f"   Chosen k           : {metrics['k']}",
        f"   Silhouette         : {metrics['silhouette']}",
        f"   Inertia            : {metrics['inertia']}",
        f"   Cluster sizes      : {metrics['sizes']}",
        "",
        "   k selection table:",
    ]
    for _, r in k_table.iterrows():
        lines.append(f"     k={int(r['k'])}  inertia={r['inertia']:>8}  "
                     f"silhouette={r['silhouette']}")

    lines += ["", "   Cluster profiles:"]
    for cluster, r in profile.iterrows():
        lines.append(f"     {cluster} {r['cluster_name']:<20} "
                     f"n={int(r['customers'])}  recency={r['recency']:.0f}d  "
                     f"freq={r['frequency']:.1f}  spend=Rs.{r['monetary']:,.0f}")

    lines += [
        "",
        "3. REAL-TIME STREAM",
        f"   Events processed   : {kpi['events']}",
        f"   Customers touched  : {kpi['customers']}",
        f"   Revenue / refunds  : Rs.{kpi['revenue']:,.0f} / Rs.{kpi['refunds']:,.0f}",
        f"   Net revenue        : Rs.{kpi['net_revenue']:,.0f}",
        f"   Points issued      : {kpi['points']:,}",
        f"   Upgrades           : {kpi['upgrades']}",
        f"   Downgrades         : {kpi['downgrades']}",
        f"   Escalations        : {kpi['escalations']}",
        f"   Net CLV movement   : Rs.{kpi['clv_delta']:+,.0f}",
        "",
        "Artefacts in outputs/: charts (*.png), segment_model.joblib,",
        "scored_customers.csv, realtime_actions.csv, summary_report.txt",
    ]
    config.SUMMARY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[main] summary written to {config.SUMMARY_REPORT.name}")


def main():
    parser = argparse.ArgumentParser(description="LoyalSense customer loyalty analytics")
    parser.add_argument("--fast", action="store_true", help="no delay in the stream")
    parser.add_argument("--no-charts", action="store_true", help="skip PNG generation")
    args = parser.parse_args()

    print("\nLOYALSENSE - CUSTOMER LOYALTY ANALYTICS")
    print(f"Started at {datetime.now():%Y-%m-%d %H:%M:%S}"
          f"   (analysis as-of {config.AS_OF_DATE})")

    banner(1, "DATA INGESTION & RFM DERIVATION")
    customers = data_loader.get_customers()

    banner(2, "RFM SCORING, SEGMENTS, CHURN RISK & CLV")
    scored, bins = rfm_engine.score_customers(customers)
    print(scored[["customer_id", "name", "recency_days", "frequency", "monetary",
                  "rfm_code", "segment", "churn_score", "churn_band",
                  "clv"]].to_string(index=False))

    banner(3, "EXPLORATORY DATA ANALYSIS")
    if args.no_charts:
        eda.print_summary(scored)
    else:
        eda.run(scored)

    banner(4, "UNSUPERVISED SEGMENTATION (K-MEANS)")
    pipe, labels, profile, metrics, k_table = model_module.train(scored)
    scored["cluster"] = labels
    scored["cluster_name"] = [profile.loc[c, "cluster_name"] for c in labels]
    scored.to_csv(config.SCORED_CUSTOMERS, index=False)
    print(f"[main] scored customers written to {config.SCORED_CUSTOMERS.name}")

    banner(5, "REAL-TIME EVENT STREAM")
    bundle = {"pipeline": pipe,
              "cluster_names": profile["cluster_name"].to_dict(),
              "metrics": metrics}
    _, kpi, _ = stream_simulator.run(bundle, bins, scored,
                                     delay=0 if args.fast else None)

    write_summary(scored, metrics, k_table, profile, kpi)
    print("\nPipeline finished successfully.\n")


if __name__ == "__main__":
    main()
