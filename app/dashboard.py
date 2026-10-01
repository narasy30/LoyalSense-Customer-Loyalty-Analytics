"""
dashboard.py  (OPTIONAL BONUS MODULE)
-------------------------------------
A live loyalty control room built on the same src/ modules. Run it only after
`python src/main.py` has produced outputs/segment_model.joblib.

    pip install streamlit
    streamlit run app/dashboard.py

No analytics logic is duplicated here - every number comes from the same RFM
engine and the same clustering model used by the console pipeline.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import config                     # noqa: E402
import data_loader                # noqa: E402
import model as model_module      # noqa: E402
import rfm_engine                 # noqa: E402
from stream_simulator import (CustomerState, RollingKPI, evaluate,   # noqa: E402
                              migration)

st.set_page_config(page_title="LoyalSense Control Room", layout="wide")
st.title("LoyalSense - Customer Loyalty Analytics")

bundle = model_module.load()
pipe, names = bundle["pipeline"], bundle["cluster_names"]
scored, bins = rfm_engine.score_customers(data_loader.get_customers())

tab_live, tab_base, tab_cust = st.tabs(
    ["Live event stream", "Customer base", "Single customer"])

# ------------------------------------------------------------- live stream --
with tab_live:
    delay = st.slider("Seconds between events", 0.0, 3.0, 1.0, 0.5)
    if st.button("Start stream", type="primary"):
        states = {r["customer_id"]: CustomerState(r) for _, r in scored.iterrows()}
        events = data_loader.get_events()
        kpi = RollingKPI()
        cols = st.columns(5)
        tiles = [c.empty() for c in cols]
        move_slot = st.container()
        table_slot = st.empty()
        rows = []

        for _, ev in events.iterrows():
            time.sleep(delay)
            state = states.get(ev["customer_id"])
            if state is None:
                continue
            before = evaluate(state.as_dict(), bins, pipe, names)
            state.apply(ev)
            after = evaluate(state.as_dict(), bins, pipe, names)
            direction, moves = migration(before, after)
            points = (state.award_points(float(ev["amount"]),
                                         after["point_multiplier"])
                      if ev["event_type"] == "PURCHASE" else 0)
            kpi.update(ev, direction, after["escalated"], points,
                       after["clv"] - before["clv"])
            snap = kpi.snapshot()

            tiles[0].metric("Events", snap["events"])
            tiles[1].metric("Net revenue", f"Rs.{snap['net_revenue']:,.0f}")
            tiles[2].metric("Upgrades", snap["upgrades"])
            tiles[3].metric("Downgrades", snap["downgrades"])
            tiles[4].metric("CLV movement", f"{snap['clv_delta']:+,.0f}")

            if after["escalated"]:
                move_slot.error(f"{state.name}: {after['action']}")
            elif moves:
                fn = move_slot.success if direction == "upgrade" else move_slot.warning
                fn(f"{state.name}: " + " | ".join(moves))

            rows.append({
                "Event": ev["event_id"], "Customer": state.name,
                "Type": ev["event_type"], "Amount": float(ev["amount"]),
                "Segment": after["segment"], "Churn": after["churn_score"],
                "Band": after["churn_band"], "CLV": after["clv"],
                "Points": points, "Action": after["action"],
            })
            table_slot.dataframe(pd.DataFrame(rows), width="stretch")

# ------------------------------------------------------------ customer base --
with tab_base:
    a, b, c, d = st.columns(4)
    a.metric("Customers", len(scored))
    b.metric("Revenue", f"Rs.{scored['monetary'].sum():,.0f}")
    c.metric("Projected CLV", f"Rs.{scored['clv'].sum():,.0f}")
    d.metric("High churn risk", int((scored["churn_band"] == "HIGH").sum()))

    st.subheader("Revenue by segment")
    st.bar_chart(scored.groupby("segment")["monetary"].sum())
    st.subheader("Customers by churn band")
    st.bar_chart(scored["churn_band"].value_counts())
    st.dataframe(
        scored[["customer_id", "name", "city", "recency_days", "frequency",
                "monetary", "rfm_code", "segment", "churn_score", "churn_band",
                "clv", "action"]],
        width="stretch")

# --------------------------------------------------------- single customer --
with tab_cust:
    who = st.selectbox("Choose a customer",
                       scored["customer_id"] + " - " + scored["name"])
    cid = who.split(" - ")[0]
    row = scored[scored["customer_id"] == cid].iloc[0]

    x, y, z, w = st.columns(4)
    x.metric("Segment", row["segment"])
    y.metric("RFM code", row["rfm_code"])
    z.metric("Churn risk", f"{row['churn_score']}/100", row["churn_band"])
    w.metric("Projected CLV", f"Rs.{row['clv']:,.0f}")

    st.write(f"**Recommended action:** {row['action']}")
    st.write(f"**Offer:** {row['offer']}")
    st.caption(f"Risk factors: {row['churn_reasons']}")

    st.subheader("Simulate a purchase")
    amount = st.number_input("Purchase amount (Rs.)", 500, 100000, 8000, 500)
    if st.button("Apply event"):
        state = CustomerState(row)
        before = evaluate(state.as_dict(), bins, pipe, names)
        state.apply(pd.Series({"event_type": "PURCHASE", "amount": amount,
                               "rating": None}))
        after = evaluate(state.as_dict(), bins, pipe, names)
        direction, moves = migration(before, after)
        p, q = st.columns(2)
        p.metric("Segment after", after["segment"])
        q.metric("Churn after", f"{after['churn_score']}/100", after["churn_band"])
        if moves:
            st.success(" | ".join(moves)) if direction == "upgrade" \
                else st.warning(" | ".join(moves))
        else:
            st.info("No segment or risk-band movement from this purchase")
        st.caption(f"CLV {before['clv']:,.0f} -> {after['clv']:,.0f}")
