"""
stream_simulator.py
-------------------
Stage 5: the real-time part - and the part that is genuinely different from an
ordinary scoring pipeline.

In most streaming projects each record is scored on its own and then forgotten.
Here every event belongs to a *customer*, and it changes that customer's
running profile. The system therefore has to hold **keyed state**: one live
record per customer, updated in place as events arrive.

That is exactly how Kafka Streams, Flink and Spark Structured Streaming model
this problem, and it brings one capability a stateless pipeline can never have:
detecting the moment a customer *moves between segments*.

For every arriving event we:
    1. look up that customer's current state
    2. apply the event to it (purchase, return, ticket or review)
    3. re-score RFM against the FROZEN quintile boundaries
    4. re-derive segment, churn risk and CLV
    5. re-assign the K-Means cluster
    6. compare before and after -> migration detected?
    7. choose the loyalty action and update rolling KPIs
"""

import time
from datetime import datetime

import pandas as pd

import config
import data_loader
import model as model_module
import rfm_engine


class CustomerState:
    """
    One live, mutable profile per customer.

    Everything the scoring rules need is held here as plain attributes, so an
    update is a handful of arithmetic operations rather than a re-read of the
    whole history. That is what makes per-event processing O(1).
    """

    def __init__(self, row):
        self.customer_id = row["customer_id"]
        self.name = row["name"]
        self.recency_days = float(row["recency_days"])
        self.frequency = float(row["frequency"])
        self.monetary = float(row["monetary"])
        self.returns = float(row["returns"])
        self.support_tickets = float(row["support_tickets"])
        self.avg_rating = float(row["avg_rating"])
        self.tenure_days = float(row["tenure_days"])
        self.loyalty_points = float(row["loyalty_points"])
        self.events_seen = 0

    # -- derived values, always recomputed so they can never go stale --------
    def as_dict(self):
        aov = self.monetary / self.frequency if self.frequency else 0.0
        # Same minimum one-year annualisation window as data_loader, so a
        # streamed profile is scored on exactly the same basis as a batch one.
        years = max(self.tenure_days / 365.25, 1.0)
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "recency_days": self.recency_days,
            "frequency": self.frequency,
            "monetary": self.monetary,
            "returns": self.returns,
            "support_tickets": self.support_tickets,
            "avg_rating": self.avg_rating,
            "tenure_days": self.tenure_days,
            "avg_order_value": round(aov, 0),
            "return_rate": round(self.returns / self.frequency, 3) if self.frequency else 0.0,
            "orders_per_year": round(self.frequency / years, 2),
            "loyalty_points": self.loyalty_points,
        }

    # -- the four event handlers ---------------------------------------------
    def apply(self, event):
        """Mutate the state according to the event type. Returns a note."""
        self.events_seen += 1
        kind = event["event_type"]
        amount = float(event["amount"])

        if kind == "PURCHASE":
            self.frequency += 1
            self.monetary += amount
            self.recency_days = 0            # they just bought: recency resets
            return f"purchase of Rs.{amount:,.0f}"

        if kind == "RETURN":
            self.returns += 1
            self.monetary = max(0.0, self.monetary - amount)
            return f"return of Rs.{amount:,.0f}"

        if kind == "SUPPORT_TICKET":
            self.support_tickets += 1
            return "support ticket raised"

        if kind == "REVIEW":
            rating = float(event["rating"])
            # Incremental mean over a bounded memory. Weighting the new review
            # against every past order would bury it; config.RATING_MEMORY keeps
            # recent feedback influential.
            n = min(max(self.frequency, 1), config.RATING_MEMORY)
            self.avg_rating = round((self.avg_rating * n + rating) / (n + 1), 2)
            return f"review of {rating:.0f}/5 (average now {self.avg_rating})"

        return "unrecognised event, state unchanged"

    def award_points(self, amount, multiplier):
        points = round(amount * config.BASE_POINTS_PER_RUPEE * multiplier)
        self.loyalty_points += points
        return points


def evaluate(state_dict, bins, pipe, names):
    """Run the full scoring stack over one customer state snapshot."""
    row = pd.Series(state_dict)
    scores = rfm_engine.score_rfm(row, bins)
    segment = rfm_engine.segment_for(scores["r_score"], scores["f_score"],
                                     scores["m_score"])
    risk = rfm_engine.churn_risk(row)
    row_with_risk = pd.Series({**state_dict, **risk})
    value = rfm_engine.clv(row_with_risk)
    cluster, cluster_name = model_module.predict_cluster(pipe, names, state_dict)
    play = rfm_engine.decide_action(segment, risk["churn_band"], value["clv"])
    return {**scores, "segment": segment, **risk, **value,
            "cluster": cluster, "cluster_name": cluster_name, **play}


def migration(before, after):
    """Compare two evaluations and describe what moved."""
    moves = []
    direction = "none"

    if before["segment"] != after["segment"]:
        rank_before = config.SEGMENT_RANK.get(before["segment"], 0)
        rank_after = config.SEGMENT_RANK.get(after["segment"], 0)
        direction = "upgrade" if rank_after > rank_before else "downgrade"
        arrow = "->" if direction == "upgrade" else "->"
        moves.append(f"segment {before['segment']} {arrow} {after['segment']}")

    if before["churn_band"] != after["churn_band"]:
        moves.append(f"churn {before['churn_band']} -> {after['churn_band']}")
        # A customer whose risk band worsens has been downgraded in business
        # terms even if their RFM segment happens not to have moved.
        if direction == "none":
            order = ["LOW", "MEDIUM", "HIGH"]
            worse = order.index(after["churn_band"]) > order.index(before["churn_band"])
            direction = "downgrade" if worse else "upgrade"

    if before["cluster_name"] != after["cluster_name"]:
        moves.append(f"cluster {before['cluster_name']} -> {after['cluster_name']}")

    return direction, moves


class RollingKPI:
    """Running totals updated in O(1) per event, exactly like a stream job."""

    def __init__(self):
        self.events = 0
        self.revenue = 0.0
        self.refunds = 0.0
        self.points = 0.0
        self.upgrades = 0
        self.downgrades = 0
        self.escalations = 0
        self.clv_delta = 0.0
        self.customers_touched = set()

    def update(self, event, direction, escalated, points, clv_change):
        self.events += 1
        self.customers_touched.add(event["customer_id"])
        if event["event_type"] == "PURCHASE":
            self.revenue += float(event["amount"])
        if event["event_type"] == "RETURN":
            self.refunds += float(event["amount"])
        self.points += points
        if direction == "upgrade":
            self.upgrades += 1
        elif direction == "downgrade":
            self.downgrades += 1
        if escalated:
            self.escalations += 1
        self.clv_delta += clv_change

    def snapshot(self):
        return {
            "events": self.events,
            "customers": len(self.customers_touched),
            "revenue": round(self.revenue, 2),
            "refunds": round(self.refunds, 2),
            "net_revenue": round(self.revenue - self.refunds, 2),
            "points": int(self.points),
            "upgrades": self.upgrades,
            "downgrades": self.downgrades,
            "escalations": self.escalations,
            "clv_delta": round(self.clv_delta, 0),
        }


def run(bundle=None, bins=None, scored=None, delay=None):
    """Consume the whole event stream. Returns (action log, final KPIs, states)."""
    bundle = bundle or model_module.load()
    pipe, names = bundle["pipeline"], bundle["cluster_names"]

    if scored is None or bins is None:
        scored, bins = rfm_engine.score_customers(data_loader.get_customers())

    delay = config.STREAM_DELAY_SECONDS if delay is None else delay
    events = data_loader.get_events()

    # ---- build the keyed state store --------------------------------------
    states = {r["customer_id"]: CustomerState(r) for _, r in scored.iterrows()}
    kpi = RollingKPI()
    log = []

    print("\n" + "=" * 72)
    print("REAL-TIME EVENT STREAM")
    print("=" * 72)
    print(f"State store initialised with {len(states)} customer profiles")

    for _, event in events.iterrows():
        time.sleep(delay)
        received = datetime.now().strftime("%H:%M:%S")
        cid = event["customer_id"]

        if cid not in states:
            print(f"\n[{received}] {event['event_id']}: unknown customer {cid}, skipped")
            continue

        state = states[cid]

        # 1. snapshot BEFORE
        before = evaluate(state.as_dict(), bins, pipe, names)

        # 2. apply the event to the keyed state
        note = state.apply(event)

        # 3. snapshot AFTER
        after_state = state.as_dict()
        after = evaluate(after_state, bins, pipe, names)

        # 4. what moved?
        direction, moves = migration(before, after)
        clv_change = after["clv"] - before["clv"]

        # 5. reward the purchase at the (new) segment's multiplier
        points = 0
        if event["event_type"] == "PURCHASE":
            points = state.award_points(float(event["amount"]),
                                        after["point_multiplier"])

        kpi.update(event, direction, after["escalated"], points, clv_change)
        snap = kpi.snapshot()

        flag = {"upgrade": "  UP  ", "downgrade": " DOWN ", "none": "  --  "}[direction]
        print(f"\n[{received}] {event['event_id']} {event['timestamp']:%H:%M} "
              f"{cid} {state.name}  [{flag}]")
        print(f"    event   : {event['event_type']} via {event['channel']} "
              f"- {event['detail']}")
        print(f"    applied : {note}")
        print(f"    profile : R{after['r_score']}F{after['f_score']}M{after['m_score']}"
              f" | recency {after_state['recency_days']:.0f}d"
              f" | orders {after_state['frequency']:.0f}"
              f" | spend Rs.{after_state['monetary']:,.0f}"
              f" | AOV Rs.{after_state['avg_order_value']:,.0f}")
        print(f"    segment : {after['segment']} ({after['cluster_name']})"
              f" | churn {after['churn_score']}/100 {after['churn_band']}"
              f" | CLV Rs.{after['clv']:,.0f} ({clv_change:+,.0f})")
        if moves:
            for mv in moves:
                print(f"    MOVED   : {mv}")
        if points:
            # Read the balance off the live state, not off the snapshot taken
            # before the points were awarded.
            print(f"    points  : +{points} at {after['point_multiplier']}x "
                  f"(balance {state.loyalty_points:,.0f})")
        print(f"    action  : {after['action']}")
        print(f"    offer   : {after['offer']}")
        if after["escalated"]:
            print(f"    *** ESCALATED: {after['churn_band']} churn risk on a "
                  f"customer still worth Rs.{after['clv']:,.0f} of margin ***")
        print(f"    LIVE KPI: {snap['events']} events | {snap['customers']} customers"
              f" | net Rs.{snap['net_revenue']:,.0f}"
              f" | up {snap['upgrades']} / down {snap['downgrades']}"
              f" | CLV {snap['clv_delta']:+,.0f}")

        log.append({
            "received_at": received,
            "event_id": event["event_id"],
            "timestamp": event["timestamp"],
            "customer_id": cid,
            "name": state.name,
            "event_type": event["event_type"],
            "amount": float(event["amount"]),
            "segment_before": before["segment"],
            "segment_after": after["segment"],
            "direction": direction,
            "churn_before": before["churn_score"],
            "churn_after": after["churn_score"],
            "churn_band": after["churn_band"],
            "cluster_name": after["cluster_name"],
            "clv_before": before["clv"],
            "clv_after": after["clv"],
            "clv_change": clv_change,
            "points_awarded": points,
            "escalated": after["escalated"],
            "action": after["action"],
            "offer": after["offer"],
            "moves": " | ".join(moves),
        })

    log_df = pd.DataFrame(log)
    log_df.to_csv(config.EVENT_LOG, index=False)
    print(f"\n[stream] action log written to {config.EVENT_LOG.name}")

    final = kpi.snapshot()
    print("\n--- SESSION SUMMARY ---")
    print(f"Events processed      : {final['events']}")
    print(f"Customers touched     : {final['customers']} of {len(states)}")
    print(f"Revenue / refunds     : Rs.{final['revenue']:,.0f} / "
          f"Rs.{final['refunds']:,.0f}  (net Rs.{final['net_revenue']:,.0f})")
    print(f"Loyalty points issued : {final['points']:,}")
    print(f"Segment upgrades      : {final['upgrades']}")
    print(f"Segment downgrades    : {final['downgrades']}")
    print(f"Escalations raised    : {final['escalations']}")
    print(f"Net CLV movement      : Rs.{final['clv_delta']:+,.0f} of projected margin")

    return log_df, final, states


if __name__ == "__main__":
    run()
