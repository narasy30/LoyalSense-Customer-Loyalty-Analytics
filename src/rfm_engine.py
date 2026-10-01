"""
rfm_engine.py
-------------
Stage 2: the explainable half of the project.

Turns raw RFM numbers into scores, scores into a named segment, behaviour into
a churn risk, and a segment into a concrete loyalty action. No machine learning
anywhere in this file - a marketing manager could reproduce every number with a
spreadsheet, which is exactly why it sits alongside the clustering model rather
than inside it.
"""

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------- RFM scoring
def build_score_bins(df):
    """
    Learn the quintile boundaries once, from the baseline population.

    This is the single most important idea in the module. If the boundaries
    were recomputed for every incoming event, a customer's score could change
    without the customer doing anything, purely because their peers moved.
    Freezing the cut-points makes scores comparable over time.
    """
    bins = {}
    for col in ["recency_days", "frequency", "monetary"]:
        # 5 quintiles -> 6 edges. Interior edges only; the ends are open.
        edges = df[col].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        bins[col] = [round(float(e), 2) for e in edges]
    print(f"[rfm_engine] Score boundaries learned from {len(df)} customers")
    for col, edges in bins.items():
        print(f"    {col:<14} quintile cut-points: {edges}")
    return bins


def _score(value, edges, reverse=False):
    """
    Map a value to 1-5 using pre-learned edges.

    reverse=True is used for recency, where a LOW number is GOOD, so the
    scale is flipped: 3 days since purchase should score 5, not 1.
    """
    rank = int(np.searchsorted(edges, value, side="right")) + 1   # 1..5
    return 6 - rank if reverse else rank


def score_rfm(row, bins):
    """Return the three 1-5 scores plus the concatenated RFM code."""
    r = _score(row["recency_days"], bins["recency_days"], reverse=True)
    f = _score(row["frequency"], bins["frequency"])
    m = _score(row["monetary"], bins["monetary"])
    return {"r_score": r, "f_score": f, "m_score": m, "rfm_code": f"{r}{f}{m}"}


# ------------------------------------------------------------ segment mapping
def segment_for(r, f, m):
    """Walk the ordered rule grid and return the first segment that matches."""
    for name, r_lo, r_hi, f_lo, f_hi, m_lo, m_hi in config.SEGMENT_RULES:
        if r_lo <= r <= r_hi and f_lo <= f <= f_hi and m_lo <= m <= m_hi:
            return name
    return "Others"


# ---------------------------------------------------------------- churn risk
def _band_penalty(value, bands, higher_is_better=False):
    """Walk a (threshold, penalty) list and return the first match."""
    for threshold, penalty in bands:
        if higher_is_better and value >= threshold:
            return penalty
        if not higher_is_better and value <= threshold:
            return penalty
    return bands[-1][1]


def churn_risk(row):
    """
    Additive, fully explainable churn score from 0 (safe) to 100 (leaving).

    Returns the score, its band, and a written reason for every penalty so the
    number can always be defended to the customer or to a manager.
    """
    reasons, score = [], 0

    p = _band_penalty(row["recency_days"], config.RECENCY_BANDS)
    if p:
        reasons.append(f"{int(row['recency_days'])} days since last purchase (+{p})")
    score += p

    p = _band_penalty(row["return_rate"], config.RETURN_RATE_BANDS)
    if p:
        reasons.append(f"return rate {row['return_rate']:.0%} (+{p})")
    score += p

    p = _band_penalty(row["avg_rating"], config.RATING_BANDS, higher_is_better=True)
    if p:
        reasons.append(f"average rating {row['avg_rating']:.1f} (+{p})")
    score += p

    tickets = int(row["support_tickets"])
    if tickets:
        p = min(tickets * config.TICKET_PENALTY, config.TICKET_PENALTY_CAP)
        reasons.append(f"{tickets} support ticket(s) (+{p})")
        score += p

    score = min(score, 100)
    band = next(label for limit, label in config.CHURN_BANDS if score <= limit)

    return {
        "churn_score": score,
        "churn_band": band,
        "churn_reasons": "; ".join(reasons) if reasons else "no risk flags",
    }


# ------------------------------------------------------------------------ CLV
def clv(row):
    """
    Projected gross margin over the remaining relationship.

        CLV = average order value
              x orders per year
              x expected remaining years
              x gross margin

    Expected remaining years shrinks as churn risk rises, which is what links
    the risk score to a rupee figure the business can act on.
    """
    remaining_years = config.MAX_HORIZON_YEARS * (1 - row["churn_score"] / 100)
    value = (row["avg_order_value"] * row["orders_per_year"]
             * remaining_years * config.GROSS_MARGIN)
    return {
        "expected_years": round(remaining_years, 2),
        "clv": round(float(value), 0),
    }


# -------------------------------------------------------------- loyalty action
def decide_action(segment, churn_band, clv_value):
    """
    Pick the next best action.

    The segment chooses the play; churn risk and value decide whether it is
    escalated to a human. A retention desk that receives every alert acts on
    none of them, so escalation is deliberately narrow.
    """
    action, offer, multiplier = config.ACTION_MATRIX.get(
        segment, config.ACTION_MATRIX["Others"])

    escalate = churn_band == "HIGH" and clv_value >= config.HIGH_VALUE_CLV
    base_action = action
    if escalate:
        action = f"{config.ESCALATE_ACTION}; then {base_action.lower()}"

    return {
        "action": action,
        "base_action": base_action,
        "offer": offer,
        "point_multiplier": multiplier,
        "escalated": escalate,
    }


# ------------------------------------------------------------ frame-level API
def score_customers(df, bins=None):
    """Apply every rule above to a whole DataFrame and return the scored copy."""
    bins = bins or build_score_bins(df)

    rfm = df.apply(lambda r: score_rfm(r, bins), axis=1, result_type="expand")
    out = df.join(rfm)
    out["segment"] = out.apply(
        lambda r: segment_for(r["r_score"], r["f_score"], r["m_score"]), axis=1)

    risk = out.apply(churn_risk, axis=1, result_type="expand")
    out = out.join(risk)
    out = out.join(out.apply(clv, axis=1, result_type="expand"))

    plays = out.apply(
        lambda r: decide_action(r["segment"], r["churn_band"], r["clv"]),
        axis=1, result_type="expand")
    out = out.join(plays)

    counts = out["segment"].value_counts().to_dict()
    print(f"[rfm_engine] Scored {len(out)} customers -> {counts}")
    return out, bins


if __name__ == "__main__":
    import data_loader

    scored, _ = score_customers(data_loader.get_customers())
    print(scored[["customer_id", "rfm_code", "segment", "churn_score",
                  "churn_band", "clv"]].to_string(index=False))
