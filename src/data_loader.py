"""
data_loader.py
--------------
Stage 1: read the customer master and the event stream, prove they are usable,
and derive the raw RFM measurements that everything downstream depends on.

Note what is NOT here: there is no target column, no "churned = yes/no" label.
This project is unsupervised, so the data preparation has to be right - there
is no ground truth later to rescue a mistake.
"""

import pandas as pd

import config


def load_csv(path, **kwargs):
    """Read a CSV and fail loudly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path, **kwargs)
    print(f"[data_loader] Loaded {len(df)} rows x {len(df.columns)} columns "
          f"from {path.name}")
    return df


def validate(df):
    """Data-quality gate for the customer master."""
    report = {
        "customers": len(df),
        "duplicate_ids": int(df["customer_id"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "zero_order_rows": int((df["total_orders"] <= 0).sum()),
        "returns_exceed_orders": int((df["returns"] > df["total_orders"]).sum()),
    }

    if report["duplicate_ids"]:
        df = df.drop_duplicates(subset="customer_id", keep="first")

    # A customer with no orders has no RFM profile at all, so they are excluded
    # rather than imputed. Inventing a purchase history would be worse.
    if report["zero_order_rows"]:
        df = df[df["total_orders"] > 0]

    # Returns can never exceed orders; clip rather than discard the customer.
    df["returns"] = df[["returns", "total_orders"]].min(axis=1)

    for col in ["avg_rating"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"[data_loader] Validation -> {report}")
    return df, report


def derive_rfm(df):
    """
    Turn raw history into the three numbers every loyalty programme runs on.

        Recency   - how many days since the customer last bought
        Frequency - how many orders they have placed in total
        Monetary  - how much they have spent in total

    Plus the supporting ratios that make the clusters interpretable.
    """
    df = df.copy()
    as_of = pd.Timestamp(config.AS_OF_DATE)

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

    # The three pillars
    df["recency_days"] = (as_of - df["last_purchase_date"]).dt.days
    df["frequency"] = df["total_orders"]
    df["monetary"] = df["total_spend"]

    # Supporting measures
    df["tenure_days"] = (as_of - df["signup_date"]).dt.days
    df["avg_order_value"] = (df["monetary"] / df["frequency"]).round(0)
    df["return_rate"] = (df["returns"] / df["frequency"]).round(3)
    # Annualise over at least one full year. Dividing a 3-month-old customer's
    # 6 orders by 0.29 years would claim 21 orders a year and inflate their
    # lifetime value above a genuine Champion's - the classic CLV bug.
    years = (df["tenure_days"] / 365.25).clip(lower=1.0)
    df["orders_per_year"] = (df["frequency"] / years).round(2)

    print(f"[data_loader] Derived RFM for {len(df)} customers "
          f"(recency {df['recency_days'].min()}-{df['recency_days'].max()} days, "
          f"spend Rs.{df['monetary'].min():,} - Rs.{df['monetary'].max():,})")
    return df


def get_customers():
    """Full pipeline for the customer master."""
    df, _ = validate(load_csv(config.CUSTOMERS_CSV))
    return derive_rfm(df)


def get_events():
    """
    Load the live event stream, sorted chronologically.

    `rating` is deliberately sparse - only REVIEW events carry one - so it is
    read as a nullable float rather than being force-filled with a fake value.
    """
    df = load_csv(config.EVENTS_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["amount"] = df["amount"].fillna(0).astype(float)
    counts = df["event_type"].value_counts().to_dict()
    print(f"[data_loader] Event mix -> {counts}")
    return df


if __name__ == "__main__":
    customers = get_customers()
    print(customers[["customer_id", "recency_days", "frequency", "monetary",
                     "avg_order_value", "return_rate"]].head(8).to_string(index=False))
    get_events()
