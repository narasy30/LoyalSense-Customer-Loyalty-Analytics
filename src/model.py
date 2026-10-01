"""
model.py
--------
Stage 4: unsupervised segmentation with K-Means.

This is the part that differs most from a typical classification project.
There is no target column and no accuracy score, because nobody ever labelled
these customers. The model does not learn to predict an answer; it learns the
natural groupings that already exist in the data, and it is the analyst's job
to decide whether those groupings mean anything.

Two questions therefore have to be answered explicitly:
    1. How many clusters? -> the elbow curve and the silhouette score
    2. What are they? ----> profile each cluster and give it a business name
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def build_pipeline(k):
    """
    Scale, then cluster.

    Scaling is not optional here, it is the whole ballgame. K-Means groups
    points by straight-line distance, and monetary values run into the hundreds
    of thousands while return_rate sits between 0 and 1. Without scaling,
    'distance' would mean 'difference in rupees' and every other feature would
    be ignored completely.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10,
                          random_state=config.RANDOM_STATE)),
    ])


def choose_k(X):
    """
    Evaluate every candidate k and report both diagnostics.

    Inertia (the elbow) always falls as k rises, so it can never pick k on its
    own - you look for the bend. The silhouette score does have a maximum:
    +1 means tight, well-separated clusters, 0 means overlapping ones, and a
    negative value means points are closer to a neighbouring cluster than to
    their own.
    """
    rows = []
    for k in config.K_RANGE:
        pipe = build_pipeline(k)
        labels = pipe.fit_predict(X)
        scaled = pipe.named_steps["scale"].transform(X)
        rows.append({
            "k": k,
            "inertia": round(float(pipe.named_steps["kmeans"].inertia_), 2),
            "silhouette": round(float(silhouette_score(scaled, labels)), 3),
        })

    table = pd.DataFrame(rows)
    best = int(table.loc[table["silhouette"].idxmax(), "k"])

    print("\n" + "=" * 72)
    print("CHOOSING THE NUMBER OF CLUSTERS")
    print("=" * 72)
    print(table.to_string(index=False))
    print(f"\nBest silhouette at k = {best} "
          f"({table['silhouette'].max():.3f})")
    return best, table


def profile_clusters(df, labels):
    """
    Describe each cluster in business language.

    A cluster number is useless to a marketing team. This function computes the
    average behaviour of each group and derives a name from it, which is what
    turns a maths output into something that can go on a slide.
    """
    tmp = df.copy()
    tmp["cluster"] = labels

    prof = tmp.groupby("cluster").agg(
        customers=("customer_id", "size"),
        recency=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        aov=("avg_order_value", "mean"),
        churn=("churn_score", "mean"),
    ).round(1)

    # Name each cluster by comparing it with the OTHER clusters, not with the
    # population. Comparing a cluster mean against a customer-level median mixes
    # two different populations and can hand the same name to two clusters.
    cut_r = prof["recency"].median()
    cut_m = prof["monetary"].median()

    names, used = {}, set()
    for cluster, r in prof.iterrows():
        active = r["recency"] <= cut_r
        valuable = r["monetary"] >= cut_m
        label = (f"{'High' if valuable else 'Low'}-value "
                 f"{'active' if active else 'lapsing'}")
        # Guarantee uniqueness so segment-migration messages stay meaningful.
        if label in used:
            label = f"{label} ({int(r['customers'])} cust)"
        used.add(label)
        names[cluster] = label
    prof["cluster_name"] = pd.Series(names)

    print("\nCluster profiles:")
    print(prof.to_string())
    return prof, names


def train(df):
    """Pick k, fit, profile, persist. Returns (pipeline, labels, profile)."""
    X = df[config.CLUSTER_FEATURES]

    k, k_table = choose_k(X)

    pipe = build_pipeline(k)
    labels = pipe.fit_predict(X)

    scaled = pipe.named_steps["scale"].transform(X)
    metrics = {
        "k": k,
        "silhouette": round(float(silhouette_score(scaled, labels)), 3),
        "inertia": round(float(pipe.named_steps["kmeans"].inertia_), 2),
        "sizes": pd.Series(labels).value_counts().sort_index().tolist(),
    }
    print(f"\nFitted K-Means with k={k}: sizes {metrics['sizes']}, "
          f"silhouette {metrics['silhouette']}")

    profile, names = profile_clusters(df, labels)

    joblib.dump({"pipeline": pipe, "cluster_names": names, "metrics": metrics,
                 "k_table": k_table}, config.MODEL_FILE)
    print(f"[model] saved to {config.MODEL_FILE.name}")

    return pipe, labels, profile, metrics, k_table


def load():
    """Load the persisted bundle, or explain how to create it."""
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError("Model not trained yet. Run main.py first.")
    return joblib.load(config.MODEL_FILE)


def predict_cluster(pipe, names, state_row):
    """Assign a single (possibly updated) customer to a cluster."""
    frame = pd.DataFrame([{f: state_row[f] for f in config.CLUSTER_FEATURES}])
    cluster = int(pipe.predict(frame)[0])
    return cluster, names.get(cluster, f"Cluster {cluster}")


if __name__ == "__main__":
    import data_loader
    import rfm_engine

    scored, _ = rfm_engine.score_customers(data_loader.get_customers())
    train(scored)
