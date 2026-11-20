"""Clustering algorithms, parameter experiments and cluster profiling.

Both DBSCAN and K-Means are wrapped here so that notebooks and the comparison
step use identical evaluation code (same metrics, same sampling).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------
def label_summary(labels: np.ndarray) -> dict:
    """Count clusters and noise points for a label vector (-1 = noise)."""
    labels = np.asarray(labels)
    unique = set(labels.tolist())
    n_noise = int((labels == -1).sum())
    n_clusters = len(unique - {-1})

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(100.0 * n_noise / len(labels), 2) if len(labels) else 0.0,
        "n_clustered_points": int(len(labels) - n_noise),
    }


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10_000,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Compute internal validity metrics on a clustering assignment.

    Noise points (label -1) are excluded from the metric computation, because
    silhouette/Davies-Bouldin/Calinski-Harabasz are undefined for noise. This
    also means DBSCAN's scores describe *only the clustered points* and are not
    directly comparable to a partition that covers every row - the notebooks
    report coverage (noise %) next to the scores for this reason.

    Silhouette is O(n^2); it is computed on a reproducible random sample to keep
    runtime reasonable. Davies-Bouldin and Calinski-Harabasz use all clustered
    points.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    summary = label_summary(labels)
    mask = labels != -1
    X_clustered = X[mask]
    y_clustered = labels[mask]

    result = dict(summary)
    result["silhouette"] = np.nan
    result["davies_bouldin"] = np.nan
    result["calinski_harabasz"] = np.nan
    result["silhouette_sample_size"] = 0

    # Metrics need at least two clusters and two samples.
    if summary["n_clusters"] < 2 or len(X_clustered) < 2:
        return result

    n = len(X_clustered)
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        # Guard against a sample that accidentally contains a single cluster.
        if len(np.unique(y_clustered[idx])) > 1:
            result["silhouette"] = float(
                silhouette_score(X_clustered[idx], y_clustered[idx])
            )
            result["silhouette_sample_size"] = sample_size
        else:
            result["silhouette"] = float(silhouette_score(X_clustered, y_clustered))
            result["silhouette_sample_size"] = n
    else:
        result["silhouette"] = float(silhouette_score(X_clustered, y_clustered))
        result["silhouette_sample_size"] = n

    result["davies_bouldin"] = float(davies_bouldin_score(X_clustered, y_clustered))
    result["calinski_harabasz"] = float(calinski_harabasz_score(X_clustered, y_clustered))

    return result


def cluster_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    include_noise: bool = True,
    value_col: str = "median_house_value",
) -> pd.DataFrame:
    """Describe every cluster with counts and mean housing characteristics.

    The cluster label is added to a copy of the frame, so the input is untouched.
    Noise (label -1) is kept by default and listed last.
    """
    data = df.copy()
    data["cluster"] = np.asarray(labels)

    if not include_noise:
        data = data[data["cluster"] != -1]

    grouped = data.groupby("cluster")

    profile = pd.DataFrame({
        "n_points": grouped.size(),
        "share_pct": grouped.size() / len(data) * 100.0,
        "mean_house_value": grouped[value_col].mean(),
        "median_house_value": grouped[value_col].median(),
        "mean_income": grouped["median_income"].mean(),
        "mean_age": grouped["housing_median_age"].mean(),
        "mean_rooms_per_household": grouped["rooms_per_household"].mean(),
        "mean_bedrooms_per_room": grouped["bedrooms_per_room"].mean(),
        "mean_population_per_household": grouped["population_per_household"].mean(),
        "mean_latitude": grouped["latitude"].mean(),
        "mean_longitude": grouped["longitude"].mean(),
    })

    if "ocean_proximity" in data.columns:
        dominant = (
            data.groupby("cluster")["ocean_proximity"]
            .agg(lambda s: s.value_counts().idxmax())
        )
        profile["dominant_ocean_proximity"] = dominant

    profile["share_pct"] = profile["share_pct"].round(2)
    numeric = profile.select_dtypes(include=[np.number]).columns
    profile[numeric] = profile[numeric].round(3)

    # Order clusters numerically, noise (-1) pushed to the end.
    order = sorted(profile.index, key=lambda c: (c == -1, c))
    return profile.loc[order]


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------
def k_distances(X: np.ndarray, k: int = 5) -> np.ndarray:
    """Sorted distance to each point's k-th nearest neighbour.

    The k-distance graph plots this curve; its "knee" is a data-driven choice
    for ``eps``: beyond the knee, points stop being close to their neighbours.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    n_neighbors = min(k + 1, len(X))  # +1 because the point itself is included
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)

    return np.sort(distances[:, -1])


def knee_index(y: np.ndarray) -> int:
    """Index of the knee of a monotonic curve, via distance to its chord.

    The curve is normalised, then each point's perpendicular distance to the
    straight line joining the first and last points is measured. The farthest
    point is the elbow/knee. This is the same geometric idea used by the
    ``kneed`` package, without the extra dependency.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return max(0, len(y) - 1)

    x = np.linspace(0.0, 1.0, len(y))
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Chord from the first to the last normalised point.
    x1, y1, x2, y2 = x[0], y_norm[0], x[-1], y_norm[-1]
    numerator = np.abs((y2 - y1) * x - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    denominator = np.hypot(y2 - y1, x2 - x1) + 1e-12

    return int(np.argmax(numerator / denominator))


def knee_point(y: np.ndarray) -> float:
    """Value of the curve at its knee (used for the k-distance ``eps`` hint)."""
    y = np.asarray(y, dtype=float)
    return float(y[knee_index(y)])


def run_dbscan(X: np.ndarray, eps: float, min_samples: int = 5) -> np.ndarray:
    """Fit DBSCAN and return labels (-1 marks noise)."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm="auto")
    return model.fit_predict(X)


def dbscan_grid(
    X: np.ndarray,
    eps_values: list,
    min_samples_values: list,
    sample_size: int = 10_000,
) -> pd.DataFrame:
    """Fit DBSCAN for every (eps, min_samples) pair and collect metrics."""
    rows = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            start = time.perf_counter()
            labels = run_dbscan(X, eps=eps, min_samples=min_samples)
            runtime = time.perf_counter() - start

            metrics = evaluate_clustering(X, labels, sample_size=sample_size)
            rows.append({
                "eps": eps,
                "min_samples": min_samples,
                **metrics,
                "runtime_s": round(runtime, 3),
            })

    return pd.DataFrame(rows)


def select_dbscan_params(results: pd.DataFrame, min_clusters: int = 2) -> pd.Series:
    """Pick the most informative parameter set from a grid result.

    Preference order:
    1. at least ``min_clusters`` clusters,
    2. noise between 1% and 30% (enough clusters to be useful, not so much that
       a third of the state is discarded),
    3. highest silhouette,
    4. lowest noise share.
    """
    usable = results[results["n_clusters"] >= min_clusters].copy()
    if usable.empty:
        raise ValueError("No parameter combination produced 2+ clusters.")

    reasonable = usable[
        (usable["noise_pct"] >= 1.0) & (usable["noise_pct"] <= 30.0)
    ]
    candidates = reasonable if not reasonable.empty else usable

    candidates = candidates.sort_values(
        ["silhouette", "noise_pct"], ascending=[False, True]
    )
    return candidates.iloc[0]


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------
def run_kmeans(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    random_state: int = RANDOM_STATE,
) -> KMeans:
    """Fit K-Means with a fixed seed so results are reproducible."""
    if k < 2:
        raise ValueError("k must be >= 2 for a meaningful clustering")

    model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    model.fit(X)
    return model


def kmeans_experiments(
    X: np.ndarray,
    k_values: range | list,
    sample_size: int = 10_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Fit K-Means for several K and record inertia and internal metrics."""
    rows = []

    for k in k_values:
        start = time.perf_counter()
        model = run_kmeans(X, k=k, random_state=random_state)
        runtime = time.perf_counter() - start

        labels = model.labels_
        metrics = evaluate_clustering(
            X, labels, sample_size=sample_size, random_state=random_state
        )

        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "runtime_s": round(runtime, 3),
        })

    return pd.DataFrame(rows)


def elbow_k(results: pd.DataFrame) -> int:
    """K at the knee of the inertia curve (Elbow Method)."""
    inertia = results["inertia"].to_numpy(dtype=float)
    return int(results["k"].to_numpy()[knee_index(inertia)])


def best_k_by_silhouette(results: pd.DataFrame) -> int:
    """K with the highest silhouette score in the experiment table."""
    valid = results.dropna(subset=["silhouette"])
    if valid.empty:
        raise ValueError("No silhouette scores available.")
    return int(valid.loc[valid["silhouette"].idxmax(), "k"])


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
def comparison_table(results: dict) -> pd.DataFrame:
    """Turn ``{algorithm: metrics_dict}`` into a tidy comparison frame."""
    rows = []
    for name, metrics in results.items():
        rows.append({
            "algorithm": name,
            "n_clusters": metrics.get("n_clusters"),
            "n_noise": metrics.get("n_noise"),
            "noise_pct": metrics.get("noise_pct"),
            "silhouette": metrics.get("silhouette"),
            "davies_bouldin": metrics.get("davies_bouldin"),
            "calinski_harabasz": metrics.get("calinski_harabasz"),
            "runtime_s": metrics.get("runtime_s"),
        })
    return pd.DataFrame(rows).set_index("algorithm")
