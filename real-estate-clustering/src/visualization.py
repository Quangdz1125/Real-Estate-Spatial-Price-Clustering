"""Static and interactive visualisations for the clustering project.

* matplotlib / seaborn -> PNG files in ``outputs/visualizations``
* folium             -> standalone HTML maps in ``outputs/maps``
* plotly             -> standalone interactive HTML charts in ``outputs/visualizations``

Every function returns the object it created and saves it with a stable file
name, so notebooks stay short and outputs are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

from preprocessing import FIGURE_DIR, MAP_DIR, ensure_output_dirs

RANDOM_STATE = 42
NOISE_COLOR = "#7f0000"

# A categorical palette that stays readable with 15+ clusters.
CLUSTER_PALETTE = (
    list(plt.get_cmap("tab20").colors)
    + list(plt.get_cmap("tab20b").colors)
    + list(plt.get_cmap("tab20c").colors)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_style() -> None:
    """Apply one consistent plotting style across all notebooks."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.autolayout": True,
    })


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a matplotlib figure to outputs/visualizations and return the path."""
    ensure_output_dirs()
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    return path


def _cluster_colors(labels: np.ndarray) -> dict:
    """Map each cluster label to a colour, with noise always dark red."""
    unique = sorted(set(np.asarray(labels).tolist()))
    colors = {}
    color_idx = 0
    for label in unique:
        if label == -1:
            colors[label] = NOISE_COLOR
        else:
            colors[label] = CLUSTER_PALETTE[color_idx % len(CLUSTER_PALETTE)]
            color_idx += 1
    return colors


def _stratified_sample(
    df: pd.DataFrame,
    labels: np.ndarray,
    sample_n: int | None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sample rows while keeping the cluster proportions intact.

    Drawing a plain random sample could miss small clusters entirely; sampling
    proportionally per cluster (at least one row each) keeps the map honest.
    """
    labels = np.asarray(labels)
    if sample_n is None or sample_n >= len(df):
        return df.reset_index(drop=True), labels

    data = df.reset_index(drop=True).copy()
    data["_label"] = labels

    frac = sample_n / len(data)
    parts = [
        group.sample(max(1, int(round(len(group) * frac))), random_state=random_state)
        for _, group in data.groupby("_label")
    ]
    sampled = pd.concat(parts)

    if len(sampled) > sample_n:
        sampled = sampled.sample(sample_n, random_state=random_state)

    return (
        sampled.drop(columns="_label").reset_index(drop=True),
        sampled["_label"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Exploratory figures
# ---------------------------------------------------------------------------
def plot_price_distribution(df: pd.DataFrame, name: str = "price_distribution"):
    """Histogram + box plot of median house value (capped values visible)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    sns.histplot(df["median_house_value"], bins=60, ax=axes[0], color="#2c7fb8")
    axes[0].axvline(df["median_house_value"].median(), color="black", ls="--",
                    lw=1.2, label=f"median = ${df['median_house_value'].median():,.0f}")
    axes[0].set_title("Distribution of median house value")
    axes[0].set_xlabel("median_house_value (USD)")
    axes[0].set_ylabel("block groups")
    axes[0].legend()

    sns.boxplot(x=df["median_house_value"], ax=axes[1], color="#9ecae1")
    axes[1].set_title("Spread and outliers")
    axes[1].set_xlabel("median_house_value (USD)")

    fig.suptitle("House value: right-skewed with a hard cap at $500,001", y=1.02)
    path = save_figure(fig, name)
    return fig, path


def plot_feature_distributions(df: pd.DataFrame,
                               columns: list | None = None,
                               name: str = "feature_distributions"):
    """Small-multiple histograms for the main numeric features."""
    columns = columns or [
        "housing_median_age", "total_rooms", "population",
        "households", "median_income", "rooms_per_household",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, col in zip(axes.ravel(), columns):
        sns.histplot(df[col], bins=50, ax=ax, color="#41ab5d")
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Feature distributions (long right tails are common)", y=1.01)
    path = save_figure(fig, name)
    return fig, path


def plot_geography(df: pd.DataFrame, name: str = "housing_geography"):
    """Geographic distribution of records, coloured by house value."""
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        df["longitude"], df["latitude"],
        c=df["median_house_value"], cmap="viridis",
        s=6, alpha=0.55, linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("median_house_value (USD)")
    ax.set_title("California housing block groups\n(colour = house value)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    path = save_figure(fig, name)
    return fig, path


def plot_correlation_heatmap(df: pd.DataFrame, columns: list | None = None,
                             name: str = "correlation_heatmap"):
    """Correlation matrix of the numeric features and the price target."""
    columns = columns or [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "rooms_per_household", "bedrooms_per_room",
        "population_per_household", "median_house_value",
    ]
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix (Pearson)")
    path = save_figure(fig, name)
    return fig, path


# ---------------------------------------------------------------------------
# DBSCAN diagnostics
# ---------------------------------------------------------------------------
def plot_k_distance(k_dist: np.ndarray, k: int, eps: float | None = None,
                    name: str | None = None):
    """k-distance graph with the chosen eps marked."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(np.arange(len(k_dist)), k_dist, color="#08519c", lw=1.5)
    if eps is not None:
        ax.axhline(eps, color="#cb181d", ls="--", lw=1.4,
                   label=f"chosen eps = {eps:.3f}")
        ax.legend()
    ax.set_title(f"k-distance graph (k = {k})\nlook for the knee")
    ax.set_xlabel("points sorted by distance")
    ax.set_ylabel(f"distance to {k}-th nearest neighbour")
    path = save_figure(fig, name or f"k_distance_k{k}")
    return fig, path


def plot_dbscan_grid_heatmap(results: pd.DataFrame, metric: str = "n_clusters",
                             name: str = "dbscan_parameter_grid"):
    """Heatmap of a grid-search metric over eps x min_samples."""
    pivot = results.pivot(index="min_samples", columns="eps", values=metric)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".0f" if metric == "n_clusters" else ".2f",
                cmap="YlGnBu", ax=ax)
    ax.set_title(f"DBSCAN grid search: {metric}")
    ax.set_xlabel("eps")
    ax.set_ylabel("min_samples")
    path = save_figure(fig, name)
    return fig, path


def plot_dbscan_clusters(df: pd.DataFrame, labels: np.ndarray,
                         eps: float | None = None,
                         min_samples: int | None = None,
                         title: str = "DBSCAN clusters",
                         name: str = "dbscan_clusters"):
    """Geographic DBSCAN map; noise points drawn as grey/red crosses.

    Noise is plotted first and smaller so clusters remain readable.
    """
    labels = np.asarray(labels)
    colors = _cluster_colors(labels)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(df.loc[noise_mask, "longitude"], df.loc[noise_mask, "latitude"],
                   c=NOISE_COLOR, s=7, marker="x", alpha=0.55,
                   label=f"noise ({int(noise_mask.sum()):,})")

    for label in sorted(set(labels.tolist())):
        if label == -1:
            continue
        mask = labels == label
        ax.scatter(df.loc[mask, "longitude"], df.loc[mask, "latitude"],
                   c=[colors[label]], s=9, alpha=0.75, linewidths=0,
                   label=f"cluster {label} ({int(mask.sum()):,})")

    n_clusters = len([c for c in set(labels.tolist()) if c != -1])
    subtitle = f"{n_clusters} clusters"
    if eps is not None:
        subtitle += f" | eps={eps}"
    if min_samples is not None:
        subtitle += f" | min_samples={min_samples}"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # 15+ clusters would create an unreadable legend: label the largest
    # clusters at their centroid and summarise noise as text instead.
    if n_clusters <= 15:
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)
    else:
        sizes = pd.Series(labels).value_counts()
        top_labels = [lab for lab in sizes.index if lab != -1][:8]
        for label in top_labels:
            mask = labels == label
            ax.annotate(
                f"C{label}",
                (df.loc[mask, "longitude"].mean(), df.loc[mask, "latitude"].mean()),
                fontsize=8, fontweight="bold", ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )
        ax.text(0.01, 0.01,
                f"noise = {int(noise_mask.sum()):,} points (dark red x)\n"
                f"labels mark the eight largest clusters; all {n_clusters} "
                "clusters are in the profile table",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    path = save_figure(fig, name)
    return fig, path


# ---------------------------------------------------------------------------
# K-Means diagnostics
# ---------------------------------------------------------------------------
def plot_elbow_curve(results: pd.DataFrame, chosen_k: int | None = None,
                     name: str = "kmeans_elbow"):
    """Inertia (elbow) and silhouette on twin axes."""
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.plot(results["k"], results["inertia"], "o-", color="#08519c", lw=2)
    ax1.set_xlabel("number of clusters (K)")
    ax1.set_ylabel("inertia (within-cluster SSE)", color="#08519c")
    ax1.tick_params(axis="y", labelcolor="#08519c")

    ax2 = ax1.twinx()
    ax2.plot(results["k"], results["silhouette"], "s--", color="#cb181d", lw=1.8)
    ax2.set_ylabel("silhouette score", color="#cb181d")
    ax2.tick_params(axis="y", labelcolor="#cb181d")

    if chosen_k is not None:
        ax1.axvline(chosen_k, color="black", ls=":", lw=1.5,
                    label=f"chosen K = {chosen_k}")
        ax1.legend(loc="upper right")

    ax1.set_title("K-Means: Elbow Method and Silhouette Score")
    fig.tight_layout()
    path = save_figure(fig, name)
    return fig, path


def plot_kmeans_clusters(df: pd.DataFrame, labels: np.ndarray,
                         title: str = "K-Means clusters",
                         name: str = "kmeans_clusters"):
    """Geographic K-Means map (every point is assigned)."""
    labels = np.asarray(labels)
    colors = _cluster_colors(labels)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    for label in sorted(set(labels.tolist())):
        mask = labels == label
        ax.scatter(df.loc[mask, "longitude"], df.loc[mask, "latitude"],
                   c=[colors[label]], s=9, alpha=0.7, linewidths=0,
                   label=f"cluster {label} ({int(mask.sum()):,})")

    ax.set_title(f"{title}\n{len(set(labels.tolist()))} clusters, no noise allowed")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if len(colors) <= 15:
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)

    path = save_figure(fig, name)
    return fig, path


def plot_avg_value_by_cluster(profile: pd.DataFrame,
                              title: str = "Average house value by cluster",
                              name: str = "avg_value_by_cluster"):
    """Bar chart of mean house value per cluster, noise clearly separated."""
    data = profile.copy()
    labels = [("noise" if idx == -1 else f"cluster {idx}") for idx in data.index]
    colors = [NOISE_COLOR if idx == -1 else "#2c7fb8" for idx in data.index]

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.55), 4.8))
    bars = ax.bar(labels, data["mean_house_value"], color=colors)
    ax.bar_label(bars, fmt="${:,.0f}", fontsize=7, padding=2)
    ax.set_title(title)
    ax.set_ylabel("mean median_house_value (USD)")
    ax.set_xlabel("cluster")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    path = save_figure(fig, name)
    return fig, path


def plot_cluster_sizes(profile: pd.DataFrame, name: str = "cluster_sizes"):
    """Bar chart of cluster sizes (the geographic footprints)."""
    data = profile.copy()
    labels = [("noise" if idx == -1 else f"cluster {idx}") for idx in data.index]
    colors = [NOISE_COLOR if idx == -1 else "#41ab5d" for idx in data.index]

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.55), 4.5))
    bars = ax.bar(labels, data["n_points"], color=colors)
    ax.bar_label(bars, fmt="{:,.0f}", fontsize=7, padding=2)
    ax.set_title("Cluster sizes")
    ax.set_ylabel("block groups")
    ax.set_xlabel("cluster")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    path = save_figure(fig, name)
    return fig, path


def plot_comparison_bars(comparison: pd.DataFrame,
                         name: str = "algorithm_comparison"):
    """Side-by-side bars comparing the two algorithms on three key metrics."""
    metrics = [
        ("n_clusters", "number of clusters"),
        ("noise_pct", "noise points (%)"),
        ("silhouette", "silhouette score"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (metric, label) in zip(axes, metrics):
        values = comparison[metric].astype(float)
        bars = ax.bar(comparison.index, values,
                      color=["#2c7fb8", "#f16913"][:len(values)])
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
        ax.set_title(label)
        ax.set_ylim(0, max(1e-9, values.max() * 1.22))
        plt.setp(ax.get_xticklabels(), rotation=15)
    fig.suptitle("DBSCAN vs K-Means on the same feature matrix", y=1.03)
    path = save_figure(fig, name)
    return fig, path


def plot_side_by_side_maps(
    df: pd.DataFrame,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    title_a: str,
    title_b: str,
    name: str = "dbscan_vs_kmeans_maps",
):
    """Place two cluster assignments on the same map axes for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), sharex=True, sharey=True)

    for ax, labels, title in zip(axes, [labels_a, labels_b], [title_a, title_b]):
        labels = np.asarray(labels)
        colors = _cluster_colors(labels)

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(df.loc[noise_mask, "longitude"], df.loc[noise_mask, "latitude"],
                       c=NOISE_COLOR, s=5, marker="x", alpha=0.45,
                       label="noise")

        for label in sorted(set(labels.tolist())):
            if label == -1:
                continue
            mask = labels == label
            ax.scatter(df.loc[mask, "longitude"], df.loc[mask, "latitude"],
                       c=[colors[label]], s=5, alpha=0.7, linewidths=0,
                       rasterized=True)

        n_clusters = len(set(labels.tolist()) - {-1})
        ax.set_title(f"{title}\n{n_clusters} clusters | "
                     f"{100 * noise_mask.mean():.2f}% noise")
        ax.set_xlabel("longitude")
        if noise_mask.any():
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("latitude")
    fig.suptitle("Same feature space, two algorithms", y=1.01)
    path = save_figure(fig, name)
    return fig, path


def plot_synthetic_shapes(shapes: np.ndarray, dbscan_labels: np.ndarray,
                          kmeans_labels: np.ndarray,
                          name: str = "synthetic_shape_comparison"):
    """Two-moons illustration: density vs centroid-based cluster shapes.

    This is a clearly-labelled synthetic example used only to explain *why*
    DBSCAN can capture non-convex clusters while K-Means cannot. It is not a
    result of the housing analysis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    for ax, labels, title in zip(
        axes,
        [dbscan_labels, kmeans_labels],
        ["DBSCAN (density-based)", "K-Means (centroid-based)"],
    ):
        colors = _cluster_colors(labels)
        for label in sorted(set(np.asarray(labels).tolist())):
            mask = np.asarray(labels) == label
            ax.scatter(shapes[mask, 0], shapes[mask, 1], s=14,
                       c=[colors[label]], alpha=0.8, linewidths=0,
                       label=("noise" if label == -1 else f"cluster {label}"))
        ax.set_title(title)
        ax.set_xlabel("synthetic feature 1")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("synthetic feature 2")
    fig.suptitle("Why shape matters: two interleaving moons (synthetic data)", y=1.03)
    path = save_figure(fig, name)
    return fig, path


# ---------------------------------------------------------------------------
# Folium interactive maps
# ---------------------------------------------------------------------------
def _popup_html(row: dict, cluster_label: int) -> str:
    cluster_text = "Noise / isolated" if cluster_label == -1 else f"Cluster {cluster_label}"
    return (
        f"<b>{cluster_text}</b><br>"
        f"House value: ${row['median_house_value']:,.0f}<br>"
        f"Median income: {row['median_income']:.2f} (x $10k)<br>"
        f"House age: {row['housing_median_age']:.0f} years<br>"
        f"Rooms/household: {row['rooms_per_household']:.2f}<br>"
        f"Population: {row['population']:,.0f}<br>"
        f"Location: {row['latitude']:.3f}, {row['longitude']:.3f}"
    )


def build_cluster_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    title: str,
    filename: str,
    sample_n: int | None = 5000,
    legend_limit: int = 12,
) -> Path:
    """Build and save a Folium map with cluster-coloured circle markers.

    ``sample_n`` defaults to 5,000 markers: 20,640 individual HTML markers make
    the browser sluggish. Sampling is stratified by cluster so rare clusters are
    still represented. Set ``sample_n=None`` to plot every record.
    """
    ensure_output_dirs()
    df = df.reset_index(drop=True)
    labels = np.asarray(labels)

    sampled_df, sampled_labels = _stratified_sample(df, labels, sample_n)
    colors = _cluster_colors(labels)

    center = [float(df["latitude"].mean()), float(df["longitude"].mean())]
    fmap = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap",
                      control_scale=True)

    # One FeatureGroup per cluster keeps the layer control usable.
    groups = {}
    for label in sorted(set(labels.tolist())):
        name = "Noise / isolated" if label == -1 else f"Cluster {label}"
        groups[label] = folium.FeatureGroup(name=name, show=True)

    for row, label in zip(sampled_df.to_dict("records"), sampled_labels):
        if label == -1:
            style = dict(radius=2, color="#4d0000", weight=0,
                         fill=True, fill_opacity=0.5, fill_color="#7f0000")
        else:
            style = dict(radius=3, color=None, weight=0,
                         fill=True, fill_opacity=0.65,
                         fill_color=_to_hex(colors[label]))
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(_popup_html(row, int(label)), max_width=230),
            tooltip=("Noise" if label == -1 else f"Cluster {int(label)}"),
            **style,
        ).add_to(groups[label])

    for group in groups.values():
        group.add_to(fmap)

    # Show the legend as a collapsible panel; hide it when it would be huge.
    if len(groups) <= legend_limit:
        folium.LayerControl(collapsed=False).add_to(fmap)
    else:
        folium.LayerControl(collapsed=True).add_to(fmap)

    title_html = (
        '<div style="position: fixed; top: 10px; left: 60px; z-index: 9999; '
        'background: white; padding: 8px 14px; border-radius: 6px; '
        'box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-family: sans-serif;">'
        f'<b>{title}</b><br><span style="font-size: 11px;">'
        f"{len(sampled_df):,} of {len(df):,} block groups shown "
        f"(stratified sample)</span></div>"
    )
    fmap.get_root().html.add_child(folium.Element(title_html))

    path = MAP_DIR / filename
    fmap.save(str(path))
    return path


def build_geography_map(
    df: pd.DataFrame,
    filename: str = "housing_geography.html",
    sample_n: int = 8000,
) -> Path:
    """Folium map of raw housing records coloured by house value."""
    ensure_output_dirs()
    sampled, _ = _stratified_sample(
        df.reset_index(drop=True), np.zeros(len(df), dtype=int), sample_n
    )

    fmap = folium.Map(
        location=[float(df["latitude"].mean()), float(df["longitude"].mean())],
        zoom_start=5, tiles="OpenStreetMap", control_scale=True,
    )
    values = sampled["median_house_value"]
    norm = Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.get_cmap("viridis")

    for row in sampled.itertuples(index=False):
        color = _to_hex(cmap(norm(row.median_house_value)))
        rooms_per_household = (
            row.rooms_per_household
            if hasattr(row, "rooms_per_household")
            else row.total_rooms / max(row.households, 1)
        )
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=3, color=None, weight=0, fill=True, fill_opacity=0.7,
            fill_color=color,
            popup=folium.Popup(
                f"<b>${row.median_house_value:,.0f}</b><br>"
                f"Median income: {row.median_income:.2f}<br>"
                f"Rooms/household: {rooms_per_household:.2f}",
                max_width=200,
            ),
        ).add_to(fmap)

    title_html = (
        '<div style="position: fixed; top: 10px; left: 60px; z-index: 9999; '
        'background: white; padding: 8px 14px; border-radius: 6px; '
        'box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-family: sans-serif;">'
        "<b>California housing records</b><br>"
        '<span style="font-size: 11px;">colour = house value '
        "(dark purple: cheap, yellow: expensive) - "
        f"{len(sampled):,} sampled points</span></div>"
    )
    fmap.get_root().html.add_child(folium.Element(title_html))

    path = MAP_DIR / filename
    fmap.save(str(path))
    return path


def build_noise_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    filename: str = "dbscan_noise.html",
    sample_n: int = 5000,
) -> Path:
    """Folium map that emphasises DBSCAN noise against a faint cluster backdrop."""
    ensure_output_dirs()
    df = df.reset_index(drop=True)
    labels = np.asarray(labels)
    sampled, sampled_labels = _stratified_sample(df, labels, sample_n)
    colors = _cluster_colors(labels)

    fmap = folium.Map(
        location=[float(df["latitude"].mean()), float(df["longitude"].mean())],
        zoom_start=5, tiles="OpenStreetMap", control_scale=True,
    )

    clustered = folium.FeatureGroup(name="Clustered points (faint)")
    noise = folium.FeatureGroup(name="Noise / isolated points", show=True)

    for row, label in zip(sampled.itertuples(index=False), sampled_labels):
        if label == -1:
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=4, color="#7f0000", weight=1, fill=True,
                fill_color="#cb181d", fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>Noise point</b><br>House value: "
                    f"${row.median_house_value:,.0f}<br>"
                    f"Lat/Lon: {row.latitude:.3f}, {row.longitude:.3f}",
                    max_width=220,
                ),
            ).add_to(noise)
        else:
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=2, color=None, weight=0, fill=True,
                fill_color=_to_hex(colors[label]), fill_opacity=0.18,
            ).add_to(clustered)

    clustered.add_to(fmap)
    noise.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    title_html = (
        '<div style="position: fixed; top: 10px; left: 60px; z-index: 9999; '
        'background: white; padding: 8px 14px; border-radius: 6px; '
        'box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-family: sans-serif;">'
        "<b>DBSCAN noise points</b><br>"
        '<span style="font-size: 11px;">dark red = block groups that do not '
        "belong to any dense neighbourhood</span></div>"
    )
    fmap.get_root().html.add_child(folium.Element(title_html))

    path = MAP_DIR / filename
    fmap.save(str(path))
    return path


def _to_hex(color) -> str:
    """Convert an RGB(A) tuple from matplotlib to a hex string for Folium."""
    if isinstance(color, str):
        return color
    rgba = tuple(color)
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgba[:3])


# ---------------------------------------------------------------------------
# Plotly interactive charts
# ---------------------------------------------------------------------------
def _cluster_series(labels: np.ndarray) -> list[str]:
    return ["Noise (-1)" if label == -1 else f"Cluster {label}"
            for label in np.asarray(labels)]


def save_plotly_cluster_scatter(
    df: pd.DataFrame,
    labels: np.ndarray,
    title: str,
    filename: str,
    algorithm: str = "DBSCAN",
) -> Path:
    """Interactive geographic scatter (WebGL) with rich hover tooltips."""
    ensure_output_dirs()
    data = df.reset_index(drop=True).copy()
    data["cluster"] = _cluster_series(labels)

    fig = px.scatter_map(
        data,
        lat="latitude",
        lon="longitude",
        color="cluster",
        color_discrete_sequence=(
            [NOISE_COLOR] + [c for c in CLUSTER_PALETTE]
        ),
        hover_name="cluster",
        hover_data={
            "median_house_value": ":,.0f",
            "median_income": ":.2f",
            "housing_median_age": ":.0f",
            "rooms_per_household": ":.2f",
            "population": ":,.0f",
            "latitude": ":.3f",
            "longitude": ":.3f",
        },
        zoom=4.6,
        center={"lat": float(data["latitude"].mean()),
                "lon": float(data["longitude"].mean())},
        height=650,
        opacity=0.65,
        map_style="open-street-map",
        title=title,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text=f"{algorithm} cluster",
    )

    path = FIGURE_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def save_plotly_price_chart(df: pd.DataFrame,
                            filename: str = "price_distribution_interactive.html"
                            ) -> Path:
    """Interactive price histogram with income hover and box plot."""
    ensure_output_dirs()
    data = df.copy()

    fig = px.histogram(
        data, x="median_house_value", nbins=80,
        color="ocean_proximity",
        title="Median house value distribution by ocean proximity",
        labels={"median_house_value": "median house value (USD)",
                "ocean_proximity": "ocean proximity"},
        height=520,
    )
    fig.update_layout(bargap=0.05, legend_title_text="ocean proximity",
                      xaxis_tickformat="$,.0f")

    path = FIGURE_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def save_plotly_cluster_scatter_2d(
    df: pd.DataFrame,
    labels: np.ndarray,
    x: str = "median_income",
    y: str = "median_house_value",
    filename: str = "cluster_scatter_interactive.html",
    title: str = "Clusters in feature space",
) -> Path:
    """Interactive 2-D scatter (income vs price) coloured by cluster."""
    ensure_output_dirs()
    data = df.copy()
    data["cluster"] = _cluster_series(labels)

    fig = px.scatter(
        data, x=x, y=y, color="cluster",
        color_discrete_sequence=[NOISE_COLOR] + list(CLUSTER_PALETTE),
        hover_data=["housing_median_age", "rooms_per_household",
                    "population_per_household", "latitude", "longitude"],
        opacity=0.55, height=560,
        title=title,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend_title_text="cluster")
    if y == "median_house_value":
        fig.update_yaxes(tickformat="$,.0f")

    path = FIGURE_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def save_plotly_bar(
    profile: pd.DataFrame,
    value_col: str = "mean_house_value",
    filename: str = "avg_value_by_cluster_interactive.html",
    title: str = "Average house value by cluster",
) -> Path:
    """Interactive bar chart of a per-cluster statistic."""
    ensure_output_dirs()
    data = profile.copy()
    data["cluster_name"] = [
        "Noise (-1)" if idx == -1 else f"Cluster {idx}" for idx in data.index
    ]

    fig = px.bar(
        data, x="cluster_name", y=value_col,
        color="cluster_name",
        color_discrete_sequence=[NOISE_COLOR] + list(CLUSTER_PALETTE),
        hover_data=["n_points", "mean_income", "median_house_value"],
        title=title, height=480,
    )
    fig.update_layout(showlegend=False, xaxis_title="cluster",
                      yaxis_title=value_col.replace("_", " "))
    if "value" in value_col:
        fig.update_yaxes(tickformat="$,.0f")

    path = FIGURE_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    return path
