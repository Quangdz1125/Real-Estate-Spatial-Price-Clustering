"""Data loading, cleaning, feature engineering and scaling.

The module is the single entry point used by every notebook so that all
notebooks operate on exactly the same cleaned dataset and feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, never machine-specific)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "visualizations"
MAP_DIR = OUTPUT_DIR / "maps"
DATA_PATH = DATA_DIR / "housing.csv"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
GEO_FEATURES = ["longitude", "latitude"]

HOUSING_FEATURES = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

ENGINEERED_FEATURES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]

TARGET = "median_house_value"

#: Clustering feature sets. ``geo_housing`` is the default because it expresses
#: "where" a block group is *and* "what it looks like", without using the price
#: we later analyse (which would make the price profile circular).
FEATURE_SETS = {
    "geo": GEO_FEATURES,
    "housing": HOUSING_FEATURES + ENGINEERED_FEATURES,
    "geo_housing": GEO_FEATURES + HOUSING_FEATURES + ENGINEERED_FEATURES,
    "geo_housing_value": GEO_FEATURES + HOUSING_FEATURES + ENGINEERED_FEATURES + [TARGET],
}

DEFAULT_FEATURE_SET = "geo_housing"

#: Ratio features are winsorised: a handful of non-residential block groups
#: report values such as 1,243 people per household, which otherwise dominate
#: Euclidean distances.
RATIO_FEATURES = ENGINEERED_FEATURES


@dataclass
class Dataset:
    """Container holding the cleaned frame and its scaled feature matrix."""

    df: pd.DataFrame
    X: np.ndarray
    features: list
    scaler: StandardScaler
    feature_set: str

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Dataset(feature_set={self.feature_set!r}, rows={len(self.df)}, "
            f"features={len(self.features)})"
        )


def ensure_output_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (DATA_DIR, FIGURE_DIR, MAP_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load ``housing.csv`` and validate that the expected columns exist.

    A clear error with download instructions is raised when the file is
    missing, instead of silently returning an empty frame.
    """
    path = Path(path) if path is not None else DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download 'housing.csv' from the Kaggle 'California Housing Prices' "
            "dataset (https://www.kaggle.com/datasets/camnugent/california-housing-prices) "
            f"and place it in: {DATA_DIR}"
        )

    df = pd.read_csv(path)

    required = ["longitude", "latitude", "housing_median_age", "total_rooms",
                "total_bedrooms", "population", "households", "median_income",
                TARGET, "ocean_proximity"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"The file {path.name} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError(f"The dataset at {path} is empty.")

    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicates and impute missing ``total_bedrooms`` values.

    ``total_bedrooms`` is the only column with missing values. It is imputed by
    the median of the same ``ocean_proximity`` group, because bedroom counts
    differ systematically between inland and coastal areas. The overall median
    is used as a fallback for any group that is entirely missing.

    Returns the cleaned frame plus a small report of what was changed.
    """
    df = df.copy()
    report = {}

    n_before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = n_before - len(df)
    report["rows_after_dedup"] = len(df)

    report["missing_before"] = int(df.isna().sum().sum())

    if "total_bedrooms" in df.columns and df["total_bedrooms"].isna().any():
        overall_median = df["total_bedrooms"].median()
        if "ocean_proximity" in df.columns:
            grouped = df.groupby("ocean_proximity")["total_bedrooms"].transform("median")
            df["total_bedrooms"] = df["total_bedrooms"].fillna(grouped)
        df["total_bedrooms"] = df["total_bedrooms"].fillna(overall_median)

    # Numeric safety net: only the imputed column can realistically still be
    # missing, but this keeps the pipeline robust to slightly different files.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isna().any().any():
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    report["missing_after"] = int(df.isna().sum().sum())

    return df, report


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add housing-density ratio features.

    Raw totals are heavily driven by block-group size; the ratios describe the
    *character* of the housing instead (crowding, bedroom share, rooms per home).
    """
    df = df.copy()

    rooms = df["total_rooms"].replace(0, np.nan)
    households = df["households"].replace(0, np.nan)
    population = df["population"]

    df["rooms_per_household"] = df["total_rooms"] / households
    df["bedrooms_per_room"] = df["total_bedrooms"] / rooms
    df["population_per_household"] = population / households

    # A block group cannot have fewer rooms than households, but a few rows in
    # this dataset are non-residential; clip to physically plausible bounds.
    df["bedrooms_per_room"] = df["bedrooms_per_room"].clip(upper=1.0)

    return df


def cap_ratio_outliers(
    df: pd.DataFrame,
    columns: list | None = None,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
    """Winsorise ratio features at the 1st/99th percentiles.

    Extreme ratio values are almost always small non-residential block groups
    (for example a catchment with 1,243 people per household). Left untouched,
    these few rows pull K-Means centroids and blur DBSCAN neighbourhoods.
    Raw count columns are intentionally *not* clipped: large totals are genuine
    dense urban districts and carry the signal we care about.
    """
    df = df.copy()
    columns = columns or RATIO_FEATURES
    report = {}

    for col in columns:
        if col not in df.columns:
            continue
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        n_clipped = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
        report[col] = {
            "lower": float(lower),
            "upper": float(upper),
            "values_clipped": n_clipped,
        }

    return df, report


def build_feature_matrix(
    df: pd.DataFrame,
    features: list | None = None,
    weights: dict | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Standardise selected features and optionally apply per-feature weights.

    StandardScaler is required: ``total_rooms`` spans 2-39,320 while longitude
    spans about 10 degrees. In raw units, Euclidean distance would be entirely
    determined by the room counts and geography would be ignored.

    ``weights`` multiplies a standardised column, letting a caller emphasise
    geography (e.g. ``{"longitude": 2, "latitude": 2}``). The default is None
    (all features weigh equally).
    """
    features = features or FEATURE_SETS[DEFAULT_FEATURE_SET]

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(
            f"Features not found in dataframe: {missing}. "
            "Did you forget to call add_features()?"
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    if weights:
        for idx, col in enumerate(features):
            X[:, idx] *= float(weights.get(col, 1.0))

    return X, scaler


def prepare_dataset(
    feature_set: str = DEFAULT_FEATURE_SET,
    clip_ratios: bool = True,
    weights: dict | None = None,
) -> Dataset:
    """Run the full preprocessing pipeline and return a :class:`Dataset`."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}. "
            f"Choose one of {list(FEATURE_SETS)}."
        )

    ensure_output_dirs()

    df = load_data()
    df, _ = clean_data(df)
    df = add_features(df)

    if clip_ratios:
        df, _ = cap_ratio_outliers(df)

    features = FEATURE_SETS[feature_set]
    X, scaler = build_feature_matrix(df, features, weights=weights)

    return Dataset(df=df, X=X, features=features, scaler=scaler,
                   feature_set=feature_set)


def save_processed(df: pd.DataFrame, filename: str = "housing_processed.csv") -> Path:
    """Persist the cleaned/engineered frame so it can be inspected outside code."""
    ensure_output_dirs()
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    ds = prepare_dataset()
    print(ds)
    print(ds.df.head())
