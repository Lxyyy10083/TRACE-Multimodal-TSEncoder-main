import os

import numpy as np
import pandas as pd


DOMAIN_NAME_TO_ID = {
    "Agriculture": 0,
    "Climate": 1,
    "Economy": 2,
    "Energy": 3,
    "Environment": 4,
    "Health": 5,
    "Security": 6,
    "SocialGood": 7,
    "Traffic": 8,
    "Weather": 9,
}

DOMAIN_ALIASES = {name.lower(): name for name in DOMAIN_NAME_TO_ID}
DOMAIN_ALIASES.update(
    {
        "social_good": "SocialGood",
        "social-good": "SocialGood",
        "social good": "SocialGood",
    }
)

# Keep this list stable. All domains use the same vector shape; unsupported
# features are left at zero.
TIME_FEATURE_NAMES = [
    "month",
    "quarter",
    "season",
    "day_of_year_sin",
    "day_of_year_cos",
    "weekday",
    "is_weekend",
    "week_of_year",
    "hour_sin",
    "hour_cos",
    "solar_term_id",
    "heating_or_cooling_season",
]

BASE_FEATURE_WEIGHT = 1.0
FOCUS_FEATURE_WEIGHT = 2.0


def infer_domain_name(data_name=None, file_path=None):
    """Infer a canonical TimeMMD domain name from a dataset name or CSV path."""
    candidates = []
    if data_name is not None:
        candidates.append(str(data_name))
    if file_path is not None:
        candidates.append(os.path.splitext(os.path.basename(str(file_path)))[0])

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in DOMAIN_ALIASES:
            return DOMAIN_ALIASES[key]

    return "Unknown"


def infer_time_granularity(dates: pd.Series):
    """Infer coarse dataset frequency from standardized timestamps."""
    valid_dates = pd.Series(dates).dropna().sort_values()
    if valid_dates.empty:
        return "unknown"

    has_hour_info = (
        (valid_dates.dt.hour != 0)
        | (valid_dates.dt.minute != 0)
        | (valid_dates.dt.second != 0)
    ).any()
    if has_hour_info:
        return "hourly"

    deltas = valid_dates.diff().dropna()
    if deltas.empty:
        return "unknown"

    median_days = deltas.dt.total_seconds().median() / 86400.0
    if 27 <= median_days <= 32:
        return "monthly"
    if median_days <= 1.5:
        return "daily"
    return "unknown"


def _season_from_month(month):
    """Map calendar month to a coarse meteorological season id."""
    # DJF=0, MAM=1, JJA=2, SON=3.
    return ((month % 12) // 3).astype(float)


def _solar_term_id(dates: pd.Series):
    """Approximate a 24 solar-term id from month and half-month position."""
    # Approximate 24 solar terms as half-month buckets. This is only used as a
    # coarse agricultural prior for daily/hourly data, not as an astronomy table.
    day_bucket = (dates.dt.day > 15).astype(int)
    return ((dates.dt.month - 1) * 2 + day_bucket).astype(float)


def _domain_focus_feature_names(domain_name, granularity):
    """Return the feature names that should receive extra domain weight."""
    if domain_name == "Agriculture":
        focus = ["month", "season", "day_of_year_sin", "day_of_year_cos"]
        if granularity in {"daily", "hourly"}:
            focus.append("solar_term_id")
        else:
            focus.append("quarter")
        return focus
    if domain_name == "Traffic":
        return ["weekday", "is_weekend", "week_of_year", "hour_sin", "hour_cos"]
    if domain_name == "Energy":
        return [
            "month",
            "season",
            "hour_sin",
            "hour_cos",
            "heating_or_cooling_season",
        ]
    if domain_name in {"Weather", "Climate", "Environment"}:
        return ["season", "month", "day_of_year_sin", "day_of_year_cos"]
    if domain_name == "Health":
        return ["month", "season", "weekday"]
    if domain_name in {"Economy", "Security", "SocialGood"}:
        return ["quarter", "month", "week_of_year"]
    return []


def _build_feature_weight(domain_name, granularity):
    """Create a domain-specific weight vector with the same order as features."""
    weights = np.full(len(TIME_FEATURE_NAMES), BASE_FEATURE_WEIGHT, dtype=np.float32)
    focus_names = _domain_focus_feature_names(domain_name, granularity)
    for feature_name in focus_names:
        if feature_name in TIME_FEATURE_NAMES:
            weights[TIME_FEATURE_NAMES.index(feature_name)] = FOCUS_FEATURE_WEIGHT
    return weights, focus_names


def build_time_prior_features(df, data_name=None, file_path=None):
    """Build fixed-width time prior features, weights, and domain ids from df["date"].

    The input dataframe must already contain a standardized datetime column named
    "date". Feature values are numeric and normalized where appropriate. Features
    that do not apply to a dataset granularity are set to zero.
    """
    if "date" not in df.columns:
        raise ValueError('build_time_prior_features requires standardized df["date"].')

    dates = pd.to_datetime(df["date"], errors="coerce")
    domain_name = infer_domain_name(data_name=data_name, file_path=file_path)
    domain_id = DOMAIN_NAME_TO_ID.get(domain_name, -1)
    granularity = infer_time_granularity(dates)

    day_of_year = dates.dt.dayofyear.astype(float)
    year_length = np.where(dates.dt.is_leap_year.to_numpy(), 366.0, 365.0)
    month = dates.dt.month.astype(float)
    quarter = dates.dt.quarter.astype(float)
    weekday = dates.dt.weekday.astype(float)
    iso_week = dates.dt.isocalendar().week.astype(float)
    hour = dates.dt.hour.astype(float)

    feature_map = {
        "month": month / 12.0,
        "quarter": quarter / 4.0,
        "season": _season_from_month(month) / 3.0,
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / year_length),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / year_length),
        "weekday": weekday / 6.0,
        "is_weekend": (weekday >= 5).astype(float),
        "week_of_year": iso_week / 53.0,
        "hour_sin": np.zeros(len(df), dtype=float),
        "hour_cos": np.zeros(len(df), dtype=float),
        "solar_term_id": np.zeros(len(df), dtype=float),
        "heating_or_cooling_season": month.isin([1, 2, 6, 7, 8, 12]).astype(float),
    }

    if granularity == "hourly":
        feature_map["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        feature_map["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    if domain_name == "Agriculture" and granularity in {"daily", "hourly"}:
        feature_map["solar_term_id"] = _solar_term_id(dates) / 23.0

    time_feat = np.stack(
        [np.asarray(feature_map[name], dtype=np.float32) for name in TIME_FEATURE_NAMES],
        axis=1,
    )
    base_weight, focus_names = _build_feature_weight(domain_name, granularity)
    time_feat_weight = np.repeat(base_weight.reshape(1, -1), len(df), axis=0)

    print(f"[Time prior] domain: {domain_name} (id={domain_id})")
    if len(dates.dropna()) > 0:
        print(f"[Time prior] date range: {dates.min()} -> {dates.max()}")
    else:
        print("[Time prior][warning] date range unavailable; no valid dates.")
    print(f"[Time prior] granularity: {granularity}")
    print(f"[Time prior] time_feat_dim: {len(TIME_FEATURE_NAMES)}")
    print(f"[Time prior] focus features: {focus_names}")

    return {
        "time_feat": time_feat.astype(np.float32),
        "time_feat_weight": time_feat_weight.astype(np.float32),
        "domain_id": int(domain_id),
        "domain_name": domain_name,
        "granularity": granularity,
        "feature_names": TIME_FEATURE_NAMES,
        "focus_features": focus_names,
    }
