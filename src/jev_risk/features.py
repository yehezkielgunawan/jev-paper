"""Shared leakage-safe modeling and Jev input feature construction."""

from __future__ import annotations

import pandas as pd

from jev_risk.config import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen primary feature set, excluding IDs and post-event fields."""
    required = set(CATEGORICAL_FEATURES).union(
        {"Date", "Amount", "System_Latency", "Login_Frequency", "Failed_Attempts"}
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot build features; missing columns: {', '.join(missing)}")

    features = frame.loc[:, list(CATEGORICAL_FEATURES)].copy()
    for column in ("Amount", "System_Latency", "Login_Frequency", "Failed_Attempts"):
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & numeric_values.isna()
        if invalid.any():
            examples = frame.loc[invalid, column].astype(str).head(3).tolist()
            raise ValueError(f"{column} contains invalid numeric values: {examples}")
        features[column] = numeric_values

    raw_dates = frame["Date"]
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce", format="%Y-%m-%d")
    invalid = raw_dates.notna() & parsed_dates.isna()
    if invalid.any():
        examples = raw_dates.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(f"Date contains invalid values: {examples}")

    features["date_year"] = parsed_dates.dt.year
    features["date_month"] = parsed_dates.dt.month
    features["date_dayofweek"] = parsed_dates.dt.dayofweek
    features["date_dayofyear"] = parsed_dates.dt.dayofyear

    return features.loc[:, list(MODEL_FEATURES)]


__all__ = ["CATEGORICAL_FEATURES", "NUMERIC_FEATURES", "build_features"]
