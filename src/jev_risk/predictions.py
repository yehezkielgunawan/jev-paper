"""Standard prediction records and cross-model consistency validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PREDICTION_COLUMNS = ("sample_id", "y_true", "y_pred", "y_prob", "latency_ms")


def build_prediction_frame(
    sample_ids: Sequence[object],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
    latency_ms: Sequence[float],
    confidence: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build and validate the shared output schema used by all models."""
    values = {
        "sample_id": list(sample_ids),
        "y_true": list(y_true),
        "y_pred": list(y_pred),
        "y_prob": list(y_prob),
        "latency_ms": list(latency_ms),
    }
    if confidence is not None:
        values["confidence"] = list(confidence)
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise ValueError("Prediction fields must all contain the same number of rows")

    predictions = pd.DataFrame(values)
    if predictions["sample_id"].isna().any() or predictions["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be present and unique")
    for column in ("y_true", "y_pred"):
        if not predictions[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only binary 0 and 1 labels")
        predictions[column] = predictions[column].astype("int8")
    for column in ("y_prob", "latency_ms"):
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
        if predictions[column].isna().any() or not np.isfinite(predictions[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
    if not predictions["y_prob"].between(0, 1).all():
        raise ValueError("y_prob must be between 0 and 1")
    if (predictions["latency_ms"] < 0).any():
        raise ValueError("latency_ms cannot be negative")
    if "confidence" in predictions:
        predictions["confidence"] = pd.to_numeric(predictions["confidence"], errors="coerce")
        if (
            predictions["confidence"].isna().any()
            or not np.isfinite(predictions["confidence"]).all()
            or not predictions["confidence"].between(0, 1).all()
        ):
            raise ValueError("confidence must contain finite values between 0 and 1")
    return predictions


def validate_prediction_comparison(predictions: Mapping[str, pd.DataFrame]) -> None:
    """Ensure every model was evaluated against the same labeled test rows."""
    if len(predictions) < 2:
        raise ValueError("At least two model prediction sets are required for comparison")
    reference_name, reference = next(iter(predictions.items()))
    required = set(BASE_PREDICTION_COLUMNS)
    for model_name, frame in predictions.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{model_name} predictions are missing: {', '.join(sorted(missing))}")
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise ValueError(f"{model_name} predictions contain missing or duplicate sample IDs")
    expected = set(reference["sample_id"])
    for model_name, frame in list(predictions.items())[1:]:
        if set(frame["sample_id"]) != expected:
            raise ValueError(
                f"All models must use the same sample IDs ({reference_name} vs {model_name})"
            )
        labels = reference.set_index("sample_id")["y_true"].sort_index()
        other_labels = frame.set_index("sample_id")["y_true"].sort_index()
        if not labels.equals(other_labels):
            raise ValueError(f"Models have inconsistent ground-truth labels ({model_name})")


def save_predictions(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write one model's predictions to CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
