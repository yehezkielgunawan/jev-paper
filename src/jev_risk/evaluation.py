"""Shared classification, selective-prediction, latency, and bootstrap metrics."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from jev_risk.predictions import validate_prediction_comparison

METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
)


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> dict[str, float]:
    """Calculate binary classification and probability metrics."""
    truth = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    if len(truth) == 0 or len({len(truth), len(predicted), len(probability)}) != 1:
        raise ValueError("y_true, y_pred, and y_prob must have the same nonzero length")
    if not np.isin(truth, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
        raise ValueError("Classification labels must be binary 0 and 1")
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    has_both_classes = len(np.unique(truth)) == 2
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probability)) if has_both_classes else float("nan"),
        "average_precision": (
            float(average_precision_score(truth, probability)) if has_both_classes else float("nan")
        ),
        "brier_score": float(brier_score_loss(truth, probability)),
    }


def confidence_threshold_metrics(
    predictions: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Report covered-case performance while counting all rejected rows as review."""
    if "confidence" not in predictions:
        raise ValueError("Confidence threshold evaluation requires a confidence column")
    if predictions.empty:
        raise ValueError("Confidence threshold evaluation requires at least one prediction")
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence thresholds must be between 0 and 1")
        accepted = predictions.loc[predictions["confidence"] >= threshold]
        row: dict[str, Any] = {
            "threshold": float(threshold),
            "coverage": float(len(accepted) / len(predictions)),
            "accepted_count": len(accepted),
            "review_count": int(len(predictions) - len(accepted)),
        }
        if accepted.empty:
            row.update({name: float("nan") for name in METRIC_NAMES})
        else:
            row.update(
                calculate_metrics(accepted["y_true"], accepted["y_pred"], accepted["y_prob"])
            )
        rows.append(row)
    return pd.DataFrame(rows)


def latency_summary(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize per-record wall-clock inference latency for each model."""
    rows = []
    for model, frame in predictions.items():
        values = pd.to_numeric(frame["latency_ms"], errors="coerce").to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{model} latency values must be non-empty, finite, and nonnegative")
        rows.append(
            {
                "model": model,
                "mean_latency_ms": float(np.mean(values)),
                "median_latency_ms": float(np.median(values)),
                "p95_latency_ms": float(np.percentile(values, 95)),
                "n_predictions": len(values),
                "latency_type": "hosted_api_round_trip"
                if model == "jev"
                else "local_per_record_inference",
            }
        )
    return pd.DataFrame(rows)


def evaluate_models(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create the paper's shared test-set model comparison table."""
    if len(predictions) > 1:
        validate_prediction_comparison(predictions)
    rows = []
    for model, frame in predictions.items():
        metrics = calculate_metrics(frame["y_true"], frame["y_pred"], frame["y_prob"])
        latency = latency_summary({model: frame}).iloc[0]
        rows.append(
            {
                "model": model,
                **metrics,
                "mean_latency_ms": float(latency["mean_latency_ms"]),
                "median_latency_ms": float(latency["median_latency_ms"]),
                "p95_latency_ms": float(latency["p95_latency_ms"]),
                "n_predictions": int(latency["n_predictions"]),
            }
        )
    return pd.DataFrame(rows)


def _metrics_for_frame(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, float]:
    selected = frame.iloc[indices]
    return calculate_metrics(selected["y_true"], selected["y_pred"], selected["y_prob"])


def bootstrap_metric_intervals(
    predictions: Mapping[str, pd.DataFrame],
    iterations: int = 1_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute percentile CIs and paired differences using common resampled IDs."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if len(predictions) < 2:
        raise ValueError("At least two models are required for paired bootstrap comparisons")
    validate_prediction_comparison(predictions)
    reference = next(iter(predictions.values()))
    sample_ids = reference["sample_id"].tolist()
    aligned = {
        model: frame.set_index("sample_id").loc[sample_ids].reset_index()
        for model, frame in predictions.items()
    }
    rng = np.random.default_rng(random_state)
    values: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in METRIC_NAMES} for model in aligned
    }
    pair_values = {
        (left, right): {metric: [] for metric in METRIC_NAMES}
        for left, right in itertools.combinations(aligned, 2)
    }
    sample_count = len(sample_ids)
    for _ in range(iterations):
        indices = rng.integers(0, sample_count, size=sample_count)
        iteration_metrics = {
            model: _metrics_for_frame(frame, indices) for model, frame in aligned.items()
        }
        for model, metrics in iteration_metrics.items():
            for metric in METRIC_NAMES:
                values[model][metric].append(metrics[metric])
        for (left, right), metric_samples in pair_values.items():
            for metric in METRIC_NAMES:
                left_value = iteration_metrics[left][metric]
                right_value = iteration_metrics[right][metric]
                metric_samples[metric].append(left_value - right_value)

    rows: list[dict[str, Any]] = []
    for model, frame in aligned.items():
        point = calculate_metrics(frame["y_true"], frame["y_pred"], frame["y_prob"])
        for metric in METRIC_NAMES:
            low, high = _percentile_interval(values[model][metric])
            rows.append(
                {
                    "model": model,
                    "comparison": "",
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_lower": low,
                    "ci_upper": high,
                    "bootstrap_iterations": iterations,
                }
            )
    for (left, right), metric_samples in pair_values.items():
        left_point = calculate_metrics(
            aligned[left]["y_true"], aligned[left]["y_pred"], aligned[left]["y_prob"]
        )
        right_point = calculate_metrics(
            aligned[right]["y_true"], aligned[right]["y_pred"], aligned[right]["y_prob"]
        )
        for metric in METRIC_NAMES:
            low, high = _percentile_interval(metric_samples[metric])
            rows.append(
                {
                    "model": "",
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "estimate": left_point[metric] - right_point[metric],
                    "ci_lower": low,
                    "ci_upper": high,
                    "bootstrap_iterations": iterations,
                }
            )
    return pd.DataFrame(rows)


def _percentile_interval(samples: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)
