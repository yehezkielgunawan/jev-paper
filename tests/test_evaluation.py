from __future__ import annotations

import numpy as np
import pytest

from jev_risk.evaluation import (
    bootstrap_metric_intervals,
    calculate_metrics,
    confidence_threshold_metrics,
    evaluate_models,
    latency_summary,
)
from jev_risk.predictions import build_prediction_frame


def test_calculate_metrics_reports_classification_and_probability_scores():
    metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.7, 0.8, 0.9])

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.8)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1
    assert metrics["brier_score"] == pytest.approx(
        np.mean((np.array([0.1, 0.7, 0.8, 0.9]) - [0, 0, 1, 1]) ** 2)
    )


def test_calculate_metrics_returns_nan_for_undefined_single_class_ranking():
    metrics = calculate_metrics([0, 0], [0, 1], [0.2, 0.8])

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["average_precision"])


def test_confidence_threshold_metrics_counts_abstentions_as_review():
    jev = build_prediction_frame(
        ["a", "b", "c", "d"],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0.1, 0.9, 0.8, 0.2],
        [10, 20, 30, 40],
        confidence=[0.95, 0.8, 0.6, 0.2],
    )

    results = confidence_threshold_metrics(jev, thresholds=[0.7, 0.9])

    assert results["coverage"].tolist() == [0.5, 0.25]
    assert results["accepted_count"].tolist() == [2, 1]
    assert results["review_count"].tolist() == [2, 3]
    assert results.loc[0, "accuracy"] == 1.0
    assert results.loc[1, "accuracy"] == 1.0


def test_confidence_threshold_with_no_accepted_rows_reports_nan_metrics():
    jev = build_prediction_frame(["a"], [1], [1], [0.9], [1], confidence=[0.2])

    result = confidence_threshold_metrics(jev, thresholds=[0.9]).iloc[0]

    assert result["coverage"] == 0
    assert result["review_count"] == 1
    assert np.isnan(result["accuracy"])


def test_latency_summary_reports_mean_median_and_p95():
    frame = build_prediction_frame(
        ["a", "b", "c", "d"],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        [1, 2, 3, 100],
    )

    result = latency_summary({"test": frame}).iloc[0]

    assert result["mean_latency_ms"] == pytest.approx(26.5)
    assert result["median_latency_ms"] == pytest.approx(2.5)
    assert result["p95_latency_ms"] == pytest.approx(85.45)


def test_bootstrap_intervals_are_seeded_and_paired_differences_are_zero():
    frame = build_prediction_frame(
        [f"id-{i}" for i in range(20)],
        [i % 2 for i in range(20)],
        [i % 2 for i in range(20)],
        [0.1 if i % 2 == 0 else 0.9 for i in range(20)],
        [1] * 20,
    )

    first = bootstrap_metric_intervals(
        {"a": frame, "b": frame.copy()}, iterations=100, random_state=3
    )
    second = bootstrap_metric_intervals(
        {"a": frame, "b": frame.copy()}, iterations=100, random_state=3
    )

    assert first.equals(second)
    paired = first[first["comparison"] == "a - b"]
    assert (paired["estimate"] == 0).all()
    assert (paired["ci_lower"] == 0).all()
    assert (paired["ci_upper"] == 0).all()


def test_evaluate_models_includes_latency_columns(sample_transactions):
    frame = build_prediction_frame(
        [f"id-{i}" for i in range(4)],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        [1, 2, 3, 4],
    )

    results = evaluate_models({"model": frame})

    assert {"accuracy", "f1", "average_precision", "mean_latency_ms"}.issubset(results.columns)
