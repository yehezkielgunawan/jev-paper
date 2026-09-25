from __future__ import annotations

import pandas as pd
import pytest

from jev_risk.predictions import (
    build_prediction_frame,
    save_predictions,
    validate_prediction_comparison,
)


def test_prediction_frame_has_standard_schema_and_optional_confidence():
    predictions = build_prediction_frame(
        sample_ids=["a", "b"],
        y_true=[0, 1],
        y_pred=[0, 1],
        y_prob=[0.1, 0.9],
        latency_ms=[1.0, 2.0],
        confidence=[0.8, 0.9],
    )

    assert list(predictions.columns) == [
        "sample_id",
        "y_true",
        "y_pred",
        "y_prob",
        "latency_ms",
        "confidence",
    ]
    assert predictions["y_prob"].tolist() == [0.1, 0.9]


def test_prediction_frame_rejects_probability_outside_unit_interval():
    with pytest.raises(ValueError, match="y_prob"):
        build_prediction_frame(["a"], [0], [1], [1.2], [1.0])


def test_prediction_frame_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number"):
        build_prediction_frame(["a", "b"], [0], [0], [0.1], [1.0])


def test_comparison_requires_same_test_ids_and_labels():
    rf = build_prediction_frame(["a", "b"], [0, 1], [0, 1], [0.1, 0.9], [1, 1])
    xgb = build_prediction_frame(["b", "c"], [1, 0], [1, 0], [0.8, 0.2], [1, 1])

    with pytest.raises(ValueError, match="same sample IDs"):
        validate_prediction_comparison({"random_forest": rf, "xgboost": xgb})


def test_comparison_requires_matching_ground_truth():
    rf = build_prediction_frame(["a", "b"], [0, 1], [0, 1], [0.1, 0.9], [1, 1])
    xgb = build_prediction_frame(["a", "b"], [1, 0], [1, 0], [0.8, 0.2], [1, 1])

    with pytest.raises(ValueError, match="ground-truth"):
        validate_prediction_comparison({"random_forest": rf, "xgboost": xgb})


def test_save_predictions_creates_parent_directory(tmp_path):
    predictions = build_prediction_frame(["a"], [0], [0], [0.1], [1.0])
    path = save_predictions(predictions, tmp_path / "predictions" / "model.csv")

    assert pd.read_csv(path).loc[0, "sample_id"] == "a"
