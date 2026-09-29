"""Paper-ready diagnostic figures for the held-out comparison."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from jev_risk.config import CONFIDENCE_THRESHOLDS
from jev_risk.evaluation import confidence_threshold_metrics
from jev_risk.predictions import validate_prediction_comparison


def create_evaluation_figures(
    predictions: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[str]:
    """Save confusion matrices, ROC, PR, calibration, and Jev coverage plots."""
    if not predictions:
        raise ValueError("At least one prediction frame is required")
    if len(predictions) > 1:
        validate_prediction_comparison(predictions)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for model, frame in predictions.items():
        figure, axis = plt.subplots(figsize=(5.5, 4.5))
        matrix = confusion_matrix(frame["y_true"], frame["y_pred"], labels=[0, 1])
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No incident", "Risk incident"],
            yticklabels=["No incident", "Risk incident"],
            xlabel="Predicted label",
            ylabel="True label",
            title=f"{model.replace('_', ' ').title()} confusion matrix",
        )
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.tight_layout()
        filename = f"confusion_matrix_{model}.png"
        figure.savefig(destination / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written.append(filename)

    roc_figure, roc_axis = plt.subplots(figsize=(7, 5.5))
    pr_figure, pr_axis = plt.subplots(figsize=(7, 5.5))
    calibration_figure, calibration_axis = plt.subplots(figsize=(7, 5.5))
    has_calibration = False
    for model, frame in predictions.items():
        y_true = frame["y_true"].to_numpy(dtype=int)
        probability = frame["y_prob"].to_numpy(dtype=float)
        if len(np.unique(y_true)) == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
            roc_axis.plot(false_positive_rate, true_positive_rate, label=model.replace("_", " "))
            if model != "dummy":
                precision, recall, _ = precision_recall_curve(y_true, probability)
                pr_axis.plot(recall, precision, label=model.replace("_", " "))
            bins = min(10, max(2, len(frame) // 2))
            fraction_positive, mean_predicted = calibration_curve(
                y_true, probability, n_bins=bins, strategy="quantile"
            )
            calibration_axis.plot(
                mean_predicted,
                fraction_positive,
                marker="o",
                label=model.replace("_", " "),
            )
            has_calibration = True

    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    roc_axis.legend(loc="lower right")
    roc_figure.tight_layout()
    roc_figure.savefig(destination / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(roc_figure)
    written.append("roc_curve.png")

    prevalence = float(next(iter(predictions.values()))["y_true"].mean())
    pr_axis.axhline(prevalence, linestyle="--", color="gray", label="Positive prevalence")
    pr_axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-recall curve (average precision)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    pr_axis.legend(loc="best")
    pr_figure.tight_layout()
    pr_figure.savefig(destination / "precision_recall_curve.png", dpi=160, bbox_inches="tight")
    plt.close(pr_figure)
    written.append("precision_recall_curve.png")

    calibration_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    calibration_axis.set(
        xlabel="Mean predicted risk probability",
        ylabel="Observed positive fraction",
        title="Calibration curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if has_calibration:
        calibration_axis.legend(loc="best")
    calibration_figure.tight_layout()
    calibration_figure.savefig(destination / "calibration_curve.png", dpi=160, bbox_inches="tight")
    plt.close(calibration_figure)
    written.append("calibration_curve.png")

    jev = predictions.get("jev")
    if jev is not None and "confidence" in jev:
        threshold_metrics = confidence_threshold_metrics(jev, CONFIDENCE_THRESHOLDS)
        figure, axis = plt.subplots(figsize=(7, 5.5))
        axis.plot(threshold_metrics["coverage"], threshold_metrics["accuracy"], marker="o")
        for _, row in threshold_metrics.iterrows():
            if np.isfinite(row["accuracy"]):
                axis.annotate(f"{row['threshold']:.2f}", (row["coverage"], row["accuracy"]))
        axis.set(
            xlabel="Automatic decision coverage",
            ylabel="Accuracy among accepted predictions",
            title="Jev confidence versus coverage",
            xlim=(0, 1.02),
            ylim=(0, 1.02),
        )
        figure.tight_layout()
        figure.savefig(destination / "confidence_coverage.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        written.append("confidence_coverage.png")
    return written
