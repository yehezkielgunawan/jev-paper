"""Offline diagnostics for Jev's separate confidence field and fixed hard choices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jev_risk.evaluation import _percentile_interval, calculate_metrics


def _decision_stats(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    count = len(truth)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    return {
        "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
        "positive_prevalence": float(truth.mean()) if count else float("nan"),
        "predicted_positive_rate": float(predicted.mean()) if count else float("nan"),
        "accuracy": (tp + tn) / count if count else float("nan"),
        "selective_risk": (fp + fn) / count if count else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else (0.0 if count else float("nan")),
        "recall": tp / (tp + fn) if tp + fn else (0.0 if count else float("nan")),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (
            0.0 if count else float("nan")
        ),
    }


def _correctness_auc(correct: np.ndarray, confidence: np.ndarray) -> float:
    return float(roc_auc_score(correct, confidence)) if len(np.unique(correct)) == 2 else float("nan")


def selective_diagnostics(
    predictions: pd.DataFrame,
    thresholds: Sequence[float],
    iterations: int = 1_000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Return composition, paired bootstrap CIs, random-selection references and curves.

    Bootstrap shared rows, recomputing fixed-cutoff acceptance in each draw. Random
    acceptance is uniform without replacement on the observed test set; its central
    95% range is a conditional randomization reference, NOT a population CI.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if "confidence" not in predictions:
        raise ValueError("A confidence column is required")
    calculate_metrics(predictions["y_true"], predictions["y_pred"], predictions["y_prob"])
    confidence = predictions["confidence"].to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or not ((confidence >= 0) & (confidence <= 1)).all():
        raise ValueError("confidence must be finite and between 0 and 1")
    thresholds = tuple(thresholds)
    if any(not np.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError("Each confidence threshold must be finite and between 0 and 1")
    truth = predictions["y_true"].to_numpy(dtype=int)
    predicted = predictions["y_pred"].to_numpy(dtype=int)
    correct = truth == predicted
    count = len(truth)
    baseline_accuracy = float(correct.mean())
    interval_metrics = (
        "accuracy", "precision", "recall", "f1", "selective_risk",
        "positive_prevalence", "predicted_positive_rate", "coverage", "accuracy_delta",
    )
    bootstrap = {t: {metric: [] for metric in interval_metrics} for t in thresholds}
    auc_samples = []
    rng = np.random.default_rng(random_state)
    for _ in range(iterations):
        indices = rng.integers(0, count, size=count)
        sampled_truth, sampled_predicted = truth[indices], predicted[indices]
        sampled_confidence = confidence[indices]
        sampled_correct = correct[indices]
        auc_samples.append(_correctness_auc(sampled_correct, sampled_confidence))
        for threshold in thresholds:
            mask = sampled_confidence >= threshold
            stats = _decision_stats(sampled_truth[mask], sampled_predicted[mask])
            stats["coverage"] = float(mask.mean())
            stats["accuracy_delta"] = stats["accuracy"] - float(sampled_correct.mean())
            for metric in interval_metrics:
                bootstrap[threshold][metric].append(stats[metric])

    # A separate stream prevents the bootstrap draws from depending on curve resolution.
    reference_rng = np.random.default_rng(random_state + 1)

    def random_accuracy_range(accepted_count: int) -> tuple[float, float]:
        if not accepted_count:
            return float("nan"), float("nan")
        successes = reference_rng.hypergeometric(
            int(correct.sum()), int((~correct).sum()), accepted_count, size=iterations,
        )
        return _percentile_interval(successes / accepted_count)

    rows = []
    for threshold in thresholds:
        mask = confidence >= threshold
        accepted_count = int(mask.sum())
        stats = _decision_stats(truth[mask], predicted[mask])
        lower, upper = random_accuracy_range(accepted_count)
        row = {
            "threshold": threshold, "coverage": accepted_count / count,
            "accepted_count": accepted_count, "review_count": count - accepted_count,
            **stats, "accuracy_delta": stats["accuracy"] - baseline_accuracy,
            "random_accuracy_expected": baseline_accuracy if accepted_count else float("nan"),
            "random_accuracy_reference_lower": lower, "random_accuracy_reference_upper": upper,
        }
        for metric in interval_metrics:
            samples = bootstrap[threshold][metric]
            low, high = _percentile_interval(samples)
            row[f"{metric}_ci_lower"] = low
            row[f"{metric}_ci_upper"] = high
            row[f"{metric}_valid_resamples"] = int(np.isfinite(samples).sum())
        rows.append(row)

    # Threshold all distinct scores, retaining an entire tie group at once.
    curve = []
    for threshold in np.unique(confidence)[::-1]:
        mask = confidence >= threshold
        accepted_count = int(mask.sum())
        lower, upper = random_accuracy_range(accepted_count)
        curve.append({
            "threshold": float(threshold), "coverage": accepted_count / count,
            "accepted_count": accepted_count,
            "selective_risk": float((~correct[mask]).mean()),
            "random_risk_expected": 1 - baseline_accuracy,
            "random_risk_reference_lower": 1 - upper,
            "random_risk_reference_upper": 1 - lower,
        })

    bins = []
    bin_index = np.minimum((confidence * 5).astype(int), 4)
    for index in range(5):
        mask = bin_index == index
        bins.append({
            "bin_lower": index / 5, "bin_upper": (index + 1) / 5,
            "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else float("nan"),
            **_decision_stats(truth[mask], predicted[mask]),
        })
    low, high = _percentile_interval(auc_samples)
    probability = predictions["y_prob"].to_numpy(dtype=float)
    # Binary specialization of the vendor's documented (p_max - 1/n)/(1 - 1/n).
    formula_error = np.abs(confidence - (2 * np.maximum(probability, 1 - probability) - 1))
    auc = {
        "metric": "confidence_correctness_auc", "estimate": _correctness_auc(correct, confidence),
        "ci_lower": low, "ci_upper": high,
        "valid_resamples": int(np.isfinite(auc_samples).sum()),
        "bootstrap_iterations": iterations,
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else float("nan"),
        "mean_confidence_incorrect": (
            float(confidence[~correct].mean()) if (~correct).any() else float("nan")
        ),
        "binary_confidence_formula_mae": float(formula_error.mean()),
        "binary_confidence_formula_tolerance": 0.015,
        "binary_confidence_formula_match_fraction": float((formula_error <= 0.015).mean()),
        "choice_probability_agreement": float(((probability > 0.5).astype(int) == predicted).mean()),
    }
    return {
        "confidence_diagnostics": pd.DataFrame(rows),
        "confidence_correctness": pd.DataFrame([auc]),
        "risk_coverage": pd.DataFrame(curve),
        "confidence_bins": pd.DataFrame(bins),
    }
