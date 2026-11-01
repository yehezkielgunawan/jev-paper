from __future__ import annotations

import numpy as np
import pytest

from jev_risk.predictions import build_prediction_frame


def example():
    return build_prediction_frame(
        ["a", "b", "c", "d"], [0, 1, 0, 1], [0, 1, 1, 0],
        [0.1, 0.8, 0.7, 0.3], [1] * 4, confidence=[0.9, 0.9, 0.4, 0.2],
    )


def test_selective_diagnostics_report_composition_and_paired_uncertainty():
    from jev_risk.selective import selective_diagnostics

    frame = example()
    original = frame.copy(deep=True)
    output = selective_diagnostics(frame, [0.8], iterations=100, random_state=7)
    row = output["confidence_diagnostics"].iloc[0]
    assert row["accepted_count"] == 2
    assert row["coverage"] == 0.5
    assert row["positive_prevalence"] == 0.5
    assert row["predicted_positive_rate"] == 0.5
    assert row["true_positive"] == row["true_negative"] == 1
    assert row["false_positive"] == row["false_negative"] == 0
    assert row["selective_risk"] == 0
    assert row["accuracy_ci_lower"] == row["accuracy_ci_upper"] == 1
    assert row["accuracy_delta"] == 0.5
    assert row["accuracy_delta_ci_upper"] > row["accuracy_delta_ci_lower"]
    assert row["random_accuracy_expected"] == 0.5
    assert row["random_accuracy_reference_lower"] <= 0.5
    assert row["random_accuracy_reference_upper"] >= 0.5
    assert 0 < row["accuracy_valid_resamples"] <= 100
    assert frame.equals(original)
    again = selective_diagnostics(frame, [0.8], iterations=100, random_state=7)
    assert all(output[name].equals(again[name]) for name in output)


def test_risk_coverage_keeps_ties_together_and_auc_ranks_correctness():
    from jev_risk.selective import selective_diagnostics

    result = selective_diagnostics(example(), [0.8], iterations=20)
    curve = result["risk_coverage"]
    assert curve["accepted_count"].tolist() == [2, 3, 4]
    assert curve["selective_risk"].tolist() == pytest.approx([0, 1 / 3, 0.5])
    assert curve["random_risk_expected"].tolist() == [0.5] * 3
    assert result["confidence_correctness"].iloc[0]["estimate"] == 1
    assert result["confidence_bins"]["count"].sum() == 4


def test_confidence_summary_audits_binary_probability_relationship():
    from jev_risk.selective import selective_diagnostics

    summary = selective_diagnostics(example(), [0.8], iterations=10)["confidence_correctness"].iloc[0]
    assert summary["mean_confidence_correct"] == pytest.approx(0.9)
    assert summary["mean_confidence_incorrect"] == pytest.approx(0.3)
    assert summary["binary_confidence_formula_mae"] == pytest.approx(0.15)
    assert summary["binary_confidence_formula_match_fraction"] == 0.25
    assert summary["choice_probability_agreement"] == 1


def test_empty_acceptance_and_single_correctness_class_are_explicit():
    from jev_risk.selective import selective_diagnostics

    frame = build_prediction_frame(
        ["a", "b"], [0, 0], [0, 0], [0.1, 0.2], [1, 1], confidence=[0.5, 0.5],
    )
    output = selective_diagnostics(frame, [0.9], iterations=10)
    row = output["confidence_diagnostics"].iloc[0]
    assert row["accepted_count"] == 0
    assert row["accuracy_valid_resamples"] == 0
    assert np.isnan(row["accuracy"])
    assert np.isnan(row["accuracy_ci_lower"])
    assert np.isnan(output["confidence_correctness"].iloc[0]["estimate"])


def test_constant_confidence_has_chance_correctness_auc():
    from jev_risk.selective import selective_diagnostics

    frame = example()
    frame["confidence"] = 0.5
    output = selective_diagnostics(frame, [0.5], iterations=10)
    assert output["confidence_correctness"].iloc[0]["estimate"] == 0.5
    assert output["risk_coverage"]["accepted_count"].tolist() == [4]
    assert output["confidence_diagnostics"].iloc[0]["accuracy_delta"] == 0


@pytest.mark.parametrize("confidence", [float("nan"), -0.1, 1.1])
def test_invalid_confidence_is_rejected(confidence):
    from jev_risk.selective import selective_diagnostics

    frame = example()
    frame.loc[0, "confidence"] = confidence
    with pytest.raises(ValueError, match="confidence"):
        selective_diagnostics(frame, [0.5], iterations=10)


@pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.1])
def test_invalid_selective_threshold_is_rejected(threshold):
    from jev_risk.selective import selective_diagnostics

    with pytest.raises(ValueError, match="threshold"):
        selective_diagnostics(example(), [threshold], iterations=10)


def test_selective_bootstrap_requires_positive_iterations():
    from jev_risk.selective import selective_diagnostics

    with pytest.raises(ValueError, match="iterations"):
        selective_diagnostics(example(), [0.5], iterations=0)


def test_paired_accuracy_bootstrap_matches_independent_row_resampling():
    from jev_risk.selective import selective_diagnostics

    frame = example()
    result = selective_diagnostics(frame, [0.4], iterations=100, random_state=11)
    row = result["confidence_diagnostics"].iloc[0]
    rng = np.random.default_rng(11)
    expected = []
    for _ in range(100):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        accepted = sample.loc[sample["confidence"] >= 0.4]
        if len(accepted):
            selected_accuracy = (accepted["y_true"] == accepted["y_pred"]).mean()
            full_accuracy = (sample["y_true"] == sample["y_pred"]).mean()
            expected.append(selected_accuracy - full_accuracy)
    assert row["accuracy_delta_ci_lower"] == pytest.approx(np.percentile(expected, 2.5))
    assert row["accuracy_delta_ci_upper"] == pytest.approx(np.percentile(expected, 97.5))
    assert row["accuracy_delta_valid_resamples"] == len(expected)


def test_confidence_bins_include_exact_boundaries_and_one():
    from jev_risk.selective import selective_diagnostics

    frame = build_prediction_frame(
        list("abcdef"), [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1],
        [0.1, 0.9, 0.1, 0.9, 0.1, 0.9], [1] * 6,
        confidence=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    result = selective_diagnostics(frame, [0.0, 1.0], iterations=10)
    assert result["confidence_bins"]["count"].tolist() == [1, 1, 1, 1, 2]
    assert result["confidence_diagnostics"]["accepted_count"].tolist() == [6, 1]
