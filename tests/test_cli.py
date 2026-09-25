from __future__ import annotations

import json

import pytest

from jev_risk.cli import (
    _validate_jev_pilot,
    build_parser,
    run_audit,
    run_evaluate,
    run_jev,
    select_pilot_rows,
)
from jev_risk.jev import JEV_MODEL, PROMPT_VERSION
from jev_risk.split import create_split


def test_pilot_selection_is_deterministic_and_contains_both_target_classes(sample_transactions):
    first = select_pilot_rows(sample_transactions, sample_size=6, random_state=11)
    second = select_pilot_rows(sample_transactions, sample_size=6, random_state=11)

    assert first["Transaction_ID"].tolist() == second["Transaction_ID"].tolist()
    assert len(first) == 6
    assert set(first["Risk_Incident"]) == {0, 1}


def test_pilot_selection_requires_room_for_both_classes(sample_transactions):
    with pytest.raises(ValueError, match="at least 2"):
        select_pilot_rows(sample_transactions, sample_size=1, random_state=11)


def test_cli_includes_a_distinct_jev_pilot_command():
    parser = build_parser()

    assert parser._subparsers._group_actions[0].choices.keys() >= {
        "audit",
        "split",
        "train",
        "jev-pilot",
        "jev",
        "evaluate",
    }


def test_run_audit_writes_the_validated_dataset_summary(tmp_path, sample_transactions):
    data_path = tmp_path / "transactions.csv"
    output_path = tmp_path / "reports" / "audit.json"
    sample_transactions.to_csv(data_path, index=False)

    summary = run_audit(data_path, output_path)

    assert output_path.is_file()
    assert summary["positive_count"] == 8
    assert summary["leakage_checks"]["Risk_Type"]["perfectly_encodes_target"]


def test_full_jev_inference_requires_matching_successful_pilot(tmp_path, sample_transactions):
    data_path = tmp_path / "transactions.csv"
    sample_transactions.to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="jev-pilot"):
        run_jev(data_path, tmp_path / "results")


def test_pilot_gate_returns_the_resolved_model_for_full_inference(tmp_path, sample_transactions):
    split = create_split(sample_transactions)
    manifest = tmp_path / "metadata" / "jev_pilot.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "dataset_sha256": split.dataset_sha256,
                "random_state": split.random_state,
                "test_size": split.test_size,
                "model_alias": JEV_MODEL,
                "prompt_version": PROMPT_VERSION,
                "returned_models": ["jev-1.13.0"],
            }
        ),
        encoding="utf-8",
    )

    assert _validate_jev_pilot(tmp_path, split, JEV_MODEL) == "jev-1.13.0"


def test_evaluate_reports_partial_jev_instead_of_silently_omitting_it(
    tmp_path, sample_transactions
):
    from jev_risk.predictions import build_prediction_frame, save_predictions

    results = tmp_path / "results"
    local = build_prediction_frame(
        sample_ids=sample_transactions["Transaction_ID"].tolist(),
        y_true=sample_transactions["Risk_Incident"].tolist(),
        y_pred=[0] * len(sample_transactions),
        y_prob=[0.2] * len(sample_transactions),
        latency_ms=[1] * len(sample_transactions),
    )
    for name in ("dummy", "random_forest"):
        save_predictions(local, results / "predictions" / f"{name}.csv")
    cache = results / "cache" / "jev.jsonl"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"sample_id":"TXN00000"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Jev inference is incomplete.*jev-risk jev"):
        run_evaluate(results, bootstrap_iterations=2)

    assert not (results / "metrics" / "model_comparison.csv").exists()

    local_outputs = run_evaluate(results, bootstrap_iterations=2, local_only=True)
    assert set(local_outputs["model_comparison"]["model"]) == {"dummy", "random_forest"}


def test_evaluate_rejects_jev_predictions_from_a_different_dataset(tmp_path, sample_transactions):
    from jev_risk.predictions import build_prediction_frame, save_predictions

    results = tmp_path / "results"
    frame = build_prediction_frame(
        sample_ids=sample_transactions["Transaction_ID"].tolist(),
        y_true=sample_transactions["Risk_Incident"].tolist(),
        y_pred=[0] * len(sample_transactions),
        y_prob=[0.2] * len(sample_transactions),
        latency_ms=[1] * len(sample_transactions),
    )
    for name in ("dummy", "random_forest", "jev"):
        prediction = frame.copy()
        if name == "jev":
            prediction["dataset_sha256"] = "previous-dataset"
        save_predictions(prediction, results / "predictions" / f"{name}.csv")
    metadata = results / "metadata" / "training_run.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"dataset_sha256": "current-dataset"}), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset fingerprint"):
        run_evaluate(results, bootstrap_iterations=2)

    corrected = frame.copy()
    corrected["dataset_sha256"] = "current-dataset"
    save_predictions(corrected, results / "predictions" / "jev.csv")
    assert set(run_evaluate(results, bootstrap_iterations=2)["model_comparison"]["model"]) == {
        "dummy",
        "random_forest",
        "jev",
    }
