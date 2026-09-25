from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from jev_risk.config import MODEL_FEATURES
from jev_risk.jev import JevClassifier, infer_with_cache, transaction_to_state


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.last_call = None

    def system_one(self, *, state, questions):
        self.calls += 1
        self.last_call = {"state": state, "questions": questions}
        answer = SimpleNamespace(
            choice="risk_incident",
            probabilities={"no_risk_incident": 0.15, "risk_incident": 0.85},
            confidence=0.83,
        )
        return SimpleNamespace(
            model="jev-latest",
            answers={"risk_incident": answer},
            usage=SimpleNamespace(input_tokens=12, output_tokens=5),
        )


def test_transaction_state_matches_features_and_excludes_leakage(sample_transactions):
    state = transaction_to_state(sample_transactions.iloc[[0]].iloc[0])

    assert set(state) == set(MODEL_FEATURES)
    assert "Risk_Incident" not in state
    assert "Risk_Type" not in state
    assert "Incident_Severity" not in state
    assert "Error_Code" not in state


def test_jev_choice_returns_risk_probability_confidence_and_latency(sample_transactions):
    client = FakeClient()
    classifier = JevClassifier(client=client)

    result = classifier.predict_one(sample_transactions.iloc[0].to_dict())

    assert result["sample_id"] == "TXN00000"
    assert result["y_pred"] == 1
    assert result["y_prob"] == 0.85
    assert result["confidence"] == 0.83
    assert result["model"] == "jev-latest"
    assert result["latency_ms"] >= 0
    assert set(client.last_call["questions"]["risk_incident"].criteria) == {
        "no_risk_incident",
        "risk_incident",
    }


def test_jev_choice_rejects_invalid_probability_for_either_option(sample_transactions):
    client = FakeClient()
    client.system_one = lambda **_: SimpleNamespace(
        model="jev-latest",
        answers={
            "risk_incident": SimpleNamespace(
                choice="risk_incident",
                probabilities={"no_risk_incident": 1.2, "risk_incident": 0.85},
                confidence=0.83,
            )
        },
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    classifier = JevClassifier(client=client)

    try:
        classifier.predict_one(sample_transactions.iloc[0].to_dict())
    except ValueError as error:
        assert "probabilities" in str(error)
    else:
        raise AssertionError("Expected invalid Choice probabilities to be rejected")


def test_inference_cache_skips_already_completed_sample_ids(tmp_path, sample_transactions):
    client = FakeClient()
    classifier = JevClassifier(client=client)
    cache_path = tmp_path / "cache" / "jev.jsonl"
    predictions_path = tmp_path / "predictions" / "jev.csv"
    selected = sample_transactions.iloc[:2]

    first = infer_with_cache(selected, classifier, cache_path, predictions_path)
    second = infer_with_cache(selected, classifier, cache_path, predictions_path)

    assert client.calls == 2
    assert len(first) == len(second) == 2
    assert list(pd.read_csv(predictions_path)["sample_id"]) == selected["Transaction_ID"].tolist()


def test_inference_cache_refreshes_truth_label_without_repeating_same_input_call(
    tmp_path, sample_transactions
):
    client = FakeClient()
    classifier = JevClassifier(client=client)
    cache_path = tmp_path / "jev.jsonl"
    predictions_path = tmp_path / "jev.csv"
    selected = sample_transactions.iloc[[0, 1, 5]].copy()
    infer_with_cache(selected, classifier, cache_path, predictions_path)
    selected.loc[selected.index[0], "Risk_Incident"] = 0

    updated = infer_with_cache(selected, classifier, cache_path, predictions_path)

    assert client.calls == 3
    assert updated.loc[0, "y_true"] == 0


def test_failed_rerun_does_not_leave_previous_complete_jev_predictions(
    tmp_path, sample_transactions
):
    client = FakeClient()
    classifier = JevClassifier(client=client)
    cache_path = tmp_path / "jev.jsonl"
    predictions_path = tmp_path / "jev.csv"
    selected = sample_transactions.iloc[:2].copy()
    infer_with_cache(selected, classifier, cache_path, predictions_path)
    selected.loc[selected.index[0], "Amount"] += 1

    def failed_request(**kwargs):
        raise RuntimeError("service unavailable")

    client.system_one = failed_request
    with pytest.raises(RuntimeError, match="service unavailable"):
        infer_with_cache(selected, classifier, cache_path, predictions_path)

    assert not predictions_path.exists()


def test_full_inference_rejects_a_model_version_different_from_the_pilot(
    tmp_path, sample_transactions
):
    client = FakeClient()
    classifier = JevClassifier(client=client)
    with pytest.raises(ValueError, match="pilot model"):
        infer_with_cache(
            sample_transactions.iloc[:2],
            classifier,
            tmp_path / "jev.jsonl",
            tmp_path / "jev.csv",
            expected_model="jev-2.0.0",
        )

    assert not (tmp_path / "jev.csv").exists()


def test_full_inference_exports_dataset_fingerprint(tmp_path, sample_transactions):
    output = tmp_path / "jev.csv"
    infer_with_cache(
        sample_transactions.iloc[:2],
        JevClassifier(client=FakeClient()),
        tmp_path / "jev.jsonl",
        output,
        dataset_sha256="frozen-dataset",
    )

    assert pd.read_csv(output)["dataset_sha256"].tolist() == ["frozen-dataset"] * 2


def test_completed_cache_regenerates_legacy_export_without_api_calls(tmp_path, sample_transactions):
    frame = sample_transactions.iloc[:2]
    cache_path = tmp_path / "jev.jsonl"
    export_path = tmp_path / "jev.csv"
    infer_with_cache(frame, JevClassifier(client=FakeClient()), cache_path, export_path)
    assert "dataset_sha256" not in pd.read_csv(export_path).columns

    class NoCallsClient:
        def system_one(self, **kwargs):
            raise AssertionError("A complete matching cache must not call TypeSafe")

    regenerated = infer_with_cache(
        frame,
        JevClassifier(client=NoCallsClient()),
        cache_path,
        export_path,
        expected_model="jev-latest",
        dataset_sha256="source-hash",
    )

    assert regenerated["dataset_sha256"].tolist() == ["source-hash", "source-hash"]
    assert pd.read_csv(export_path)["dataset_sha256"].tolist() == [
        "source-hash",
        "source-hash",
    ]


def test_inference_cache_drops_a_truncated_final_line_before_resuming(
    tmp_path, sample_transactions
):
    client = FakeClient()
    classifier = JevClassifier(client=client)
    cache_path = tmp_path / "jev.jsonl"
    predictions_path = tmp_path / "jev.csv"
    cache_path.write_text('{"sample_id":"interrupted"', encoding="utf-8")
    selected = sample_transactions.iloc[:2]

    infer_with_cache(selected, classifier, cache_path, predictions_path)
    infer_with_cache(selected, classifier, cache_path, predictions_path)

    assert client.calls == 2
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2


def test_resumed_inference_reports_cache_and_request_progress(tmp_path, sample_transactions):
    client = FakeClient()
    classifier = JevClassifier(client=client)
    cache_path = tmp_path / "jev.jsonl"
    predictions_path = tmp_path / "jev.csv"
    infer_with_cache(sample_transactions.iloc[:2], classifier, cache_path, predictions_path)
    events = []

    infer_with_cache(
        sample_transactions.iloc[:3],
        classifier,
        cache_path,
        predictions_path,
        on_progress=lambda event, completed, total, requests: events.append(
            (event, completed, total, requests)
        ),
    )

    assert client.calls == 3
    assert events[0] == ("start", 0, 3, 2)
    assert ("request", 2, 3, 1) in events
    assert events[-1] == ("complete", 3, 3, 1)
