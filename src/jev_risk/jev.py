"""TypeSafe Jev Choice inference with a resumable JSONL response cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

import pandas as pd

from jev_risk.config import ID_COLUMN, TARGET_COLUMN
from jev_risk.data import validate_dataset
from jev_risk.features import build_features
from jev_risk.predictions import build_prediction_frame, save_predictions

QUESTION_NAME = "risk_incident"
PROMPT_VERSION = "binary-risk-choice-v1"
JEV_MODEL = "jev-latest"

QUESTION_INSTRUCTIONS = (
    "Classify the transaction in the supplied state. Choose risk_incident when the available "
    "transaction information indicates that this transaction should be flagged as a risk "
    "incident; otherwise choose no_risk_incident. Use only the supplied state."
)
QUESTION_CRITERIA = {
    "risk_incident": "The transaction should be flagged as a risk incident.",
    "no_risk_incident": "The transaction should not be flagged as a risk incident.",
}


def _plain_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def transaction_to_state(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Represent a raw transaction using exactly the shared primary feature set."""
    values = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    features = build_features(pd.DataFrame([values]))
    return {key: _plain_value(value) for key, value in features.iloc[0].items()}


def _read_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _get_answer(response: Any) -> Any:
    answers = _read_field(response, "answers")
    if answers is None:
        answers = _read_field(response, "choices")
    answer = _read_field(answers, QUESTION_NAME)
    if answer is None:
        raise ValueError(f"TypeSafe response is missing the {QUESTION_NAME!r} answer")
    return answer


def _state_fingerprint(state: Mapping[str, Any]) -> str:
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class JevClassifier:
    """Wrap one TypeSafe binary Choice question and normalize its response."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = JEV_MODEL,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._owns_client = client is None
        if client is None:
            resolved_key = api_key or os.getenv("TYPESAFE_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "Set TYPESAFE_API_KEY in your environment or .env file before Jev inference"
                )
            from typesafe_sdk import TypeSafeClient

            client = TypeSafeClient(api_key=resolved_key, model=model, timeout=timeout)
        self.client = client

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client and hasattr(self.client, "close"):
            self.client.close()

    def predict_one(self, row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        values = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        state = transaction_to_state(values)
        from typesafe_sdk import Choice

        question = Choice(instructions=QUESTION_INSTRUCTIONS, criteria=QUESTION_CRITERIA)
        start = time.perf_counter()
        response = self.client.system_one(
            state=state,
            questions={QUESTION_NAME: question},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        answer = _get_answer(response)
        choice = _read_field(answer, "choice")
        probabilities = _read_field(answer, "probabilities")
        confidence = _read_field(answer, "confidence")
        if choice not in QUESTION_CRITERIA:
            raise ValueError(f"Unexpected Jev Choice label: {choice!r}")
        if not isinstance(probabilities, Mapping):
            raise TypeError("Jev Choice answer must include a probabilities mapping")
        if set(QUESTION_CRITERIA).difference(probabilities):
            raise ValueError("Jev Choice probabilities must include both binary choices")
        risk_probability = float(probabilities["risk_incident"])
        no_risk_probability = float(probabilities["no_risk_incident"])
        confidence = float(confidence)
        if any(
            not math.isfinite(probability) or not 0 <= probability <= 1
            for probability in (risk_probability, no_risk_probability)
        ):
            raise ValueError("Jev Choice probabilities must be finite and between 0 and 1")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Jev confidence must be between 0 and 1")

        usage = _read_field(response, "usage")
        return {
            "sample_id": str(values[ID_COLUMN]),
            "y_pred": int(choice == "risk_incident"),
            "y_prob": risk_probability,
            "probability_no_risk": no_risk_probability,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model": str(_read_field(response, "model", self.model)),
            "usage_input_tokens": _read_field(usage, "input_tokens"),
            "usage_output_tokens": _read_field(usage, "output_tokens"),
        }


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
            valid_bytes = sum(len(previous) for previous in lines[:index])
            with path.open("r+b") as cache_handle:
                cache_handle.truncate(valid_bytes)
            break
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Jev cache record on line {index + 1}: {path}") from error
        if not isinstance(record, dict) or "sample_id" not in record:
            raise ValueError(f"Jev cache record on line {index + 1} has no sample_id")
        records[str(record["sample_id"])] = record
    return records


def _append_cache_record(path: Path, record: Mapping[str, Any]) -> None:
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as cache_handle:
        cache_handle.write(encoded + "\n")
        cache_handle.flush()


def infer_with_cache(
    frame: pd.DataFrame,
    classifier: JevClassifier,
    cache_path: str | Path,
    predictions_path: str | Path,
    on_progress: Callable[[str, int, int, int], None] | None = None,
    expected_model: str | None = None,
    dataset_sha256: str | None = None,
) -> pd.DataFrame:
    """Infer missing/changed rows, append each response, and rewrite predictions CSV."""
    validated = validate_dataset(frame)
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_file)
    model_alias = getattr(classifier, "model", JEV_MODEL)

    # A previous completed export must not be evaluated if this run fails midway.
    output_file = Path(predictions_path)
    output_file.unlink(missing_ok=True)

    ordered_records: list[dict[str, Any]] = []
    total = len(validated)
    requests = 0
    if on_progress:
        on_progress("start", 0, total, len(cache))
    for completed, (_, row) in enumerate(validated.iterrows()):
        sample_id = str(row[ID_COLUMN])
        state = transaction_to_state(row)
        fingerprint = _state_fingerprint(state)
        cached = cache.get(sample_id)
        if (
            cached is None
            or cached.get("state_sha256") != fingerprint
            or cached.get("prompt_version") != PROMPT_VERSION
            or cached.get("model_alias") != model_alias
        ):
            if on_progress:
                on_progress("request", completed, total, requests + 1)
            prediction = classifier.predict_one(row)
            if expected_model is not None and prediction["model"] != expected_model:
                raise ValueError(
                    f"Jev returned {prediction['model']!r}, not the pilot model "
                    f"{expected_model!r}; rerun the pilot with a pinned model version"
                )
            requests += 1
            record = {
                **prediction,
                "sample_id": sample_id,
                "y_true": int(row[TARGET_COLUMN]),
                "state_sha256": fingerprint,
                "prompt_version": PROMPT_VERSION,
                "model_alias": model_alias,
            }
            _append_cache_record(cache_file, record)
            cache[sample_id] = record
            cached = record
        elif cached.get("y_true") != int(row[TARGET_COLUMN]):
            cached = {**cached, "y_true": int(row[TARGET_COLUMN])}
            _append_cache_record(cache_file, cached)
            cache[sample_id] = cached
        if expected_model is not None and cached.get("model") != expected_model:
            raise ValueError(
                f"Cached Jev model {cached.get('model')!r} differs from the pilot model "
                f"{expected_model!r}; use a matching pinned model version"
            )
        ordered_records.append(cached)
        if on_progress and (completed + 1) % 25 == 0:
            on_progress("checkpoint", completed + 1, total, requests)

    if on_progress:
        on_progress("complete", total, total, requests)

    predictions = build_prediction_frame(
        sample_ids=[record["sample_id"] for record in ordered_records],
        y_true=[record["y_true"] for record in ordered_records],
        y_pred=[record["y_pred"] for record in ordered_records],
        y_prob=[record["y_prob"] for record in ordered_records],
        latency_ms=[record["latency_ms"] for record in ordered_records],
        confidence=[record["confidence"] for record in ordered_records],
    )
    predictions["model"] = [record["model"] for record in ordered_records]
    predictions["usage_input_tokens"] = [
        record.get("usage_input_tokens") for record in ordered_records
    ]
    predictions["usage_output_tokens"] = [
        record.get("usage_output_tokens") for record in ordered_records
    ]
    if dataset_sha256 is not None:
        predictions["dataset_sha256"] = dataset_sha256
    save_predictions(predictions, output_file)
    return predictions
