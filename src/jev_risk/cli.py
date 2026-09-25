"""Command-line entry points for the reproducible research workflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from jev_risk.config import (
    BOOTSTRAP_ITERATIONS,
    CONFIDENCE_THRESHOLDS,
    DEFAULT_DATA_PATH,
    EXCLUDED_FEATURES,
    ID_COLUMN,
    MODEL_FEATURES,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from jev_risk.data import load_dataset, write_audit_summary
from jev_risk.evaluation import (
    bootstrap_metric_intervals,
    confidence_threshold_metrics,
    evaluate_models,
    latency_summary,
)
from jev_risk.figures import create_evaluation_figures
from jev_risk.jev import (
    JEV_MODEL,
    PROMPT_VERSION,
    JevClassifier,
    infer_with_cache,
)
from jev_risk.models import fit_classical_models, predict_model
from jev_risk.predictions import build_prediction_frame, save_predictions
from jev_risk.split import DatasetSplit, create_split, file_sha256, save_split_manifest


def select_pilot_rows(
    frame: pd.DataFrame,
    sample_size: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Choose a deterministic pilot sample containing both target classes."""
    if sample_size < 2:
        raise ValueError("The Jev pilot sample_size must be at least 2 to include both classes")
    if sample_size > len(frame):
        raise ValueError("Pilot sample_size cannot exceed the available held-out rows")
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Pilot rows require the {TARGET_COLUMN} column")
    positive_indices = frame.index[frame[TARGET_COLUMN].astype(int) == 1].to_numpy()
    negative_indices = frame.index[frame[TARGET_COLUMN].astype(int) == 0].to_numpy()
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        raise ValueError("The Jev pilot sample must include both target classes")

    rng = np.random.default_rng(random_state)
    selected = [int(rng.choice(negative_indices)), int(rng.choice(positive_indices))]
    remaining = np.asarray([index for index in frame.index if index not in selected])
    if sample_size > 2:
        selected.extend(
            rng.choice(remaining, size=sample_size - 2, replace=False).astype(int).tolist()
        )
    return frame.loc[selected].reset_index(drop=True)


def _split_for_dataset(data_path: str | Path) -> tuple[pd.DataFrame, DatasetSplit]:
    frame = load_dataset(data_path)
    split = create_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        dataset_sha256=file_sha256(data_path),
    )
    return frame, split


def run_audit(data_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Audit a CSV and write its JSON summary."""
    frame = load_dataset(data_path)
    return write_audit_summary(frame, output_path)


def run_split(
    data_path: str | Path,
    output_path: str | Path,
) -> DatasetSplit:
    """Create and persist the shared deterministic split manifest."""
    frame, split = _split_for_dataset(data_path)
    save_split_manifest(frame, split, output_path)
    return split


def run_train(
    data_path: str | Path,
    results_dir: str | Path = RESULTS_DIR,
    tune: bool = True,
) -> dict[str, dict[str, Any]]:
    """Train local models, export predictions, and save fitted model artifacts."""
    frame, split = _split_for_dataset(data_path)
    destination = Path(results_dir)
    save_split_manifest(frame, split, destination / "splits" / "split_manifest.csv")
    models, tuning = fit_classical_models(split, tune=tune)

    prediction_dir = destination / "predictions"
    model_dir = destination / "models"
    for name, model in models.items():
        save_predictions(predict_model(name, model, split), prediction_dir / f"{name}.csv")
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f"{name}.joblib")

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(Path(data_path).resolve()),
        "dataset_sha256": split.dataset_sha256,
        "target": TARGET_COLUMN,
        "positive_class": 1,
        "random_state": split.random_state,
        "test_size": split.test_size,
        "train_rows": len(split.train_ids),
        "test_rows": len(split.test_ids),
        "feature_columns": list(MODEL_FEATURES),
        "excluded_features": list(EXCLUDED_FEATURES),
        "tuning": tuning,
        "tuned": tune,
        "threshold": 0.5,
        "versions": {
            package: version(package)
            for package in ("jev-risk-classification", "pandas", "scikit-learn", "xgboost")
        },
    }
    metadata_path = destination / "metadata" / "training_run.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return tuning


def _test_frame(frame: pd.DataFrame, split: DatasetSplit) -> pd.DataFrame:
    indexed = frame.set_index(ID_COLUMN, drop=False)
    return indexed.loc[split.test_ids].reset_index(drop=True)


def _report_jev_progress(event: str, completed: int, total: int, count: int) -> None:
    if event == "start":
        message = f"Jev: {total} test rows; {count} cached responses available."
    elif event == "request":
        if count != 1:
            return
        message = f"Jev: requesting uncached row {completed + 1}/{total} from TypeSafe..."
    elif event == "checkpoint":
        message = f"Jev: {completed}/{total} rows processed; {count} new API calls."
    else:
        message = f"Jev: {completed}/{total} rows processed; {count} new API calls this run."
    print(message, file=sys.stderr, flush=True)


def run_jev_pilot(
    data_path: str | Path,
    results_dir: str | Path = RESULTS_DIR,
    sample_size: int = 5,
    model: str = JEV_MODEL,
) -> pd.DataFrame:
    """Validate TypeSafe Choice responses on a small stratified test sample."""
    frame, split = _split_for_dataset(data_path)
    destination = Path(results_dir)
    save_split_manifest(frame, split, destination / "splits" / "split_manifest.csv")
    test_frame = _test_frame(frame, split)
    pilot_frame = select_pilot_rows(test_frame, sample_size, RANDOM_STATE)
    load_dotenv()
    cache_path = destination / "cache" / "jev.jsonl"
    with JevClassifier(model=model) as classifier:
        predictions = infer_with_cache(
            pilot_frame,
            classifier,
            cache_path,
            destination / "predictions" / "jev_pilot.csv",
            on_progress=_report_jev_progress,
        )

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_sha256": split.dataset_sha256,
        "test_size": split.test_size,
        "random_state": split.random_state,
        "model_alias": model,
        "returned_models": sorted(predictions["model"].dropna().unique().tolist()),
        "prompt_version": PROMPT_VERSION,
        "sample_ids": predictions["sample_id"].tolist(),
        "validated_fields": ["choice", "probabilities", "confidence", "model", "latency_ms"],
    }
    manifest_path = destination / "metadata" / "jev_pilot.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return predictions


def _validate_jev_pilot(
    results_dir: str | Path,
    split: DatasetSplit,
    model: str,
) -> str:
    path = Path(results_dir) / "metadata" / "jev_pilot.json"
    if not path.is_file():
        raise ValueError("Run `uv run jev-risk jev-pilot` successfully before full Jev inference")
    pilot = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_sha256": split.dataset_sha256,
        "random_state": split.random_state,
        "test_size": split.test_size,
        "model_alias": model,
        "prompt_version": PROMPT_VERSION,
    }
    mismatches = [key for key, value in expected.items() if pilot.get(key) != value]
    if mismatches:
        raise ValueError(
            "Jev pilot manifest does not match this experiment ("
            + ", ".join(mismatches)
            + "); rerun `uv run jev-risk jev-pilot`"
        )
    returned_models = pilot.get("returned_models")
    if (
        not isinstance(returned_models, list)
        or len(returned_models) != 1
        or not isinstance(returned_models[0], str)
    ):
        raise ValueError(
            "Jev pilot must record exactly one resolved model version; rerun the pilot"
        )
    return returned_models[0]


def run_jev(
    data_path: str | Path,
    results_dir: str | Path = RESULTS_DIR,
    model: str = JEV_MODEL,
) -> pd.DataFrame:
    """Run or resume Jev inference across the complete frozen test set."""
    frame, split = _split_for_dataset(data_path)
    destination = Path(results_dir)
    expected_model = _validate_jev_pilot(destination, split, model)
    save_split_manifest(frame, split, destination / "splits" / "split_manifest.csv")
    load_dotenv()
    with JevClassifier(model=model) as classifier:
        return infer_with_cache(
            _test_frame(frame, split),
            classifier,
            destination / "cache" / "jev.jsonl",
            destination / "predictions" / "jev.csv",
            on_progress=_report_jev_progress,
            expected_model=expected_model,
            dataset_sha256=split.dataset_sha256,
        )


def _read_prediction_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    confidence = raw["confidence"].tolist() if "confidence" in raw else None
    frame = build_prediction_frame(
        sample_ids=raw["sample_id"].tolist(),
        y_true=raw["y_true"].tolist(),
        y_pred=raw["y_pred"].tolist(),
        y_prob=raw["y_prob"].tolist(),
        latency_ms=raw["latency_ms"].tolist(),
        confidence=confidence,
    )
    for column in ("model", "usage_input_tokens", "usage_output_tokens", "dataset_sha256"):
        if column in raw:
            frame[column] = raw[column]
    return frame


def run_evaluate(
    results_dir: str | Path = RESULTS_DIR,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    local_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Evaluate available prediction files, compare, bootstrap, and save figures."""
    destination = Path(results_dir)
    prediction_dir = destination / "predictions"
    jev_predictions = prediction_dir / "jev.csv"
    jev_cache = destination / "cache" / "jev.jsonl"
    if not local_only and not jev_predictions.is_file() and jev_cache.is_file():
        cached_count = sum(1 for line in jev_cache.open(encoding="utf-8") if line.strip())
        raise ValueError(
            f"Jev inference is incomplete ({cached_count} cached responses, no jev.csv). "
            "Resume with `uv run jev-risk jev`, or use `uv run jev-risk evaluate --local-only` "
            "to intentionally evaluate local models."
        )
    model_names = (
        ("dummy", "random_forest", "xgboost")
        if local_only
        else ("dummy", "random_forest", "xgboost", "jev")
    )
    predictions = {
        name: _read_prediction_file(prediction_dir / f"{name}.csv")
        for name in model_names
        if (prediction_dir / f"{name}.csv").is_file()
    }
    if len(predictions) < 2:
        raise ValueError("At least two prediction CSVs are required; run local training first")

    if "jev" in predictions:
        training_path = destination / "metadata" / "training_run.json"
        if not training_path.is_file():
            raise ValueError("Training metadata is required to verify the Jev dataset fingerprint")
        training_sha256 = json.loads(training_path.read_text(encoding="utf-8")).get(
            "dataset_sha256"
        )
        jev_hashes = predictions["jev"].get("dataset_sha256")
        if not training_sha256 or jev_hashes is None or set(jev_hashes) != {training_sha256}:
            raise ValueError(
                "Jev dataset fingerprint does not match local training; rerun Jev inference "
                "on this dataset before evaluating"
            )

    metrics = evaluate_models(predictions)
    latencies = latency_summary(predictions)
    intervals = bootstrap_metric_intervals(
        predictions,
        iterations=bootstrap_iterations,
        random_state=RANDOM_STATE,
    )
    metrics_dir = destination / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "model_comparison.csv", index=False)
    latencies.to_csv(metrics_dir / "latency_summary.csv", index=False)
    intervals.to_csv(metrics_dir / "bootstrap_intervals.csv", index=False)

    confidence_results = None
    if "jev" in predictions and "confidence" in predictions["jev"]:
        confidence_results = confidence_threshold_metrics(predictions["jev"], CONFIDENCE_THRESHOLDS)
        confidence_results.to_csv(metrics_dir / "confidence_thresholds.csv", index=False)
    create_evaluation_figures(predictions, destination / "figures")
    return {
        "model_comparison": metrics,
        "latency_summary": latencies,
        "bootstrap_intervals": intervals,
        **({"confidence_thresholds": confidence_results} if confidence_results is not None else {}),
    }


def _add_data_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input CSV path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-risk",
        description="Reproducible TypeSafe Jev transaction-risk classification experiment",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate and summarize the source dataset")
    _add_data_option(audit)
    audit.add_argument("--output", type=Path, default=RESULTS_DIR / "audit" / "dataset_audit.json")

    split = subparsers.add_parser("split", help="Create the frozen stratified split manifest")
    _add_data_option(split)
    split.add_argument("--output", type=Path, default=RESULTS_DIR / "splits" / "split_manifest.csv")

    train = subparsers.add_parser("train", help="Fit local baseline, Random Forest, and XGBoost")
    _add_data_option(train)
    train.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    train.add_argument(
        "--no-tune", action="store_true", help="Fit the baseline configurations without CV search"
    )

    pilot = subparsers.add_parser(
        "jev-pilot", help="Validate the Jev Choice response on a small sample"
    )
    _add_data_option(pilot)
    pilot.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    pilot.add_argument("--sample-size", type=int, default=5)
    pilot.add_argument(
        "--model", default=JEV_MODEL, help="TypeSafe model alias (default: jev-latest)"
    )

    jev = subparsers.add_parser("jev", help="Run/resume Jev inference over the frozen test set")
    _add_data_option(jev)
    jev.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    jev.add_argument(
        "--model", default=JEV_MODEL, help="TypeSafe model alias (default: jev-latest)"
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="Export metrics, bootstrap intervals, and figures"
    )
    evaluate.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    evaluate.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    evaluate.add_argument(
        "--local-only", action="store_true", help="Explicitly evaluate local models without Jev"
    )

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "audit":
        summary = run_audit(args.data, args.output)
        print(json.dumps(summary, indent=2))
    elif args.command == "split":
        split = run_split(args.data, args.output)
        print(
            f"Saved split manifest: {args.output} ({len(split.train_ids)} train, {len(split.test_ids)} test)"
        )
    elif args.command == "train":
        metadata = run_train(args.data, args.results_dir, tune=not args.no_tune)
        print(json.dumps(metadata, indent=2, default=str))
    elif args.command == "jev-pilot":
        predictions = run_jev_pilot(args.data, args.results_dir, args.sample_size, args.model)
        print(f"Jev pilot validated {len(predictions)} responses; full-inference gate recorded.")
    elif args.command == "jev":
        predictions = run_jev(args.data, args.results_dir, args.model)
        print(f"Jev predictions ready for {len(predictions)} held-out samples.")
    elif args.command == "evaluate":
        outputs = run_evaluate(
            args.results_dir, args.bootstrap_iterations, local_only=args.local_only
        )
        print(outputs["model_comparison"].to_string(index=False))
    else:
        raise ValueError(f"Unknown command: {args.command}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
