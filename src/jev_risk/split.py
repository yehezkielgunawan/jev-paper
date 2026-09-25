"""Reproducible train/test splitting and persisted sample manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from jev_risk.config import ID_COLUMN, TARGET_COLUMN
from jev_risk.data import validate_dataset
from jev_risk.features import build_features


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_ids: list[str]
    test_ids: list[str]
    dataset_sha256: str
    random_state: int
    test_size: float


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a stable CSV serialization of a DataFrame."""
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash the exact source file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_sha256: str | None = None,
) -> DatasetSplit:
    """Create the single fixed stratified split used by all model comparisons."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    validated = validate_dataset(frame).reset_index(drop=True)
    labels = validated[TARGET_COLUMN]
    indices = np.arange(len(validated))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    features = build_features(validated)
    return DatasetSplit(
        X_train=features.iloc[train_indices].reset_index(drop=True),
        X_test=features.iloc[test_indices].reset_index(drop=True),
        y_train=labels.iloc[train_indices].reset_index(drop=True),
        y_test=labels.iloc[test_indices].reset_index(drop=True),
        train_ids=validated.iloc[train_indices][ID_COLUMN].astype(str).tolist(),
        test_ids=validated.iloc[test_indices][ID_COLUMN].astype(str).tolist(),
        dataset_sha256=dataset_sha256 or dataframe_sha256(validated),
        random_state=random_state,
        test_size=test_size,
    )


def save_split_manifest(
    frame: pd.DataFrame,
    split: DatasetSplit,
    path: str | Path,
) -> Path:
    """Persist every sample ID, split assignment, and the dataset fingerprint."""
    validated = validate_dataset(frame)
    assignment = {sample_id: "train" for sample_id in split.train_ids}
    assignment.update({sample_id: "test" for sample_id in split.test_ids})
    manifest = pd.DataFrame(
        {
            "sample_id": validated[ID_COLUMN].astype(str),
            "y_true": validated[TARGET_COLUMN].astype(int),
        }
    )
    manifest["split"] = manifest["sample_id"].map(assignment)
    if manifest["split"].isna().any():
        raise ValueError("Split does not cover every row in the dataset")
    manifest["dataset_sha256"] = split.dataset_sha256
    manifest["random_state"] = split.random_state
    manifest["test_size"] = split.test_size

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    return destination
