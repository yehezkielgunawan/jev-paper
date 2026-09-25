from __future__ import annotations

import pandas as pd
import pytest

from jev_risk.split import create_split, save_split_manifest


def test_split_is_stratified_deterministic_and_keeps_ids(sample_transactions):
    first = create_split(sample_transactions, test_size=0.2, random_state=17)
    second = create_split(sample_transactions, test_size=0.2, random_state=17)

    assert len(first.test_ids) == 8
    assert first.test_ids == second.test_ids
    assert first.train_ids == second.train_ids
    assert sum(first.y_test) == 2
    assert set(first.test_ids).isdisjoint(first.train_ids)
    assert len(first.X_test) == len(first.y_test) == 8


def test_split_manifest_records_split_and_dataset_hash(tmp_path, sample_transactions):
    split = create_split(sample_transactions, test_size=0.2, random_state=42)
    path = tmp_path / "split_manifest.csv"

    save_split_manifest(sample_transactions, split, path)
    manifest = pd.read_csv(path)

    assert set(manifest["split"]) == {"train", "test"}
    assert manifest["sample_id"].nunique() == len(sample_transactions)
    assert manifest["dataset_sha256"].nunique() == 1
    assert set(manifest.loc[manifest["split"] == "test", "sample_id"]) == set(split.test_ids)


def test_split_rejects_invalid_test_fraction(sample_transactions):
    with pytest.raises(ValueError, match="test_size"):
        create_split(sample_transactions, test_size=1.0, random_state=42)
