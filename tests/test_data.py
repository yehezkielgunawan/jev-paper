from __future__ import annotations

import pandas as pd
import pytest

from jev_risk.data import audit_dataset, load_dataset, validate_dataset


def test_load_dataset_and_audit_summary(tmp_path, sample_transactions):
    path = tmp_path / "transactions.csv"
    sample_transactions.to_csv(path, index=False)

    loaded = load_dataset(path)
    summary = audit_dataset(loaded)

    assert summary["rows"] == 40
    assert summary["positive_count"] == 8
    assert summary["positive_rate"] == pytest.approx(0.2)
    assert summary["duplicate_transaction_ids"] == 0


def test_load_dataset_preserves_na_as_an_ip_region_and_none_as_literal_text(
    tmp_path, sample_transactions
):
    sample_transactions.loc[0, "IP_Region"] = "NA"
    path = tmp_path / "transactions.csv"
    sample_transactions.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert loaded.loc[0, "IP_Region"] == "NA"
    assert loaded.loc[1, "Risk_Type"] == "None"
    assert loaded["IP_Region"].isna().sum() == 0


def test_audit_marks_post_event_columns_as_target_leakage(sample_transactions):
    summary = audit_dataset(sample_transactions)

    assert all(
        summary["leakage_checks"][column]["perfectly_encodes_target"]
        for column in ("Risk_Type", "Incident_Severity", "Error_Code")
    )


def test_validate_dataset_rejects_duplicate_transaction_ids(sample_transactions):
    sample_transactions.loc[1, "Transaction_ID"] = sample_transactions.loc[0, "Transaction_ID"]

    with pytest.raises(ValueError, match="unique"):
        validate_dataset(sample_transactions)


def test_validate_dataset_rejects_non_binary_target(sample_transactions):
    sample_transactions.loc[0, "Risk_Incident"] = 2

    with pytest.raises(ValueError, match="0 and 1"):
        validate_dataset(sample_transactions)


def test_validate_dataset_rejects_missing_required_column(sample_transactions):
    with pytest.raises(ValueError, match="IP_Region"):
        validate_dataset(sample_transactions.drop(columns="IP_Region"))


def test_validate_dataset_rejects_null_target(sample_transactions):
    sample_transactions["Risk_Incident"] = pd.to_numeric(sample_transactions["Risk_Incident"])
    sample_transactions.loc[0, "Risk_Incident"] = None

    with pytest.raises(ValueError, match="missing"):
        validate_dataset(sample_transactions)
