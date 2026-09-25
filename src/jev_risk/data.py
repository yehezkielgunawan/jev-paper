"""Dataset loading, validation, and audit reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jev_risk.config import ID_COLUMN, TARGET_COLUMN

REQUIRED_COLUMNS = (
    ID_COLUMN,
    "Date",
    "Account_Number",
    "Transaction_Type",
    "Amount",
    "Currency",
    "Counterparty",
    "Category",
    "Payment_Method",
    TARGET_COLUMN,
    "Risk_Type",
    "Incident_Severity",
    "Error_Code",
    "User_ID",
    "System_Latency",
    "Login_Frequency",
    "Failed_Attempts",
    "IP_Region",
)

LEAKAGE_COLUMNS = ("Risk_Type", "Incident_Severity", "Error_Code")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataset contract and return a normalized copy."""
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    validated = frame.copy()
    if (
        validated[ID_COLUMN].isna().any()
        or validated[ID_COLUMN].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"{ID_COLUMN} contains missing or empty values")
    if validated[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} values must be unique")

    target = pd.to_numeric(validated[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing or non-numeric values")
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1")
    if set(target.unique()) != {0, 1}:
        raise ValueError(f"{TARGET_COLUMN} must contain both classes 0 and 1")
    validated[TARGET_COLUMN] = target.astype("int8")
    return validated


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV dataset."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    try:
        # Keep legitimate values such as the ISO region code "NA" and the dataset's
        # literal post-event placeholder "None"; only empty CSV cells mean missing.
        frame = pd.read_csv(source, keep_default_na=False, na_values=[""])
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not parse dataset CSV: {source}") from error
    return validate_dataset(frame)


def audit_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable summary, including target leakage checks."""
    validated = validate_dataset(frame)
    target_counts = validated[TARGET_COLUMN].value_counts().sort_index()
    leakage = {}
    for column in LEAKAGE_COLUMNS:
        values = validated[column].fillna("<missing>").astype(str)
        value_target_pairs = pd.DataFrame(
            {"value": values.to_numpy(), "target": validated[TARGET_COLUMN].to_numpy()}
        )
        leakage[column] = {
            "unique_values": sorted(values.unique().tolist()),
            "perfectly_encodes_target": bool(
                value_target_pairs.groupby("value", dropna=False)["target"].nunique().le(1).all()
            ),
        }

    numeric = validated.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in validated.columns if column not in numeric]
    return {
        "rows": len(validated),
        "columns": len(validated.columns),
        "column_names": validated.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in validated.dtypes.items()},
        "missing_values": {column: int(count) for column, count in validated.isna().sum().items()},
        "duplicate_rows": int(validated.duplicated().sum()),
        "duplicate_transaction_ids": int(validated[ID_COLUMN].duplicated().sum()),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "positive_count": int(target_counts.get(1, 0)),
        "negative_count": int(target_counts.get(0, 0)),
        "positive_rate": float(validated[TARGET_COLUMN].mean()),
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
        "leakage_checks": leakage,
        "constant_columns": [
            column for column in validated.columns if validated[column].nunique(dropna=False) <= 1
        ],
        "date_min": str(validated["Date"].min()),
        "date_max": str(validated["Date"].max()),
    }


def write_audit_summary(frame: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Write a dataset audit report and return it."""
    summary = audit_dataset(frame)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
