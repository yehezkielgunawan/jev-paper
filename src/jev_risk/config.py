"""Shared experiment configuration and dataset feature policy."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "accounting_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "Risk_Incident"
ID_COLUMN = "Transaction_ID"
BOOTSTRAP_ITERATIONS = 1_000
CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

EXCLUDED_FEATURES = (
    "Transaction_ID",
    "Account_Number",
    "Currency",
    "Counterparty",
    "Risk_Incident",
    "Risk_Type",
    "Incident_Severity",
    "Error_Code",
    "User_ID",
    "Date",
)

CATEGORICAL_FEATURES = (
    "Transaction_Type",
    "Category",
    "Payment_Method",
    "IP_Region",
)

NUMERIC_FEATURES = (
    "Amount",
    "System_Latency",
    "Login_Frequency",
    "Failed_Attempts",
    "date_year",
    "date_month",
    "date_dayofweek",
    "date_dayofyear",
)

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
