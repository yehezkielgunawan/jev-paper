from __future__ import annotations

from jev_risk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features


def test_build_features_excludes_ids_target_and_post_event_leakage(sample_transactions):
    features = build_features(sample_transactions)

    forbidden = {
        "Transaction_ID",
        "Account_Number",
        "Counterparty",
        "Currency",
        "Risk_Incident",
        "Risk_Type",
        "Incident_Severity",
        "Error_Code",
        "User_ID",
        "Date",
    }
    assert forbidden.isdisjoint(features.columns)
    assert set(features.columns) == set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    assert {"Transaction_Type", "Category", "Payment_Method", "IP_Region"}.issubset(
        features.columns
    )


def test_build_features_derives_calendar_fields(sample_transactions):
    features = build_features(sample_transactions)

    assert {"date_year", "date_month", "date_dayofweek", "date_dayofyear"}.issubset(
        features.columns
    )
    assert features.loc[0, "date_month"] == 1


def test_build_features_rejects_invalid_dates(sample_transactions):
    sample_transactions.loc[0, "Date"] = "not-a-date"

    try:
        build_features(sample_transactions)
    except ValueError as error:
        assert "Date" in str(error)
    else:
        raise AssertionError("Expected an invalid Date to be rejected")


def test_build_features_rejects_non_numeric_non_missing_amount(sample_transactions):
    sample_transactions["Amount"] = sample_transactions["Amount"].astype(object)
    sample_transactions.loc[0, "Amount"] = "not-an-amount"

    try:
        build_features(sample_transactions)
    except ValueError as error:
        assert "Amount" in str(error)
    else:
        raise AssertionError("Expected invalid numeric input to be rejected")
