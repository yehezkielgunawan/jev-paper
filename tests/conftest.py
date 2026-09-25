from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    rows = []
    for index in range(40):
        risky = int(index % 5 == 0)
        rows.append(
            {
                "Transaction_ID": f"TXN{index:05d}",
                "Date": f"2024-{index % 12 + 1:02d}-{index % 27 + 1:02d}",
                "Account_Number": 100000 + index,
                "Transaction_Type": "Debit" if index % 2 else "Credit",
                "Amount": float(index * 10 + 1),
                "Currency": "USD",
                "Counterparty": f"Counterparty {index}",
                "Category": "Payroll" if index % 2 else "Sales",
                "Payment_Method": "Cash" if index % 3 else "Credit Card",
                "Risk_Incident": risky,
                "Risk_Type": ("Fraud", "Error", "Misstatement")[(index // 5) % 3]
                if risky
                else "None",
                "Incident_Severity": ("Low", "Medium", "High")[(index // 5) % 3]
                if risky
                else "None",
                "Error_Code": ("F001", "E002", "M003")[(index // 5) % 3] if risky else "None",
                "User_ID": f"U{index % 4:03d}",
                "System_Latency": float(100 + index),
                "Login_Frequency": index % 10,
                "Failed_Attempts": index % 6,
                "IP_Region": f"R{index % 3}",
            }
        )
    return pd.DataFrame(rows)
