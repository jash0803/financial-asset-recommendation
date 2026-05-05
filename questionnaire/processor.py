"""Logic for scoring questionnaire responses and updating customer profiles."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

_ANSWER_SCORE = {"a": 4, "b": 3, "c": 2, "d": 1, "e": 0}

_RISK_WEIGHTS: dict[str, float] = {
    "q16": 0.3,
    "q17": 0.3,
    "q18": 0.2,
    "q19": 0.2,
}


def process_questionnaire_responses(
    responses: dict[str, str],
) -> tuple[str, str]:
    """Derive (risk_level, investment_capacity) from raw answers."""
    risk_score = sum(
        weight * _ANSWER_SCORE.get(responses.get(q, "e"), 0)
        for q, weight in _RISK_WEIGHTS.items()
    )

    if risk_score >= 3.5:
        risk_level = "Aggressive"
    elif risk_score >= 2.5:
        risk_level = "Balanced"
    elif risk_score >= 1.5:
        risk_level = "Income"
    else:
        risk_level = "Conservative"

    inv = responses.get("q13", "e")
    capacity_map = {
        "a": "CAP_GT300K",
        "b": "CAP_80K_300K",
        "c": "CAP_30K_80K",
        "d": "CAP_30K_80K",
    }
    investment_capacity = capacity_map.get(inv, "CAP_LT30K")

    return risk_level, investment_capacity


def update_customer_profile(
    customer_id: str,
    risk_level: str,
    investment_capacity: str,
    customer_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append a new row reflecting updated questionnaire results."""
    now = datetime.now()
    new_row = pd.DataFrame({
        "customerID": [customer_id],
        "customerType": ["Mass"],
        "riskLevel": [risk_level],
        "investmentCapacity": [investment_capacity],
        "lastQuestionnaireDate": [now.strftime("%Y-%m-%d")],
        "timestamp": [now.strftime("%Y-%m-%d %H:%M:%S")],
    })
    return pd.concat([customer_df, new_row], ignore_index=True)
