"""Train/test splitting strategies for recommendation evaluation."""

from __future__ import annotations

import pandas as pd


def leave_one_out_split(
    buys: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's chronologically last purchase as the test item."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, grp in buys.groupby("customerID"):
        if len(grp) < 2:
            train_parts.append(grp)
        else:
            train_parts.append(grp.iloc[:-1])
            test_parts.append(grp.iloc[-1:])

    train_df = pd.concat(train_parts)
    test_df = (
        pd.concat(test_parts) if test_parts else pd.DataFrame(columns=buys.columns)
    )
    return train_df, test_df


def temporal_split(
    buys: pd.DataFrame,
    ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Global temporal split -- earliest *ratio* fraction goes to train."""
    buys = buys.sort_values("timestamp")
    split_idx = int(len(buys) * ratio)
    return buys.iloc[:split_idx].copy(), buys.iloc[split_idx:].copy()
