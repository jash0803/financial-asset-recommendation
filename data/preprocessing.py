import pandas as pd


def preprocess_data(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Buy transactions and sort chronologically."""
    buys = transactions_df[transactions_df.transactionType == "Buy"].copy()
    buys["timestamp"] = pd.to_datetime(buys.timestamp)
    buys = buys.sort_values("timestamp")
    return buys


def build_rating_matrix(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a user-item interaction matrix from training buy transactions.

    Returns (rating_matrix, rating_df) where rating_df is the long-form
    counts and rating_matrix is the pivoted sparse matrix.
    """
    rating_df = (
        train_df.groupby(["customerID", "ISIN"]).size().reset_index(name="count")
    )
    rating_matrix = rating_df.pivot(
        index="customerID", columns="ISIN", values="count"
    ).fillna(0)
    return rating_matrix, rating_df


def compute_momentum(close_prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 30-day and 90-day return momentum per ISIN.

    Returns a DataFrame indexed by ISIN with columns
    ``return_30d`` and ``return_90d``.
    """
    df = close_prices_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ISIN", "timestamp"])

    latest = df.groupby("ISIN").tail(1).set_index("ISIN")
    max_date = df["timestamp"].max()

    def _get_return(days: int) -> pd.Series:
        cutoff = max_date - pd.Timedelta(days=days)
        past = df[df["timestamp"] <= cutoff].groupby("ISIN").tail(1).set_index("ISIN")
        merged = latest[["closePrice"]].join(
            past[["closePrice"]], rsuffix="_past", how="inner"
        )
        return (
            (merged["closePrice"] - merged["closePrice_past"])
            / merged["closePrice_past"]
        )

    return pd.DataFrame({
        "return_30d": _get_return(30),
        "return_90d": _get_return(90),
    })
