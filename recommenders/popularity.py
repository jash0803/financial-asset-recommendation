import pandas as pd

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Global popularity baseline -- ranks assets by total purchase frequency.

    Useful as a non-personalised baseline and as a cold-start fallback for
    users with no transaction history.
    """

    def __init__(self):
        self._popularity_scores: pd.Series | None = None

    @property
    def name(self) -> str:
        return "Popularity"

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "PopularityRecommender":
        counts = train_df.groupby("ISIN").size()
        self._popularity_scores = counts / counts.max()
        return self

    def predict(self, customer_id: str, n: int = 10) -> pd.Series:
        if self._popularity_scores is None:
            raise RuntimeError("Call fit() before predict()")
        return self._popularity_scores
