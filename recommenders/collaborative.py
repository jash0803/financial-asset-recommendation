import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from .base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    """SVD-based matrix-factorisation collaborative filtering."""

    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self._pred_df: pd.DataFrame | None = None
        self._all_items: pd.Index | None = None

    @property
    def name(self) -> str:
        return "Collaborative Filtering"

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "CollaborativeRecommender":
        rating_matrix: pd.DataFrame = kwargs["rating_matrix"]
        self._all_items = rating_matrix.columns

        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        U = svd.fit_transform(rating_matrix)
        V = svd.components_.T

        pred_ratings = np.dot(U, V.T)
        self._pred_df = pd.DataFrame(
            pred_ratings, index=rating_matrix.index, columns=rating_matrix.columns
        )
        return self

    def predict(self, customer_id: str, n: int = 10) -> pd.Series:
        if self._pred_df is None:
            raise RuntimeError("Call fit() before predict()")

        if customer_id in self._pred_df.index:
            return self._pred_df.loc[customer_id]
        return pd.Series(0.0, index=self._all_items)

    @property
    def pred_df(self) -> pd.DataFrame:
        """Expose full prediction matrix for RMSE evaluation."""
        if self._pred_df is None:
            raise RuntimeError("Call fit() before accessing pred_df")
        return self._pred_df
