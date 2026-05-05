import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class KNNRecommender(BaseRecommender):
    """Item-item KNN collaborative filtering.

    For each candidate item, the score is the weighted average of the user's
    interactions with the k most similar items.
    """

    def __init__(self, k: int = 20, metric: str = "cosine"):
        self.k = k
        self.metric = metric
        self._item_sim: pd.DataFrame | None = None
        self._rating_matrix: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "KNN (Item-Item)"

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "KNNRecommender":
        rating_matrix: pd.DataFrame = kwargs["rating_matrix"]
        self._rating_matrix = rating_matrix

        item_vectors = rating_matrix.T.values
        sim_matrix = cosine_similarity(item_vectors)
        np.fill_diagonal(sim_matrix, 0)
        self._item_sim = pd.DataFrame(
            sim_matrix,
            index=rating_matrix.columns,
            columns=rating_matrix.columns,
        )
        return self

    def predict(self, customer_id: str, n: int = 10) -> pd.Series:
        if self._item_sim is None or self._rating_matrix is None:
            raise RuntimeError("Call fit() before predict()")

        if customer_id not in self._rating_matrix.index:
            return pd.Series(0.0, index=self._rating_matrix.columns)

        user_ratings = self._rating_matrix.loc[customer_id]
        interacted = user_ratings[user_ratings > 0].index.tolist()

        scores: dict[str, float] = {}
        for item in self._rating_matrix.columns:
            sims = self._item_sim.loc[item, interacted]
            top_k = sims.nlargest(self.k)
            if top_k.sum() > 0:
                weighted = (top_k * user_ratings[top_k.index]).sum() / top_k.sum()
            else:
                weighted = 0.0
            scores[item] = weighted

        return pd.Series(scores)
