import pandas as pd

from .base import BaseRecommender


def _normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng > 0:
        return (s - s.min()) / rng
    return s


class HybridRecommender(BaseRecommender):
    """Weighted hybrid that combines an arbitrary list of recommenders.

    Accepts ``(BaseRecommender, weight)`` pairs. The UI can dynamically
    generate sliders for each registered recommender, so adding a new
    algorithm requires zero changes here.
    """

    def __init__(self, recommenders: list[tuple[BaseRecommender, float]]):
        self._recommenders = recommenders

    @property
    def name(self) -> str:
        return "Hybrid"

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "HybridRecommender":
        for rec, _ in self._recommenders:
            rec.fit(train_df, **kwargs)
        return self

    def predict(self, customer_id: str, n: int = 10) -> pd.Series:
        combined: pd.Series | None = None

        for rec, weight in self._recommenders:
            scores = _normalize(rec.predict(customer_id, n=n))
            if combined is None:
                combined = weight * scores
            else:
                combined = combined.add(weight * scores, fill_value=0)

        if combined is None:
            return pd.Series(dtype=float)
        return combined

    def recommend(
        self,
        customer_id: str,
        rating_df: pd.DataFrame,
        n: int = 10,
    ) -> pd.Series:
        """Generate top-N recommendations, excluding already-bought assets."""
        scores = self.predict(customer_id, n=n)

        bought = (
            rating_df[rating_df["customerID"] == customer_id]["ISIN"].unique()
            if not rating_df[rating_df["customerID"] == customer_id].empty
            else []
        )
        scores = scores.drop(labels=bought, errors="ignore")
        return scores.sort_values(ascending=False).head(n)

    @property
    def sub_recommenders(self) -> list[tuple[BaseRecommender, float]]:
        return self._recommenders

    def update_weights(self, weights: list[float]) -> None:
        if len(weights) != len(self._recommenders):
            raise ValueError(
                f"Expected {len(self._recommenders)} weights, got {len(weights)}"
            )
        self._recommenders = [
            (rec, w) for (rec, _), w in zip(self._recommenders, weights)
        ]
