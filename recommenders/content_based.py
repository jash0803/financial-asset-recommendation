import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Content-based filtering using asset features, profitability, and price momentum."""

    def __init__(self):
        self._encoded_features: pd.DataFrame | None = None
        self._rating_df: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "Content-Based"

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "ContentBasedRecommender":
        asset_df: pd.DataFrame = kwargs["asset_df"]
        limit_prices_df: pd.DataFrame = kwargs["limit_prices_df"]
        self._rating_df = kwargs["rating_df"]
        momentum_df: pd.DataFrame | None = kwargs.get("momentum_df")

        features = asset_df.drop_duplicates(subset="ISIN", keep="last")[
            ["ISIN", "assetCategory", "assetSubCategory", "sector", "industry", "marketID"]
        ].copy()

        features = features.merge(
            limit_prices_df[["ISIN", "profitability"]], on="ISIN", how="left"
        )
        features["profitability"] = features["profitability"].fillna(
            features["profitability"].median()
        )
        features["sector"] = features["sector"].fillna("Unknown")
        features["industry"] = features["industry"].fillna("Unknown")

        cat_cols = ["assetCategory", "assetSubCategory", "sector", "industry", "marketID"]
        encoded = pd.get_dummies(features[cat_cols])
        encoded["profitability"] = features["profitability"].values

        if momentum_df is not None:
            merged_mom = features[["ISIN"]].merge(momentum_df, on="ISIN", how="left")
            encoded["return_30d"] = merged_mom["return_30d"].fillna(0).values
            encoded["return_90d"] = merged_mom["return_90d"].fillna(0).values

        encoded.index = features["ISIN"]
        self._encoded_features = encoded
        return self

    def predict(self, customer_id: str, n: int = 10) -> pd.Series:
        if self._encoded_features is None or self._rating_df is None:
            raise RuntimeError("Call fit() before predict()")

        user_assets = (
            self._rating_df[self._rating_df["customerID"] == customer_id]["ISIN"]
            .unique()
            .tolist()
        )
        user_assets = [a for a in user_assets if a in self._encoded_features.index]

        if len(user_assets) == 0:
            return pd.Series(0.5, index=self._encoded_features.index)

        user_profile = self._encoded_features.loc[user_assets].mean()
        sim = cosine_similarity(
            user_profile.values.reshape(1, -1), self._encoded_features.values
        )[0]

        return pd.Series(sim, index=self._encoded_features.index)
