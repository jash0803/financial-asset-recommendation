import numpy as np
import pandas as pd

from .base import BaseRecommender

RISK_MAP = {"Conservative": 1, "Income": 2, "Balanced": 3, "Aggressive": 4}
CAP_MAP = {"CAP_LT30K": 1, "CAP_30K_80K": 2, "CAP_80K_300K": 3, "CAP_GT300K": 4}


def _normalize_label(label) -> str | None:
    if pd.isna(label) or label == "Not_Available":
        return None
    return label.replace("Predicted_", "")


class DemographicRecommender(BaseRecommender):
    """Scores assets by alignment between customer demographics and asset categories.

    For each asset category, we compute the average demographic profile of
    users who actually purchased assets in that category, then measure how
    similar the target user is to that average.
    """

    def __init__(self):
        self._asset_df: pd.DataFrame | None = None
        self._customer_df: pd.DataFrame | None = None
        self._category_vectors: dict[str, np.ndarray] | None = None

    @property
    def name(self) -> str:
        return "Demographic"

    def fit(self, train_df: pd.DataFrame, **kwargs) -> "DemographicRecommender":
        self._asset_df = kwargs["asset_df"]
        self._customer_df = kwargs["customer_df"]

        customer_df = self._customer_df
        asset_df = self._asset_df

        cdf = customer_df.copy()
        cdf["riskLevel"] = cdf["riskLevel"].apply(_normalize_label)
        cdf["investmentCapacity"] = cdf["investmentCapacity"].apply(_normalize_label)
        cdf = cdf.dropna(subset=["riskLevel", "investmentCapacity"])
        cdf = cdf[
            cdf["riskLevel"].isin(RISK_MAP) & cdf["investmentCapacity"].isin(CAP_MAP)
        ]
        cdf = cdf.sort_values("timestamp").drop_duplicates("customerID", keep="last")

        # Map transactions to asset categories
        tx_with_cat = train_df[["customerID", "ISIN"]].merge(
            asset_df[["ISIN", "assetCategory"]], on="ISIN", how="left"
        )

        self._category_vectors = {}
        neutral = np.array([2.5, 2.5, 0.5, 0.5])

        for cat in asset_df["assetCategory"].unique():
            buyers = tx_with_cat[tx_with_cat["assetCategory"] == cat]["customerID"].unique()
            cat_demos = cdf[cdf["customerID"].isin(buyers)]

            if cat_demos.empty:
                self._category_vectors[cat] = neutral
            else:
                self._category_vectors[cat] = np.array([
                    cat_demos["riskLevel"].map(RISK_MAP).mean(),
                    cat_demos["investmentCapacity"].map(CAP_MAP).mean(),
                    (cat_demos["customerType"] == "Premium").mean(),
                    (cat_demos["customerType"] == "Professional").mean(),
                ])

        return self

    def predict(self, customer_id: str, n: int = 10) -> pd.Series:
        if self._asset_df is None or self._customer_df is None:
            raise RuntimeError("Call fit() before predict()")

        asset_df = self._asset_df
        customer_df = self._customer_df

        cdf_sorted = customer_df.sort_values("timestamp").drop_duplicates(
            "customerID", keep="last"
        )
        user_info = cdf_sorted[cdf_sorted["customerID"] == customer_id]

        if user_info.empty:
            return pd.Series(0.5, index=asset_df["ISIN"])

        risk = _normalize_label(user_info["riskLevel"].values[0])
        cap = _normalize_label(user_info["investmentCapacity"].values[0])
        customer_type = user_info["customerType"].values[0]

        if risk not in RISK_MAP or cap not in CAP_MAP:
            return pd.Series(0.5, index=asset_df["ISIN"])

        user_vector = np.array([
            RISK_MAP[risk],
            CAP_MAP[cap],
            1 if customer_type == "Premium" else 0,
            1 if customer_type == "Professional" else 0,
        ])

        weights = np.array([0.4, 0.3, 0.2, 0.1])
        max_dist = np.sqrt(np.sum(weights * np.array([3, 3, 1, 1]) ** 2))

        category_scores: dict[str, float] = {}
        for cat, avg_vector in self._category_vectors.items():
            sim = 1 - np.sqrt(np.sum(weights * (user_vector - avg_vector) ** 2)) / max_dist
            category_scores[cat] = sim

        scores = asset_df["assetCategory"].map(category_scores).fillna(0.5)
        return pd.Series(scores.values, index=asset_df["ISIN"])
