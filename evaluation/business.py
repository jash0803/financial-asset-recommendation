"""Business / catalog-level evaluation metrics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

from recommenders.hybrid import HybridRecommender

logger = logging.getLogger(__name__)


# ────────────────────── individual metrics ──────────────────────


def roi_at_k(
    recommendations: pd.Series,
    limit_prices_df: pd.DataFrame,
    k: int = 10,
) -> float | None:
    """Average profitability of the top-K recommended assets."""
    if recommendations is None or len(recommendations) == 0:
        return None

    top_k = recommendations.head(k)
    prof = limit_prices_df.set_index("ISIN")["profitability"]
    valid = top_k.index.intersection(prof.index)
    if valid.empty:
        return None
    return float(prof.loc[valid].mean())


def coverage_at_k(
    all_recommendations: dict[str, pd.Series],
    catalog_size: int,
    k: int = 10,
) -> float:
    """Fraction of the full catalog that appears in at least one user's top-K."""
    recommended_items: set[str] = set()
    for recs in all_recommendations.values():
        recommended_items.update(recs.head(k).index.tolist())
    return len(recommended_items) / catalog_size if catalog_size > 0 else 0.0


def diversity_at_k(
    recommendations: pd.Series,
    feature_matrix: pd.DataFrame,
    k: int = 10,
) -> float:
    """Average pairwise cosine distance among top-K recommended items.

    Higher values mean more diverse recommendations.
    """
    top_k = recommendations.head(k)
    valid = top_k.index.intersection(feature_matrix.index)
    if len(valid) < 2:
        return 0.0

    vecs = feature_matrix.loc[valid].values
    sim = _cos_sim(vecs)
    n = len(valid)
    total_dist = sum(
        1 - sim[i][j] for i in range(n) for j in range(i + 1, n)
    )
    pairs = n * (n - 1) / 2
    return total_dist / pairs if pairs > 0 else 0.0


def novelty_at_k(
    recommendations: pd.Series,
    item_popularity: pd.Series,
    k: int = 10,
) -> float:
    """Average self-information of top-K items.

    Items that are less popular (lower probability of being purchased)
    contribute higher novelty.  ``item_popularity`` should contain
    probabilities (purchase_count / total_purchases).
    """
    top_k = recommendations.head(k)
    valid = top_k.index.intersection(item_popularity.index)
    if valid.empty:
        return 0.0

    pops = item_popularity.loc[valid].clip(lower=1e-10)
    return float((-np.log2(pops)).mean())


# ────────────────────── aggregate helper ──────────────────────


def _build_feature_matrix(
    asset_df: pd.DataFrame, limit_prices_df: pd.DataFrame
) -> pd.DataFrame:
    """One-hot encode asset features for diversity calculation."""
    feats = asset_df[
        ["ISIN", "assetCategory", "assetSubCategory", "sector", "industry", "marketID"]
    ].copy()
    feats = feats.merge(
        limit_prices_df[["ISIN", "profitability"]], on="ISIN", how="left"
    )
    feats["profitability"] = feats["profitability"].fillna(feats["profitability"].median())
    feats["sector"] = feats["sector"].fillna("Unknown")
    feats["industry"] = feats["industry"].fillna("Unknown")

    cat_cols = ["assetCategory", "assetSubCategory", "sector", "industry", "marketID"]
    encoded = pd.get_dummies(feats[cat_cols])
    encoded["profitability"] = feats["profitability"].values
    encoded.index = feats["ISIN"]
    return encoded


def evaluate_business_metrics(
    hybrid: HybridRecommender,
    test_df: pd.DataFrame,
    rating_matrix: pd.DataFrame,
    rating_df: pd.DataFrame,
    asset_df: pd.DataFrame,
    limit_prices_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 10,
    max_users: int = 500,
) -> dict[str, float | None]:
    """Compute business metrics on a random sample of test users.

    ``max_users`` caps how many users are evaluated (default 500).
    """
    feature_matrix = _build_feature_matrix(asset_df, limit_prices_df)

    total_purchases = len(train_df)
    item_counts = train_df.groupby("ISIN").size()
    item_popularity = item_counts / total_purchases

    catalog_size = len(asset_df["ISIN"].unique())
    all_recs: dict[str, pd.Series] = {}

    roi_vals: list[float] = []
    div_vals: list[float] = []
    nov_vals: list[float] = []

    users = test_df["customerID"].unique() if not test_df.empty else np.array([])
    if len(users) > max_users:
        rng = np.random.RandomState(42)
        users = rng.choice(users, size=max_users, replace=False)

    total = len(users)
    logger.info("[Business Eval] Starting evaluation on %d users (k=%d)", total, k)
    evaluated = 0

    for uid in users:
        if uid not in rating_matrix.index:
            continue
        try:
            recs = hybrid.recommend(uid, rating_df, n=k)
        except Exception:
            continue
        if recs is None or len(recs) == 0:
            continue

        all_recs[uid] = recs

        r = roi_at_k(recs, limit_prices_df, k)
        if r is not None:
            roi_vals.append(r)
        div_vals.append(diversity_at_k(recs, feature_matrix, k))
        nov_vals.append(novelty_at_k(recs, item_popularity, k))
        evaluated += 1

        if evaluated % 100 == 0:
            logger.info("[Business Eval] %d / %d users evaluated…", evaluated, total)

    final = {
        "ROI@K": float(np.mean(roi_vals)) if roi_vals else None,
        "Coverage@K": coverage_at_k(all_recs, catalog_size, k),
        "Diversity@K": float(np.mean(div_vals)) if div_vals else None,
        "Novelty@K": float(np.mean(nov_vals)) if nov_vals else None,
    }
    logger.info("[Business Eval] Done. %d users evaluated. Results: %s", evaluated, final)
    return final
