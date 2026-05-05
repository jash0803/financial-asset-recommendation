"""Ranking-oriented evaluation metrics for recommendation quality."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from recommenders.hybrid import HybridRecommender

logger = logging.getLogger(__name__)


# ────────────────────── per-user helpers ──────────────────────


def _relevance_vector(recommended_isins: pd.Index, relevant_isins: set) -> list[int]:
    return [1 if isin in relevant_isins else 0 for isin in recommended_isins]


# ────────────────────── individual metrics ──────────────────────


def precision_at_k(recommended: pd.Series, relevant: set, k: int) -> float:
    top_k = recommended.head(k)
    hits = sum(1 for isin in top_k.index if isin in relevant)
    return hits / k


def recall_at_k(recommended: pd.Series, relevant: set, k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    top_k = recommended.head(k)
    hits = sum(1 for isin in top_k.index if isin in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: pd.Series, relevant: set, k: int) -> float:
    top_k = recommended.head(k)
    rels = _relevance_vector(top_k.index, relevant)

    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: pd.Series, relevant: set, k: int) -> float:
    """Average Precision at K for a single user."""
    top_k = recommended.head(k)
    score = 0.0
    hits = 0
    for i, isin in enumerate(top_k.index, start=1):
        if isin in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k) if relevant else 0.0


def mrr_at_k(recommended: pd.Series, relevant: set, k: int) -> float:
    """Reciprocal Rank -- 1/rank of the first relevant item within top-K."""
    top_k = recommended.head(k)
    for i, isin in enumerate(top_k.index, start=1):
        if isin in relevant:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(recommended: pd.Series, relevant: set, k: int) -> float:
    """1.0 if at least one relevant item appears in top-K, else 0.0."""
    top_k = recommended.head(k)
    return 1.0 if any(isin in relevant for isin in top_k.index) else 0.0


# ────────────────────── RMSE (collaborative only) ──────────────────────


def compute_rmse(pred_df: pd.DataFrame, test_df: pd.DataFrame) -> float | None:
    if test_df.empty:
        return None

    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        u, i = row["customerID"], row["ISIN"]
        if u in pred_df.index and i in pred_df.columns:
            y_true.append(1.0)
            y_pred.append(pred_df.at[u, i])

    if not y_true:
        return None
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ────────────────────── aggregate over all test users ──────────────────────


def evaluate_ranking_metrics(
    hybrid: HybridRecommender,
    test_df: pd.DataFrame,
    rating_matrix: pd.DataFrame,
    rating_df: pd.DataFrame,
    k: int = 10,
    max_users: int = 500,
) -> dict[str, float | None]:
    """Run all ranking metrics on a random sample of test users.

    ``max_users`` caps how many users are evaluated (default 500) to keep
    runtime reasonable while still being statistically representative.
    """
    _empty: dict[str, float | None] = {
        "Precision@K": None,
        "Recall@K": None,
        "nDCG@K": None,
        "MAP@K": None,
        "MRR@K": None,
        "HitRate@K": None,
    }
    if test_df.empty:
        return _empty

    # Deduplicate to one test row per user and sample
    user_test = test_df.drop_duplicates(subset="customerID")
    if len(user_test) > max_users:
        user_test = user_test.sample(n=max_users, random_state=42)

    total = len(user_test)
    logger.info("[Ranking Eval] Starting evaluation on %d users (k=%d)", total, k)

    results: dict[str, list[float]] = {n: [] for n in _empty}
    evaluated = 0

    for _, row in user_test.iterrows():
        uid, test_isin = row["customerID"], row["ISIN"]
        if uid not in rating_matrix.index:
            continue

        try:
            recs = hybrid.recommend(uid, rating_df, n=k)
        except Exception:
            continue

        if recs is None or len(recs) == 0:
            continue

        relevant = {test_isin}
        results["Precision@K"].append(precision_at_k(recs, relevant, k))
        results["Recall@K"].append(recall_at_k(recs, relevant, k))
        results["nDCG@K"].append(ndcg_at_k(recs, relevant, k))
        results["MAP@K"].append(map_at_k(recs, relevant, k))
        results["MRR@K"].append(mrr_at_k(recs, relevant, k))
        results["HitRate@K"].append(hit_rate_at_k(recs, relevant, k))
        evaluated += 1

        if evaluated % 100 == 0:
            logger.info("[Ranking Eval] %d / %d users evaluated…", evaluated, total)

    final = {
        name: float(np.mean(vals)) if vals else None
        for name, vals in results.items()
    }
    logger.info("[Ranking Eval] Done. %d users evaluated. Results: %s", evaluated, final)
    return final
