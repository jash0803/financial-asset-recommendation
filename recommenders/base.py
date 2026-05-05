from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Interface that every recommender must implement."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs) -> "BaseRecommender":
        """Train / fit the model on training data."""

    @abstractmethod
    def predict(self, customer_id: str, n: int = 10) -> pd.Series:
        """Return a Series of scores indexed by ISIN, highest first."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name shown in the UI."""
