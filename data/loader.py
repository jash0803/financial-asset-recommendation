from dataclasses import dataclass

import pandas as pd

from config import DataConfig


@dataclass
class DataBundle:
    """Container for all loaded dataframes."""
    assets: pd.DataFrame
    customers: pd.DataFrame
    transactions: pd.DataFrame
    limit_prices: pd.DataFrame
    close_prices: pd.DataFrame
    markets: pd.DataFrame


def _dedup_by_isin(df: pd.DataFrame, date_col: str = "timestamp") -> pd.DataFrame:
    """Keep the latest row per ISIN (assets can appear multiple times)."""
    if date_col in df.columns:
        df = df.sort_values(date_col)
    return df.drop_duplicates(subset="ISIN", keep="last").reset_index(drop=True)


def load_data(cfg: DataConfig | None = None) -> DataBundle:
    if cfg is None:
        cfg = DataConfig()

    base = cfg.data_dir

    assets = pd.read_csv(base / cfg.asset_file)
    assets = _dedup_by_isin(assets, "timestamp")

    limit_prices = pd.read_csv(base / cfg.limit_prices_file)
    limit_prices = _dedup_by_isin(limit_prices, "maxDate")

    return DataBundle(
        assets=assets,
        customers=pd.read_csv(base / cfg.customer_file),
        transactions=pd.read_csv(base / cfg.transactions_file),
        limit_prices=limit_prices,
        close_prices=pd.read_csv(base / cfg.close_prices_file),
        markets=pd.read_csv(base / cfg.markets_file),
    )
