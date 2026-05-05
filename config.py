from pydantic import BaseModel
from pathlib import Path


class DataConfig(BaseModel):
    data_dir: Path = Path("FAR-Trans-Data")
    asset_file: str = "asset_information.csv"
    customer_file: str = "customer_information.csv"
    transactions_file: str = "transactions.csv"
    limit_prices_file: str = "limit_prices.csv"
    close_prices_file: str = "close_prices.csv"
    markets_file: str = "markets.csv"
    questionnaires_file: str = "questionnaires.csv"


class ModelConfig(BaseModel):
    svd_components: int = 5
    knn_neighbors: int = 20
    knn_metric: str = "cosine"
    default_weights: list[float] = [0.25, 0.25, 0.2, 0.15, 0.15]


class CacheConfig(BaseModel):
    enabled: bool = True
    cache_dir: Path = Path(".cache")
    ttl_seconds: int = 3600


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    cache: CacheConfig = CacheConfig()
    top_n: int = 10
