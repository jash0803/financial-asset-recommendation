# FAR-Trans Asset Recommender

A modular hybrid recommendation system for financial assets, built on the [FAR-Trans dataset](https://arxiv.org/abs/2407.17727) -- a real-world investment dataset from a large European financial institution containing stocks, bonds, and mutual funds.

Created by [Jash Shah](https://www.linkedin.com/in/jashshah0803/).

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
financial-asset-recommendation/
├── app.py                       # Streamlit UI entry point
├── config.py                    # Pydantic configuration models
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── loader.py                # CSV loading, ISIN deduplication, DataBundle
│   └── preprocessing.py         # Buy filtering, rating matrix, price momentum
├── recommenders/
│   ├── __init__.py
│   ├── base.py                  # Abstract BaseRecommender interface
│   ├── collaborative.py         # SVD matrix factorisation
│   ├── content_based.py         # Asset feature similarity + momentum signals
│   ├── demographic.py           # Customer profile → asset category alignment
│   ├── popularity.py            # Global purchase frequency baseline
│   ├── knn.py                   # Item-item KNN collaborative filtering
│   └── hybrid.py                # Weighted combiner with dynamic weight updates
├── evaluation/
│   ├── __init__.py
│   ├── ranking.py               # Precision, Recall, MAP, MRR, Hit Rate, nDCG @K
│   ├── business.py              # ROI, Coverage, Diversity, Novelty @K
│   └── splitters.py             # Leave-one-out and temporal train/test splits
├── cache/
│   ├── __init__.py
│   └── manager.py               # Joblib disk cache with TTL invalidation
├── questionnaire/
│   ├── __init__.py
│   ├── questions.py             # Parser for the MiFID questionnaire file
│   └── processor.py             # Risk level and investment capacity scoring
└── FAR-Trans-Data/              # Dataset files
```

## Features

### Existing Users
- Select a customer ID from the dropdown and generate personalised recommendations.
- Adjust component weights via sidebar sliders (auto-normalised to sum to 1).
- View enriched recommendation tables with asset metadata, market info, profitability, and 30d/90d price momentum.
- Opt-in evaluation metrics computed on a sampled test set.

### New Users (Cold Start)
- Toggle "I'm a new user" in the sidebar to enter onboarding mode.
- Complete the full 25-question MiFID risk assessment questionnaire.
- History-dependent recommenders (CF, Content-Based, KNN) are automatically disabled.
- Recommendations are generated using Popularity and Demographic algorithms.

### Dataset Explorer
- View summary statistics (asset count, customers, transactions, markets).
- Interactive charts for asset categories and transaction channels.
- Browseable markets table.

## Dataset

The FAR-Trans dataset includes:

| File | Records | Description |
|------|---------|-------------|
| `asset_information.csv` | ~836 | Stocks, bonds, mutual funds with category, sector, and market |
| `customer_information.csv` | ~32K | Customer profiles with risk level and investment capacity |
| `transactions.csv` | ~388K | Buy/sell transactions with value, units, and channel |
| `close_prices.csv` | ~560K | Daily close prices for all assets |
| `limit_prices.csv` | ~807 | ROI and price range per asset |
| `markets.csv` | ~38 | Market metadata (country, trading hours) |
| `questionnaires.csv` | 25 Qs | MiFID risk assessment questionnaire |

## Recommendation Algorithms

| # | Algorithm | Description | Cold-Start |
|---|-----------|-------------|:----------:|
| 1 | **Collaborative Filtering** | TruncatedSVD on the user-item interaction matrix (implicit buy counts) | No |
| 2 | **Content-Based** | Cosine similarity between user profile and asset features (category, sector, profitability, 30d/90d price momentum) | No |
| 3 | **Demographic** | Matches user risk/capacity profile against the average demographics of buyers in each asset category | Yes |
| 4 | **Popularity** | Global purchase frequency; non-personalised baseline | Yes |
| 5 | **KNN (Item-Item)** | Item-item cosine similarity on the rating matrix | No |

The **Hybrid** combiner normalises each algorithm's scores to [0, 1] and blends them with configurable weights that are auto-normalised to sum to 1.0. For new users, only cold-start-capable algorithms are active.

## Evaluation Metrics

Evaluation is opt-in (click "Run Evaluation" after generating recommendations). Metrics are computed via leave-one-out splitting on a random sample of 500 test users for fast turnaround. Progress is logged to the terminal every 100 users.

### Ranking Metrics
| Metric | Description |
|--------|-------------|
| RMSE | Root mean squared error on held-out test interactions |
| Precision@K | Fraction of top-K that are relevant |
| Recall@K | Fraction of relevant items found in top-K |
| MAP@K | Mean Average Precision -- rewards relevant items appearing earlier |
| MRR@K | Mean Reciprocal Rank -- 1/rank of the first relevant item |
| Hit Rate@K | Fraction of users with at least one hit in top-K |
| nDCG@K | Normalised Discounted Cumulative Gain |

### Business Metrics
| Metric | Description |
|--------|-------------|
| ROI@K | Average profitability of recommended assets |
| Coverage@K | Fraction of the catalogue appearing in any user's recommendations |
| Diversity@K | Average pairwise cosine distance among recommended items |
| Novelty@K | Average self-information (less popular = more novel) |

## Configuration

All settings are in `config.py` via Pydantic models:

```python
from config import AppConfig

cfg = AppConfig()
cfg.model.svd_components    # 5         — SVD latent factors
cfg.model.knn_neighbors     # 20        — K for item-item KNN
cfg.model.default_weights   # [0.25, 0.25, 0.2, 0.15, 0.15]
cfg.data.data_dir           # FAR-Trans-Data/
cfg.cache.ttl_seconds       # 3600      — disk cache TTL
cfg.top_n                   # 10        — default recommendation count
```

## Caching Strategy

| Layer | Mechanism | Scope |
|-------|-----------|-------|
| Streamlit in-memory | `@st.cache_data` / `@st.cache_resource` | Data loading, model fitting, evaluation results |
| Disk persistence | `CacheManager` (joblib + TTL) | Fitted models that survive app restarts |

## Citation

> Javier Sanz-Cruzado, Nikolaos Droukas, Richard McCreadie. *FAR-Trans: An Investment Dataset for Financial Asset Recommendation.* IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys). Jeju, South Korea, August 2024.
