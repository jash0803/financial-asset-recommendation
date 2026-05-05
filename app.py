import hashlib
import logging
import uuid

import streamlit as st
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

from config import AppConfig
from data.loader import load_data
from data.preprocessing import preprocess_data, build_rating_matrix, compute_momentum
from evaluation.splitters import leave_one_out_split
from evaluation.ranking import evaluate_ranking_metrics, compute_rmse
from evaluation.business import evaluate_business_metrics
from questionnaire.questions import QUESTIONS, OPTIONS
from questionnaire.processor import process_questionnaire_responses, update_customer_profile
from recommenders import (
    CollaborativeRecommender,
    ContentBasedRecommender,
    DemographicRecommender,
    PopularityRecommender,
    KNNRecommender,
    HybridRecommender,
)

cfg = AppConfig()

# Indices into the hybrid's sub_recommenders list
_HISTORY_DEPENDENT = {0, 1, 4}  # CF, Content-Based, KNN
_COLD_START = {2, 3}            # Demographic, Popularity


# ── Cached helpers ───────────────────────────────────────────────


@st.cache_data(show_spinner="Loading dataset…")
def _load_data(_cfg_hash: str):
    return load_data(cfg.data)


@st.cache_resource(show_spinner="Fitting recommendation models…")
def _fit_hybrid(_cfg_hash: str, svd_components: int, knn_neighbors: int):
    bundle = load_data(cfg.data)
    buys = preprocess_data(bundle.transactions)
    train_df, test_df = leave_one_out_split(buys)
    rating_matrix, rating_df = build_rating_matrix(train_df)
    momentum_df = compute_momentum(bundle.close_prices)

    cf = CollaborativeRecommender(n_components=svd_components)
    cb = ContentBasedRecommender()
    demo = DemographicRecommender()
    pop = PopularityRecommender()
    knn = KNNRecommender(k=knn_neighbors)

    default_w = cfg.model.default_weights
    hybrid = HybridRecommender([
        (cf, default_w[0]),
        (cb, default_w[1]),
        (demo, default_w[2]),
        (pop, default_w[3]),
        (knn, default_w[4]),
    ])

    hybrid.fit(
        train_df,
        rating_matrix=rating_matrix,
        rating_df=rating_df,
        asset_df=bundle.assets,
        customer_df=bundle.customers,
        limit_prices_df=bundle.limit_prices,
        momentum_df=momentum_df,
    )

    return hybrid, rating_matrix, rating_df, train_df, test_df, bundle, momentum_df


@st.cache_data(show_spinner="Evaluating ranking metrics (sampled)…")
def _cached_ranking_eval(_weights_hash: str, _k: int, _test_hash: str):
    """Cached wrapper — recomputes only when weights or K change."""
    _, rating_matrix, rating_df, _, test_df, _, _ = _fit_hybrid(
        hashlib.md5(cfg.model_dump_json().encode()).hexdigest(),
        cfg.model.svd_components,
        cfg.model.knn_neighbors,
    )
    hybrid = st.session_state["_eval_hybrid"]
    return evaluate_ranking_metrics(hybrid, test_df, rating_matrix, rating_df, k=_k)


@st.cache_data(show_spinner="Evaluating business metrics (sampled)…")
def _cached_business_eval(_weights_hash: str, _k: int, _test_hash: str):
    """Cached wrapper — recomputes only when weights or K change."""
    _, rating_matrix, rating_df, train_df, test_df, bundle, _ = _fit_hybrid(
        hashlib.md5(cfg.model_dump_json().encode()).hexdigest(),
        cfg.model.svd_components,
        cfg.model.knn_neighbors,
    )
    hybrid = st.session_state["_eval_hybrid"]
    return evaluate_business_metrics(
        hybrid, test_df, rating_matrix, rating_df,
        bundle.assets, bundle.limit_prices, train_df, k=_k,
    )


# ── Recommendation display helper ────────────────────────────────


def _build_rec_table(recs, bundle, momentum_df):
    asset_lookup = bundle.assets.set_index("ISIN")
    lp_lookup = bundle.limit_prices.set_index("ISIN")
    market_lookup = bundle.markets.set_index("marketID")

    rows = []
    for isin in recs.index:
        a = asset_lookup.loc[isin] if isin in asset_lookup.index else None
        lp = lp_lookup.loc[isin] if isin in lp_lookup.index else None
        mom = momentum_df.loc[isin] if isin in momentum_df.index else None
        mid = a["marketID"] if a is not None else None
        mkt = market_lookup.loc[mid] if mid is not None and mid in market_lookup.index else None

        rows.append({
            "Asset Name": a["assetName"] if a is not None else "N/A",
            "Category": a["assetCategory"] if a is not None else "N/A",
            "Subcategory": a["assetSubCategory"] if a is not None else "N/A",
            "Market": mkt["name"] if mkt is not None else "N/A",
            "Country": mkt["country"] if mkt is not None else "N/A",
            "Sector": a["sector"] if a is not None else "N/A",
            "Profitability": lp["profitability"] if lp is not None else None,
            "30d Return": mom["return_30d"] if mom is not None else None,
            "90d Return": mom["return_90d"] if mom is not None else None,
            "Latest Price": lp["priceMaxDate"] if lp is not None else None,
            "Score": recs[isin],
        })

    df = pd.DataFrame(rows, index=recs.index)
    df.index.name = "ISIN"
    return df


# ── Main ─────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="FAR-Trans Asset Recommender", layout="wide")

    cfg_hash = hashlib.md5(cfg.model_dump_json().encode()).hexdigest()
    bundle = _load_data(cfg_hash)
    hybrid, rating_matrix, rating_df, train_df, test_df, bundle, momentum_df = (
        _fit_hybrid(cfg_hash, cfg.model.svd_components, cfg.model.knn_neighbors)
    )

    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.title("Settings")

    is_new_user = st.sidebar.toggle("I'm a new user", value=False)

    if is_new_user:
        new_id = st.sidebar.text_input(
            "Choose a User ID",
            value=f"NEW_{uuid.uuid4().hex[:8].upper()}",
        )
        customer_id = new_id
    else:
        customer_list = list(rating_matrix.index)
        customer_id = st.sidebar.selectbox("Customer ID", customer_list)

    N = st.sidebar.slider("Top N Recommendations", min_value=1, max_value=20, value=cfg.top_n)

    # Weight sliders
    st.sidebar.markdown("---")
    st.sidebar.subheader("Component Weights")

    weight_sliders: list[float] = []
    for idx, (rec, default_w) in enumerate(hybrid.sub_recommenders):
        if is_new_user and idx in _HISTORY_DEPENDENT:
            weight_sliders.append(0.0)
            st.sidebar.slider(
                rec.name,
                min_value=0.0, max_value=1.0,
                value=0.0, step=0.05,
                key=f"w_{rec.name}",
                disabled=True,
                help="Disabled for new users (no purchase history)",
            )
        else:
            default = 0.5 if (is_new_user and idx in _COLD_START) else float(default_w)
            w = st.sidebar.slider(
                rec.name,
                min_value=0.0, max_value=1.0,
                value=default, step=0.05,
                key=f"w_{rec.name}",
            )
            weight_sliders.append(w)

    # Normalize weights to sum to 1
    total = sum(weight_sliders)
    if total > 0:
        normalized = [w / total for w in weight_sliders]
    else:
        normalized = [1.0 / len(weight_sliders)] * len(weight_sliders)
    hybrid.update_weights(normalized)

    st.sidebar.caption(
        "Weights: " + " / ".join(f"{w:.0%}" for w in normalized)
    )

    # ── Main area ─────────────────────────────────────────────
    st.title("FAR-Trans Asset Recommender")
    st.caption(
        "Hybrid recommendation system combining collaborative filtering, "
        "content-based filtering, demographic matching, popularity baseline, "
        "and item-item KNN -- powered by the FAR-Trans dataset. "
        "Created by [Jash Shah](https://www.linkedin.com/in/jashshah0803/)."
    )

    # Questionnaire — shown in main area ONLY for new users
    if is_new_user:
        st.info(
            "**New user mode** -- complete the questionnaire below to create "
            "your risk profile, then generate recommendations based on your "
            "profile and overall asset popularity."
        )

        st.subheader("Risk Assessment Questionnaire")

        if "questionnaire_responses" not in st.session_state:
            st.session_state.questionnaire_responses = {}

        for q_id, question_text in QUESTIONS.items():
            opts = OPTIONS[q_id]
            response = st.radio(
                question_text,
                options=list(opts.keys()),
                format_func=lambda x, _opts=opts: _opts[x],
                key=f"risk_{q_id}",
            )
            st.session_state.questionnaire_responses[q_id] = response

        if st.button("Create Profile & Continue"):
            risk_level, inv_cap = process_questionnaire_responses(
                st.session_state.questionnaire_responses
            )
            bundle.customers = update_customer_profile(
                customer_id, risk_level, inv_cap, bundle.customers
            )
            st.success(f"Profile created! Risk: **{risk_level}** | Capacity: **{inv_cap}**")

        st.markdown("---")

    # Generate recommendations
    if st.button("Generate Recommendations", type="primary", width="stretch"):
        st.session_state["run_recs"] = True

    if st.session_state.get("run_recs"):
        st.markdown("---")
        st.subheader(f"Top {N} Recommendations for `{customer_id}`")

        recs = hybrid.recommend(customer_id, rating_df, n=int(N))
        rec_details = _build_rec_table(recs, bundle, momentum_df)

        st.dataframe(
            rec_details.style.format({
                "Score": "{:.4f}",
                "Profitability": "{:.2%}",
                "30d Return": "{:.2%}",
                "90d Return": "{:.2%}",
                "Latest Price": "€{:.2f}",
            }, na_rep="--"),
            width="stretch",
        )

        # ── Evaluation Metrics (opt-in) ───────────────────────
        if not is_new_user:
            st.markdown("---")
            st.subheader("Evaluation Metrics")
            st.caption(
                "Leave-one-out evaluation on a random sample of 500 test users. "
                "Click the button below to compute."
            )

            if st.button("Run Evaluation", key="eval_btn"):
                st.session_state["run_eval"] = True

            if st.session_state.get("run_eval"):
                weights_hash = hashlib.md5(
                    str(weight_sliders).encode()
                ).hexdigest()
                test_hash = hashlib.md5(
                    str(len(test_df)).encode()
                ).hexdigest()

                st.session_state["_eval_hybrid"] = hybrid

                with st.spinner("Computing metrics (sampled, ~500 users)…"):
                    ranking = _cached_ranking_eval(weights_hash, int(N), test_hash)
                    business = _cached_business_eval(weights_hash, int(N), test_hash)
                    cf_rec = hybrid.sub_recommenders[0][0]
                    rmse = compute_rmse(cf_rec.pred_df, test_df)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Ranking Metrics**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RMSE", f"{rmse:.4f}" if rmse is not None else "N/A")
                    for i, (name, val) in enumerate(ranking.items()):
                        target = [m1, m2, m3][(i + 1) % 3]
                        target.metric(name, f"{val:.4f}" if val is not None else "N/A")

                with col2:
                    st.markdown("**Business Metrics**")
                    b1, b2 = st.columns(2)
                    for i, (name, val) in enumerate(business.items()):
                        target = b1 if i % 2 == 0 else b2
                        if val is not None:
                            fmt = f"{val:.2%}" if name == "ROI@K" else f"{val:.4f}"
                            target.metric(name, fmt)
                        else:
                            target.metric(name, "N/A")

    # ── Dataset Explorer ──────────────────────────────────────
    st.markdown("---")
    with st.expander("Dataset Explorer", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Assets", len(bundle.assets))
        d2.metric("Customers", bundle.customers["customerID"].nunique())
        d3.metric("Transactions", len(bundle.transactions))
        d4.metric("Markets", len(bundle.markets))

        st.markdown("**Assets by Category**")
        cat_counts = bundle.assets["assetCategory"].value_counts()
        st.bar_chart(cat_counts)

        st.markdown("**Transaction Volume by Channel**")
        ch_counts = bundle.transactions["channel"].value_counts()
        st.bar_chart(ch_counts)

        st.markdown("**Markets**")
        st.dataframe(
            bundle.markets[["marketID", "name", "country", "tradingDays", "tradingHours"]],
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
