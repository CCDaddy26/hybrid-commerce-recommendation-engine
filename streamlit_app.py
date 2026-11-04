"""Interactive Recommendation Engine Dashboard.

Provides interactive exploration of the Hybrid Commerce Recommendation Engine:
- Customer Recommendations across Popularity, ALS, Content, and Hybrid models.
- New User Cold-Start query simulation.
- Product Discovery & Similar Items.
- Benchmark Metrics comparison with visualizations.
- Architecture overview & configuration inspection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from app.data_loader import load_interactions
from app.item_metadata import load_item_metadata
from app.preprocessor import preprocess_interactions
from app.matrix_builder import build_user_item_matrix
from app.popularity import PopularityRecommender
from app.als_recommender import ImplicitALSRecommender
from app.content_recommender import ContentRecommender
from app.hybrid_recommender import HybridRecommender, HybridRecommendation

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & ENTERPRISE STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.8;
    }
    .rec-card {
        border-left: 4px solid #1E88E5;
        background-color: var(--secondary-background-color);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .rec-card-alt {
        border-left: 4px solid #43A047;
        background-color: var(--secondary-background-color);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .rec-title {
        font-size: 15px;
        font-weight: 600;
    }
    .rec-meta {
        font-size: 12px;
        opacity: 0.75;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(30, 136, 229, 0.15);
        color: #1E88E5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -----------------------------------------------------------------------------
# CACHED DATA & MODEL INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_data():
    """Load interactions, metadata, and benchmark results safely."""
    interactions_df = load_interactions(DATA_DIR / "sample_interactions.csv")
    metadata_df = load_item_metadata(DATA_DIR / "item_metadata.csv")

    benchmark_path = DATA_DIR / "benchmark_results.csv"
    if benchmark_path.is_file():
        benchmark_df = pd.read_csv(benchmark_path, index_col=0)
    else:
        benchmark_df = pd.DataFrame()

    return interactions_df, metadata_df, benchmark_df


@st.cache_resource
def initialize_models(
    interactions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    als_weight: float = 0.50,
    content_weight: float = 0.35,
    popularity_weight: float = 0.15,
):
    """Fit all recommendation components and wrap in HybridRecommender."""
    prep_data = preprocess_interactions(interactions_df)
    matrix = build_user_item_matrix(prep_data)

    popularity = PopularityRecommender().fit(prep_data.aggregated_df)

    als = ImplicitALSRecommender(
        factors=8,
        regularization=0.05,
        iterations=15,
        random_state=42,
    ).fit(matrix, prep_data)

    content = ContentRecommender().fit(metadata_df)

    weights = {
        "als": als_weight,
        "content": content_weight,
        "popularity": popularity_weight,
    }

    hybrid = HybridRecommender(
        als_recommender=als,
        content_recommender=content,
        popularity_recommender=popularity,
        weights=weights,
    )

    return {
        "prep_data": prep_data,
        "matrix": matrix,
        "popularity": popularity,
        "als": als,
        "content": content,
        "hybrid": hybrid,
    }


# Load data and models
try:
    interactions_df, metadata_df, benchmark_df = load_all_data()
    models = initialize_models(interactions_df, metadata_df)
except Exception as e:
    st.error(f"Failed to load application data: {str(e)}")
    st.stop()

# Lookup table for metadata
meta_lookup = metadata_df.set_index("item_id").to_dict(orient="index")

def get_item_title(item_id: str) -> str:
    return meta_lookup.get(item_id, {}).get("title", str(item_id))

def get_item_category(item_id: str) -> str:
    return meta_lookup.get(item_id, {}).get("category", "General")

def get_item_tags(item_id: str) -> str:
    return meta_lookup.get(item_id, {}).get("tags", "")


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.title("🛍️ Commerce Engine")
st.sidebar.caption("Project 02: Hybrid Recommendation Platform")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "System Overview",
        "Customer Recommendations",
        "New User Cold-Start",
        "Product Discovery",
        "Algorithm Comparison",
        "Technical Architecture",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Ensemble Weights")
w_als = st.sidebar.slider("ALS Collaborative", 0.0, 1.0, 0.50, 0.05)
w_cnt = st.sidebar.slider("Content TF-IDF", 0.0, 1.0, 0.35, 0.05)
w_pop = st.sidebar.slider("Popularity", 0.0, 1.0, 0.15, 0.05)

# Re-normalize if updated by user
total_w = w_als + w_cnt + w_pop
if total_w > 0:
    models["hybrid"].weights = {
        "als": w_als / total_w,
        "content": w_cnt / total_w,
        "popularity": w_pop / total_w,
    }

st.sidebar.caption(
    f"Normalized: ALS={models['hybrid'].weights['als']:.2f}, "
    f"Content={models['hybrid'].weights['content']:.2f}, "
    f"Pop={models['hybrid'].weights['popularity']:.2f}"
)


# -----------------------------------------------------------------------------
# SECTION A: SYSTEM OVERVIEW
# -----------------------------------------------------------------------------
if nav_choice == "System Overview":
    st.title("🛍️ Hybrid Commerce Recommendation Engine")
    st.markdown(
        "A multi-strategy recommendation platform combining **Collaborative Filtering (Implicit ALS)**, "
        "**Content-Based Similarity (TF-IDF)**, and a **Popularity Fallback** with min-max score normalization "
        "and dynamic weight redistribution."
    )

    st.warning(
        "⚠️ **Dataset Disclosure & Limitations**: This prototype runs on a verified 12-interaction sample dataset "
        "with 5 customers (`U101`–`U105`) and 6 products (`PROD_001`–`PROD_006`). No metrics or outputs are fabricated. "
        "Real-world enterprise deployments require hundreds of thousands of interactions.",
        icon="ℹ️",
    )

    st.subheader("Platform Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Active Customers</div>
                <div class="metric-val">{interactions_df['user_id'].nunique()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Catalog Products</div>
                <div class="metric-val">{len(metadata_df)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Interaction Events</div>
                <div class="metric-val">{len(interactions_df)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Models Ensembled</div>
                <div class="metric-val">3</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("System Architecture")
    st.markdown(
        """
        ```text
        [ Customer Request ] ---> [ Check User History & Query Context ]
                                                |
                 +------------------------------+------------------------------+
                 |                              |                              |
        [ Implicit ALS ]             [ Content Recommender ]       [ Popularity Fallback ]
        Matrix Factorization          TF-IDF + Cosine Similarity    Frequency Ranking
        (Interacted users only)       (Seeds / cold-start items)    (Global baseline)
                 |                              |                              |
                 +------------------------------+------------------------------+
                                                |
                                                v
                              [ Min-Max Score Normalization ]
                              - Maps raw scores into [0.0, 1.0]
                              - Dynamic weight redistribution on empty models
                                                |
                                                v
                              [ Candidate Aggregation & Exclusion ]
                              - Strictly excludes previously interacted items
                              - Deterministic tie-breaking (-score, str(item_id))
                                                |
                                                v
                                  Top-N Recommendations
        ```
        """
    )


# -----------------------------------------------------------------------------
# SECTION B: CUSTOMER RECOMMENDATIONS
# -----------------------------------------------------------------------------
elif nav_choice == "Customer Recommendations":
    st.title("👤 Customer Recommendations")
    st.markdown("Explore personalized recommendations generated across individual and ensembled recommendation models.")

    customers = sorted(interactions_df["user_id"].unique())
    selected_customer = st.selectbox("Select Customer ID:", customers, index=0)
    top_n = st.slider("Top N Recommendations:", min_value=1, max_value=5, value=3)

    user_history_df = interactions_df[interactions_df["user_id"] == selected_customer]
    interacted_item_ids = list(user_history_df["item_id"].unique())

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Interaction History")
        st.caption(f"Historical events recorded for {selected_customer}:")

        for _, row in user_history_df.iterrows():
            item_id = row["item_id"]
            title = get_item_title(item_id)
            cat = get_item_category(item_id)
            event = row["event_type"].upper()
            w = row["weight"]

            st.markdown(
                f"""
                <div class="rec-card-alt">
                    <div class="rec-title">{title} <span class="badge">{item_id}</span></div>
                    <div class="rec-meta">Category: {cat} | Event: <b>{event}</b> (Weight: {w})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info("💡 Interacted products above are automatically excluded from personalized recommendation sets.")

    with col_right:
        st.subheader("Model Comparison")

        tab_hybrid, tab_als, tab_content, tab_popularity = st.tabs(
            ["⭐ Hybrid Ensemble", "🤖 Implicit ALS", "📄 Content-Based", "🔥 Popularity"]
        )

        with tab_hybrid:
            st.markdown("##### Weighted Ensemble (ALS + Content + Popularity)")
            hybrid_recs = models["hybrid"].recommend(
                user_id=selected_customer,
                user_history=interacted_item_ids,
                n=top_n,
                return_breakdown=True,
            )

            if not hybrid_recs:
                st.write("No unseen recommendations available for this customer.")
            else:
                for rank, rec in enumerate(hybrid_recs, start=1):
                    title = get_item_title(rec.item_id)
                    cat = get_item_category(rec.item_id)
                    breakdown_str = " | ".join(
                        [f"{k.capitalize()}: {v:.3f}" for k, v in rec.breakdown.items()]
                    )
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div class="rec-title">#{rank} {title} <span class="badge">{rec.item_id}</span></div>
                            <div class="rec-meta">Category: {cat} &nbsp;|&nbsp; <b>Ensembled Score: {rec.score:.4f}</b></div>
                            <div class="rec-meta" style="margin-top: 4px; color: #1E88E5;">Contributions &rarr; {breakdown_str}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_als:
            st.markdown("##### Collaborative Filtering (Implicit ALS)")
            als_recs = models["als"].recommend(
                user_id=selected_customer,
                n=top_n,
                filter_already_liked_items=True,
            )
            if not als_recs:
                st.write("No ALS recommendations generated.")
            else:
                for rank, (item_id, score) in enumerate(als_recs, start=1):
                    title = get_item_title(item_id)
                    cat = get_item_category(item_id)
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div class="rec-title">#{rank} {title} <span class="badge">{item_id}</span></div>
                            <div class="rec-meta">Category: {cat} &nbsp;|&nbsp; Dot Product Score: <b>{score:.4f}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_content:
            st.markdown("##### Content-Based (TF-IDF Similarity to History)")
            content_recs = models["content"].recommend_for_user(
                user_history_items=interacted_item_ids,
                n=top_n,
            )
            if not content_recs:
                st.write("No content recommendations generated.")
            else:
                for rank, (item_id, score) in enumerate(content_recs, start=1):
                    title = get_item_title(item_id)
                    cat = get_item_category(item_id)
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div class="rec-title">#{rank} {title} <span class="badge">{item_id}</span></div>
                            <div class="rec-meta">Category: {cat} &nbsp;|&nbsp; Cosine Similarity: <b>{score:.4f}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_popularity:
            st.markdown("##### Popularity Fallback (Filtered by History)")
            all_pop = models["popularity"].recommend(n=len(metadata_df))
            pop_recs = [
                (i, s) for i, s in all_pop if i not in interacted_item_ids
            ][:top_n]
            if not pop_recs:
                st.write("No popularity items remaining.")
            else:
                for rank, (item_id, score) in enumerate(pop_recs, start=1):
                    title = get_item_title(item_id)
                    cat = get_item_category(item_id)
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <div class="rec-title">#{rank} {title} <span class="badge">{item_id}</span></div>
                            <div class="rec-meta">Category: {cat} &nbsp;|&nbsp; Aggregated Weight: <b>{score:.1f}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# -----------------------------------------------------------------------------
# SECTION C: NEW USER COLD START
# -----------------------------------------------------------------------------
elif nav_choice == "New User Cold-Start":
    st.title("❄️ New User Cold-Start Simulator")
    st.markdown(
        "Demonstrates cold-start handling when a user has zero interaction history. "
        "If preference text is provided, the engine leverages **TF-IDF content similarity** "
        "supplemented with popularity fallback. If no preference text is entered, it falls back purely to **Popularity**."
    )

    pref_input = st.text_input(
        "Customer Preference Query (or leave blank to test fallback):",
        value="wireless ergonomic computer mouse",
        placeholder="e.g. noise cancelling headphones, mechanical keyboard, wireless mouse",
    )

    n_cold = st.slider("Number of Recommendations:", min_value=1, max_value=5, value=3)

    if st.button("Generate Cold-Start Recommendations", type="primary"):
        with st.spinner("Processing cold-start recommendation pipeline..."):
            recs = models["hybrid"].recommend(
                user_id="NEW_GUEST_USER",
                user_history=[],
                preference_text=pref_input if pref_input.strip() else None,
                n=n_cold,
                return_breakdown=True,
            )

        if not recs:
            st.info("No recommendations returned.")
        else:
            if pref_input.strip():
                st.success(f"Matched against query: *\"{pref_input.strip()}\"* (Content + Popularity)")
            else:
                st.info("No query entered: Engine engaged pure Popularity Fallback.")

            for rank, rec in enumerate(recs, start=1):
                title = get_item_title(rec.item_id)
                cat = get_item_category(rec.item_id)
                tags = get_item_tags(rec.item_id)
                breakdown_str = " | ".join(
                    [f"{k.capitalize()}: {v:.3f}" for k, v in rec.breakdown.items()]
                )

                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div class="rec-title">#{rank} {title} <span class="badge">{rec.item_id}</span></div>
                        <div class="rec-meta">Category: <b>{cat}</b> &nbsp;|&nbsp; Ensembled Score: <b>{rec.score:.4f}</b></div>
                        <div class="rec-meta">Tags: <i>{tags}</i></div>
                        <div class="rec-meta" style="margin-top: 4px; color: #1E88E5;">Contribution Breakdown &rarr; {breakdown_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# -----------------------------------------------------------------------------
# SECTION D: PRODUCT DISCOVERY (CONTENT-BASED)
# -----------------------------------------------------------------------------
elif nav_choice == "Product Discovery":
    st.title("🔍 Product Discovery & Similar Items")
    st.markdown(
        "Select a seed product to explore lexical and semantic similarity computed via TF-IDF over titles, categories, and tags."
    )

    all_product_ids = list(metadata_df["item_id"])
    product_options = {pid: f"{pid} — {get_item_title(pid)}" for pid in all_product_ids}

    selected_pid = st.selectbox(
        "Choose a Seed Product:",
        all_product_ids,
        format_func=lambda x: product_options[x],
        index=0,
    )

    st.markdown(
        f"""
        **Selected Seed Product**: `{selected_pid}` — **{get_item_title(selected_pid)}**  
        *Category*: {get_item_category(selected_pid)} | *Tags*: {get_item_tags(selected_pid)}
        """
    )

    if selected_pid == "PROD_006":
        st.info(
            "🌟 **Note on PROD_006**: This is a dedicated cold-start item with zero interaction history in the interactions CSV. "
            "Content similarity allows it to be discovered immediately based on product metadata."
        )

    n_sim = st.slider("Number of Similar Products:", min_value=1, max_value=5, value=3)

    similar_items = models["content"].recommend_similar_items(
        seed_item_ids=selected_pid,
        n=n_sim,
    )

    st.subheader(f"Products Most Similar to {get_item_title(selected_pid)}")

    if not similar_items:
        st.write("No similar products found.")
    else:
        for rank, (item_id, score) in enumerate(similar_items, start=1):
            title = get_item_title(item_id)
            cat = get_item_category(item_id)
            tags = get_item_tags(item_id)

            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-title">#{rank} {title} <span class="badge">{item_id}</span></div>
                    <div class="rec-meta">Category: <b>{cat}</b> &nbsp;|&nbsp; Cosine Similarity: <b>{score:.4f}</b></div>
                    <div class="rec-meta">Tags: <i>{tags}</i></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# SECTION E: ALGORITHM COMPARISON & BENCHMARKS
# -----------------------------------------------------------------------------
elif nav_choice == "Algorithm Comparison":
    st.title("📊 Empirical Benchmark Comparison")
    st.markdown(
        "Standardized offline evaluation across all models on held-out test data (Cutoff $K=2$, zero-leakage split). "
        "Metrics are loaded directly from the verified benchmark pipeline."
    )

    if benchmark_df.empty:
        st.warning("Benchmark results file not found at `data/benchmark_results.csv`.")
    else:
        st.subheader("Performance Summary Table")
        st.dataframe(
            benchmark_df.style.format(
                {
                    "precision@2": "{:.3f}",
                    "recall@2": "{:.3f}",
                    "map@2": "{:.3f}",
                    "ndcg@2": "{:.4f}",
                    "hit_rate@2": "{:.3f}",
                    "catalog_coverage@2": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Visual Comparisons")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("##### Ranking Performance (NDCG@2 & MAP@2)")
            chart_data_ranking = benchmark_df[["ndcg@2", "map@2"]]
            st.bar_chart(chart_data_ranking)

        with col_c2:
            st.markdown("##### Catalog Coverage (Diversity / Exploration)")
            chart_data_cov = benchmark_df[["catalog_coverage@2"]]
            st.bar_chart(chart_data_cov)

        st.caption(
            "Note: In small catalogs with concentrated weights, Popularity and ALS prioritize repeat-frequent items, "
            "whereas Content-Based models maximize catalog coverage (83.3%) by exploring lexical similarities."
        )


# -----------------------------------------------------------------------------
# SECTION F: TECHNICAL INFORMATION & ARCHITECTURE
# -----------------------------------------------------------------------------
elif nav_choice == "Technical Information":
    st.title("⚙️ Technical Architecture & Configuration")
    st.markdown("Detailed breakdown of algorithms, mathematical formulations, and runtime configurations.")

    with st.expander("1. Collaborative Filtering (Implicit ALS)", expanded=True):
        st.markdown(
            r"""
            - **Algorithm**: Alternating Least Squares with implicit feedback (`implicit.als.AlternatingLeastSquares`).
            - **Confidence Formulation**: $C_{ui} = 1 + \alpha r_{ui}$ where $\alpha = 1.0$.
            - **Objective**: Minimizes weighted squared error over binary preferences $p_{ui} \in \{0, 1\}$.
            - **Configuration**: Latent factors: 8 | Regularization: 0.05 | Iterations: 15 | Random state: 42.
            """
        )

    with st.expander("2. Content-Based Filtering (TF-IDF)", expanded=True):
        st.markdown(
            r"""
            - **Feature Extraction**: `TfidfVectorizer(stop_words='english')`.
            - **Text Corpus**: Concatenation of item `title`, `category`, and `tags`.
            - **Similarity Measure**: Cosine similarity between query/seed TF-IDF vector and catalog representations:
              $$\text{Sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|}$$
            """
        )

    with st.expander("3. Score Normalization & Weight Ensembling", expanded=True):
        st.markdown(
            r"""
            - **Score Normalization**: Min-max normalization per algorithm to scale raw scores into $[0.0, 1.0]$:
              $$s_{\text{norm}} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$
            - **Current Active Weights**:
            """
        )
        st.json(models["hybrid"].weights)
        st.markdown(
            """
            - **Dynamic Redistribution**: If an algorithm produces no candidates (e.g. ALS for cold-start users),
              its configured weight is redistributed proportionally across remaining active models.
            """
        )

    with st.expander("4. Production Considerations & Scaling", expanded=False):
        st.markdown(
            """
            - **Threadpool Optimization**: Set `OPENBLAS_NUM_THREADS=1` to prevent CPU threadpool contention in multithreaded servers.
            - **Vector Indexing**: For catalogs $> 100,000$ items, replace dense pairwise cosine similarity with Approximate Nearest Neighbors (e.g., Faiss or HNSW).
            """
        )
