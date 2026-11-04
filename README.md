# Hybrid Commerce Recommendation Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3116/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end, multi-strategy commerce recommendation engine that unites **Collaborative Filtering (Implicit ALS)**, **Content-Based Filtering (TF-IDF & Cosine Similarity)**, and a **Popularity Fallback** into a unified, weighted ensemble with dynamic weight redistribution and cold-start support.

Features an interactive **Streamlit Analytics Dashboard** (`streamlit_app.py`) for real-time customer inspection, query-driven cold-start simulation, product discovery, and empirical offline benchmark visualizations.

---

## Architecture Overview

```text
               +-------------------------------------------------------+
               |                  Recommendation Request               |
               | (user_id, user_history, preference_text, seed_items)  |
               +-------------------------------------------------------+
                                           |
             +-----------------------------+-----------------------------+
             |                             |                             |
      [ALS Recommender]           [ContentRecommender]          [PopularityRecommender]
      Collaborative signal        Content similarity             Frequency / Pop fallback
             |                             |                             |
             v                             v                             v
      Raw Candidates &              Raw Candidates &              Raw Candidates &
          Scores                        Scores                        Scores
             |                             |                             |
             +-----------------------------+-----------------------------+
                                           |
                                           v
                             [Score Normalization (0.0 to 1.0)]
                             - Min-Max scaling per algorithm
                             - Finite sanitation (NaN/Inf purged)
                             - Equal/single score default to 1.0
                                           |
                                           v
                        [Dynamic Weight Redistribution]
                        - Baseline weights: ALS 0.50, Content 0.35, Pop 0.15
                        - Inactive / empty algorithms redistribute proportionally
                                           |
                                           v
                        [Candidate Aggregation & Exclusions]
                        - Remove interacted items & seeds
                        - Aggregate weighted normalized scores
                        - Deterministic tie-breaking (-score, str(item_id))
                                           |
                                           v
                        Top-N Hybrid Recommendations
```

### Recommendation Approaches
1. **Implicit ALS Collaborative Filtering (`app.als_recommender`)**:
   - Matrix factorization via `implicit.als.AlternatingLeastSquares`.
   - Confidence scaling: $C_{ui} = 1 + \alpha r_{ui}$ where $\alpha = 1.0$.
   - Personalizes for users with interaction history while strictly filtering previously consumed products.
2. **Content-Based Filtering (`app.content_recommender`)**:
   - `TfidfVectorizer` (with English stop words) and cosine similarity across `title`, `category`, and `tags`.
   - Powers item-to-item similarity and surfaces cold-start catalog items (like `PROD_006`) with zero interaction history.
3. **Popularity Fallback (`app.popularity`)**:
   - Global frequency scoring with deterministic secondary tie-breaking by item ID.
   - Provides a guaranteed fallback baseline for new visitors without history or preferences.
4. **Hybrid Ensemble & Normalization (`app.hybrid_recommender`)**:
   - Normalizes heterogeneous score distributions to $[0.0, 1.0]$ via min-max scaling.
   - Redistributes algorithm weights dynamically when specific models are inactive (e.g. ALS for cold-start users).

---

## Dataset Description & Limitations

> ⚠️ **Demo Dataset Disclosure**:
> This repository contains a small fictional dataset designed for architecture validation, algorithm prototyping, and unit testing:
> - **12 interaction events** across **5 customers** (`U101`–`U105`) and **5 products** (`PROD_001`–`PROD_005`).
> - **1 un-interacted product** (`PROD_006`: Ergonomic Vertical Mouse) specifically added to validate cold-start item discovery.
> - **No wall-clock timestamps**: Evaluated via stratified per-user Leave-K-Out splitting rather than temporal splits.
>
> **Production Scaling Note**: The small scale allows fast unit tests and reproducible verification. Real-world commerce deployments require tuning $\alpha$ and latent factors across millions of interactions.

---

## Offline Evaluation & Empirical Benchmarks

Evaluated under a standardized protocol ($K=2$, zero-leakage test split):

| Recommender | Precision@2 | Recall@2 | MAP@2 | NDCG@2 | Hit Rate@2 | Catalog Coverage@2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Fallback** | 0.100 | 0.200 | 0.100 | 0.126 | 0.200 | 50.0% |
| **Implicit ALS** | 0.100 | 0.200 | 0.100 | 0.126 | 0.200 | 66.7% |
| **Content-Based (TF-IDF)** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 83.3% |
| **Hybrid Ensemble** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 66.7% |

*Key Findings*:
- **Content Coverage**: Content-based filtering achieves the highest catalog exploration (83.3% coverage), successfully recommending diverse categories and un-interacted cold-start items.
- **Root Cause of 0.0 Test Scores**: Cross-category transitions in tiny catalogs (e.g. keyboard $\rightarrow$ headphones) share 0 lexical overlap in titles/tags; the evaluation report honestly documents these findings without inflating results.

---

## Repository Structure

```text
.
├── .gitignore
├── EVALUATION_REPORT.md             # In-depth statistical evaluation and benchmarks
├── README.md                        # Documentation and project manual
├── demo.py                          # CLI terminal demonstration script
├── requirements.txt                 # Python dependencies
├── streamlit_app.py                 # Interactive Streamlit dashboard application
├── app/
│   ├── __init__.py
│   ├── als_recommender.py           # Implicit ALS collaborative filtering model
│   ├── benchmark.py                 # Multi-model benchmarking & weight optimization
│   ├── content_recommender.py       # TF-IDF & Cosine similarity content recommender
│   ├── data_loader.py               # Robust CSV data loader and validator
│   ├── evaluator.py                 # Leakage-free train/val/test splitting
│   ├── hybrid_recommender.py        # Weighted hybrid ensemble & score normalization
│   ├── item_metadata.py             # Product metadata loader and validator
│   ├── matrix_builder.py            # User-Item sparse CSR matrix construction
│   ├── metrics.py                   # Ranking metrics (Precision, Recall, NDCG, MAP, Coverage)
│   ├── popularity.py                # Frequency baseline recommender
│   └── preprocessor.py              # Aggregation & 0-based contiguous mapping
├── data/
│   ├── benchmark_results.csv        # Pre-computed benchmark evaluation results
│   ├── item_metadata.csv            # Fictional product catalog metadata
│   └── sample_interactions.csv      # Fictional commerce user-item interaction log
└── tests/
    ├── __init__.py
    ├── test_content_recommender.py
    ├── test_dashboard.py            # Dashboard loading and interaction tests
    ├── test_data_loader.py
    ├── test_evaluation_and_benchmark.py
    ├── test_hybrid_recommender.py
    ├── test_matrix_and_als.py
    └── test_preprocessor_and_popularity.py
```

---

## Local Setup & Quickstart

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/<YOUR_USERNAME>/hybrid-commerce-recommendation-engine.git
cd hybrid-commerce-recommendation-engine

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Activate on macOS/Linux
# source .venv/bin/activate
```

### 2. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```
*(All 66 tests should pass with 0 failures).*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## Deployment to Streamlit Community Cloud

1. Push this repository to GitHub under your account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository: `<YOUR_USERNAME>/hybrid-commerce-recommendation-engine`.
5. Branch: `main`.
6. Main file path: `streamlit_app.py`.
7. Click **"Deploy!"**.
