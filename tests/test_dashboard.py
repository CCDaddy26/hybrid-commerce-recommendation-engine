"""Tests for Streamlit dashboard modules, data loading, and model initialization."""

import pytest
import pandas as pd
from streamlit_app import load_all_data, initialize_models


def test_dashboard_data_loading():
    """Verify that dashboard loads interactions, metadata, and benchmark results cleanly."""
    interactions_df, metadata_df, benchmark_df = load_all_data()

    assert isinstance(interactions_df, pd.DataFrame)
    assert not interactions_df.empty
    assert set(interactions_df.columns).issuperset({"user_id", "item_id", "weight"})

    assert isinstance(metadata_df, pd.DataFrame)
    assert not metadata_df.empty
    assert set(metadata_df.columns).issuperset({"item_id", "title", "category", "tags"})

    assert isinstance(benchmark_df, pd.DataFrame)
    assert not benchmark_df.empty
    assert "ndcg@2" in benchmark_df.columns


def test_dashboard_model_initialization():
    """Verify that initialize_models fits all components and creates a valid HybridRecommender."""
    interactions_df, metadata_df, _ = load_all_data()
    models = initialize_models(interactions_df, metadata_df)

    assert "popularity" in models and models["popularity"].is_fitted
    assert "als" in models and models["als"].is_fitted
    assert "content" in models and models["content"].is_fitted
    assert "hybrid" in models

    # Test generating recommendations for U101
    hybrid = models["hybrid"]
    recs = hybrid.recommend(user_id="U101", user_history=["PROD_001"], n=2, return_breakdown=True)
    assert len(recs) <= 2
    assert "PROD_001" not in [r.item_id for r in recs]


def test_dashboard_cold_start_simulation():
    """Verify dashboard cold-start preference query logic."""
    interactions_df, metadata_df, _ = load_all_data()
    models = initialize_models(interactions_df, metadata_df)
    hybrid = models["hybrid"]

    # Cold start with preference
    recs_pref = hybrid.recommend(
        user_id="NEW_USER",
        user_history=[],
        preference_text="ergonomic vertical mouse",
        n=2,
        return_breakdown=True,
    )
    assert len(recs_pref) > 0
    rec_item_ids = [r.item_id for r in recs_pref]
    assert "PROD_001" in rec_item_ids or "PROD_006" in rec_item_ids

    # Cold start without preference falls back cleanly to popularity
    recs_fallback = hybrid.recommend(
        user_id="NEW_USER_NO_PREF",
        user_history=[],
        preference_text=None,
        n=2,
        return_breakdown=True,
    )
    assert len(recs_fallback) > 0
    # Top popular item in sample data is PROD_002
    assert recs_fallback[0].item_id == "PROD_002"

