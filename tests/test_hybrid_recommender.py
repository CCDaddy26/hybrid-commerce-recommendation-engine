"""Tests for HybridRecommender and score normalization."""

import math
import pytest
import pandas as pd
import numpy as np

from app.data_loader import load_interactions
from app.preprocessor import preprocess_interactions
from app.matrix_builder import build_user_item_matrix
from app.als_recommender import ImplicitALSRecommender
from app.popularity import PopularityRecommender
from app.item_metadata import load_item_metadata
from app.content_recommender import ContentRecommender
from app.hybrid_recommender import HybridRecommender, HybridRecommendation, normalize_scores


# ---------------------------------------------------------------------------
# Unit tests for Score Normalization
# ---------------------------------------------------------------------------

def test_normalize_scores_standard():
    """Verify min-max normalization scales raw scores between 0.0 and 1.0."""
    raw = [("item_1", 10.0), ("item_2", 20.0), ("item_3", 30.0)]
    norm = normalize_scores(raw)
    assert norm == [("item_1", 0.0), ("item_2", 0.5), ("item_3", 1.0)]


def test_normalize_scores_equal_values():
    """Verify equal scores are normalized to 1.0 (indicating top relevance)."""
    raw = [("item_1", 5.0), ("item_2", 5.0)]
    norm = normalize_scores(raw)
    assert norm == [("item_1", 1.0), ("item_2", 1.0)]


def test_normalize_scores_single_item():
    """Verify single-item list normalizes to 1.0."""
    raw = [("item_1", 42.0)]
    norm = normalize_scores(raw)
    assert norm == [("item_1", 1.0)]


def test_normalize_scores_negative_values():
    """Verify negative scores are normalized properly to [0.0, 1.0]."""
    raw = [("item_1", -10.0), ("item_2", 0.0), ("item_3", 10.0)]
    norm = normalize_scores(raw)
    assert norm == [("item_1", 0.0), ("item_2", 0.5), ("item_3", 1.0)]


def test_normalize_scores_nan_inf_handling():
    """Verify NaN and infinity scores are discarded without breaking normalization."""
    raw = [
        ("item_1", 1.0),
        ("item_bad_nan", float("nan")),
        ("item_bad_inf", float("inf")),
        ("item_bad_neginf", float("-inf")),
        ("item_2", 3.0),
    ]
    norm = normalize_scores(raw)
    assert len(norm) == 2
    assert norm == [("item_1", 0.0), ("item_2", 1.0)]


def test_normalize_scores_empty():
    """Verify empty input returns empty list."""
    assert normalize_scores([]) == []


# ---------------------------------------------------------------------------
# Unit tests for HybridRecommender Weights & Initialization
# ---------------------------------------------------------------------------

def test_weight_normalization_and_validation():
    """Verify weights are normalized to sum to 1.0 and invalid inputs raise ValueError."""
    # Valid custom weights
    recommender = HybridRecommender(weights={"als": 2.0, "content": 2.0, "popularity": 0.0})
    assert recommender.weights["als"] == 0.5
    assert recommender.weights["content"] == 0.5
    assert recommender.weights["popularity"] == 0.0

    # Negative weight
    with pytest.raises(ValueError, match="non-negative"):
        HybridRecommender(weights={"als": -0.5, "content": 0.5, "popularity": 0.5})

    # Non-finite weight
    with pytest.raises(ValueError, match="finite"):
        HybridRecommender(weights={"als": float("nan"), "content": 0.5, "popularity": 0.5})

    # Total weight zero
    with pytest.raises(ValueError, match="strictly positive"):
        HybridRecommender(weights={"als": 0.0, "content": 0.0, "popularity": 0.0})

    # Non-dict weights
    with pytest.raises(ValueError, match="non-empty dictionary"):
        HybridRecommender(weights="invalid")  # type: ignore


# ---------------------------------------------------------------------------
# Integration tests using fitted Phase 1–4 modules
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_models():
    """Build and fit all three underlying recommenders on sample dataset."""
    raw_interactions = load_interactions("data/sample_interactions.csv")
    prep_data = preprocess_interactions(raw_interactions)
    user_item_matrix = build_user_item_matrix(prep_data)
    metadata_df = load_item_metadata("data/item_metadata.csv")

    als = ImplicitALSRecommender(factors=8, iterations=15, random_state=42)
    als.fit(user_item_matrix, prep_data)

    content = ContentRecommender().fit(metadata_df)
    popularity = PopularityRecommender().fit(prep_data.aggregated_df)

    return {
        "als": als,
        "content": content,
        "popularity": popularity,
        "prep_data": prep_data,
    }


def test_basic_hybrid_recommendation(fitted_models):
    """Test standard hybrid recommendation for a known user (U101)."""
    hybrid = HybridRecommender(
        als_recommender=fitted_models["als"],
        content_recommender=fitted_models["content"],
        popularity_recommender=fitted_models["popularity"],
    )

    # U101 interacted with PROD_001 and PROD_002
    user_history = ["PROD_001", "PROD_002"]
    recs = hybrid.recommend(user_id="U101", user_history=user_history, n=3)

    assert isinstance(recs, list)
    assert len(recs) <= 3
    rec_item_ids = [item_id for item_id, _ in recs]

    # Exclusions respected
    assert "PROD_001" not in rec_item_ids
    assert "PROD_002" not in rec_item_ids

    # Scores should be in descending order
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_recommendation_with_breakdown(fitted_models):
    """Test recommendation with detailed score contribution breakdown."""
    hybrid = HybridRecommender(
        als_recommender=fitted_models["als"],
        content_recommender=fitted_models["content"],
        popularity_recommender=fitted_models["popularity"],
    )

    user_history = ["PROD_001"]
    recs = hybrid.recommend(
        user_id="U101",
        user_history=user_history,
        n=3,
        return_breakdown=True,
    )

    assert len(recs) > 0
    first = recs[0]
    assert isinstance(first, HybridRecommendation)
    assert isinstance(first.breakdown, dict)
    assert first.score == sum(first.breakdown.values())
    assert first.score > 0.0


def test_cold_start_new_user_with_preference_text(fitted_models):
    """Test cold-start user (unknown ID) with explicit preference text."""
    hybrid = HybridRecommender(
        als_recommender=fitted_models["als"],
        content_recommender=fitted_models["content"],
        popularity_recommender=fitted_models["popularity"],
    )

    recs = hybrid.recommend(
        user_id="NEW_USER_999",
        user_history=[],
        preference_text="wireless mouse precision optical",
        n=3,
    )

    assert len(recs) > 0
    rec_ids = [item for item, _ in recs]
    # PROD_001 (wireless ergonomic mouse) or PROD_006 (ergonomic vertical mouse) should rank high
    assert "PROD_001" in rec_ids or "PROD_006" in rec_ids


def test_cold_start_new_user_without_preference_text(fitted_models):
    """Test cold-start user without history and without preference falls back to popularity."""
    hybrid = HybridRecommender(
        als_recommender=fitted_models["als"],
        content_recommender=fitted_models["content"],
        popularity_recommender=fitted_models["popularity"],
    )

    # Empty user history & no preference text
    recs = hybrid.recommend(
        user_id="NEW_USER_NO_HISTORY",
        user_history=[],
        preference_text=None,
        n=3,
    )

    # Popularity top items: PROD_002 is most popular (weight 8.0)
    assert len(recs) > 0
    top_item_id, _ = recs[0]
    assert top_item_id == "PROD_002"


def test_sparse_interaction_history(fitted_models):
    """Test user with only 1 interaction (sparse history)."""
    hybrid = HybridRecommender(
        als_recommender=fitted_models["als"],
        content_recommender=fitted_models["content"],
        popularity_recommender=fitted_models["popularity"],
    )

    # User with single interaction
    recs = hybrid.recommend(
        user_id="U105",
        user_history=["PROD_002"],
        n=3,
    )
    rec_ids = [item for item, _ in recs]
    assert "PROD_002" not in rec_ids
    assert len(recs) <= 3


def test_cold_start_item_surfacing(fitted_models):
    """Test that a new item with zero interactions (PROD_006) can surface via content signal."""
    hybrid = HybridRecommender(
        als_recommender=fitted_models["als"],
        content_recommender=fitted_models["content"],
        popularity_recommender=fitted_models["popularity"],
        weights={"als": 0.2, "content": 0.7, "popularity": 0.1},
    )

    # Query with seed or preference related to vertical ergonomic mouse
    recs = hybrid.recommend(
        seed_item_ids=["PROD_001"],
        n=3,
    )
    rec_ids = [item for item, _ in recs]
    # PROD_006 is the vertical ergonomic mouse, highly similar to PROD_001
    assert "PROD_006" in rec_ids


def test_als_unavailable_fallback(fitted_models):
    """Test fallback when ALS is completely None or unfitted."""
    hybrid = HybridRecommender(
        als_recommender=None,
        content_recommender=fitted_models["content"],
        popularity_recommender=fitted_models["popularity"],
    )

    # Should gracefully use content and popularity
    recs = hybrid.recommend(user_history=["PROD_001"], n=3)
    assert len(recs) > 0
    assert "PROD_001" not in [i for i, _ in recs]


def test_content_unavailable_fallback(fitted_models):
    """Test fallback when ContentRecommender is unavailable."""
    hybrid = HybridRecommender(
        als_recommender=fitted_models["als"],
        content_recommender=None,
        popularity_recommender=fitted_models["popularity"],
    )

    recs = hybrid.recommend(user_id="U101", user_history=["PROD_001", "PROD_002"], n=2)
    assert len(recs) <= 2
    assert "PROD_001" not in [i for i, _ in recs]


def test_all_algorithms_unavailable():
    """Test RuntimeError when no recommenders are fitted or available."""
    hybrid = HybridRecommender(
        als_recommender=None,
        content_recommender=None,
        popularity_recommender=None,
    )
    with pytest.raises(RuntimeError, match="All recommenders are unavailable"):
        hybrid.recommend(user_id="U101", n=3)


def test_edge_cases_n_values(fitted_models):
    """Test n=0 and negative n return empty list."""
    hybrid = HybridRecommender(
        popularity_recommender=fitted_models["popularity"],
    )
    assert hybrid.recommend(n=0) == []
    assert hybrid.recommend(n=-5) == []


def test_deterministic_tie_breaking():
    """Verify deterministic tie breaking using item_id secondary sort key."""
    class MockRecommender:
        is_fitted = True

        def recommend(self, n=10):
            # Equal scores for all items
            return [("item_Z", 5.0), ("item_A", 5.0), ("item_M", 5.0)]

    hybrid = HybridRecommender(popularity_recommender=MockRecommender())  # type: ignore
    recs = hybrid.recommend(n=3)
    rec_ids = [item for item, _ in recs]
    # With equal scores, alphabetical order must prevail
    assert rec_ids == ["item_A", "item_M", "item_Z"]


def test_empty_candidate_pool(fitted_models):
    """Test when all catalog items are excluded."""
    hybrid = HybridRecommender(
        popularity_recommender=fitted_models["popularity"],
    )
    # Exclude all 5 known interaction items
    all_items = ["PROD_001", "PROD_002", "PROD_003", "PROD_004", "PROD_005"]
    recs = hybrid.recommend(exclude_items=all_items, n=5)
    assert recs == []

