"""Tests for preprocessor and popularity recommender modules."""

import pytest
import pandas as pd
from app.data_loader import load_interactions
from app.preprocessor import preprocess_interactions, PreprocessedData
from app.popularity import PopularityRecommender


def test_preprocess_sample_interactions():
    """Verify preprocessing on the actual Phase 1 sample dataset."""
    raw_df = load_interactions("data/sample_interactions.csv")
    result = preprocess_interactions(raw_df)

    assert isinstance(result, PreprocessedData)
    # Original CSV has 12 rows, U101 has 2 rows for PROD_001 (view: 1.0, click: 2.0 -> sum 3.0),
    # U103 has 2 rows for PROD_004 (view: 1.0, purchase: 5.0 -> sum 6.0).
    # Total unique pairs should be 10.
    assert len(result.aggregated_df) == 10

    # Contiguous 0-based indexing checks
    assert set(result.user_to_idx.values()) == set(range(5))
    assert set(result.item_to_idx.values()) == set(range(5))
    assert len(result.idx_to_user) == 5
    assert len(result.idx_to_item) == 5

    # Reverse mapping fidelity
    for user, idx in result.user_to_idx.items():
        assert result.idx_to_user[idx] == user
    for item, idx in result.item_to_idx.items():
        assert result.idx_to_item[idx] == item

    # Check that aggregated weights match expectations
    u101_prod1 = result.aggregated_df[
        (result.aggregated_df["user_id"] == "U101") & (result.aggregated_df["item_id"] == "PROD_001")
    ]
    assert len(u101_prod1) == 1
    assert u101_prod1.iloc[0]["weight"] == 3.0


def test_preprocess_duplicate_aggregation_and_non_numeric_ids():
    """Test duplicate aggregation with alphanumeric/non-numeric strings as IDs."""
    data = pd.DataFrame(
        {
            "user_id": ["user-alpha", "user-beta", "user-alpha", "user-gamma"],
            "item_id": ["sku#1", "sku#2", "sku#1", "sku#3"],
            "weight": [2.5, 4.0, 1.5, 3.0],
        }
    )
    result = preprocess_interactions(data)

    # user-alpha + sku#1 aggregated: 2.5 + 1.5 = 4.0
    alpha_sku1 = result.aggregated_df[
        (result.aggregated_df["user_id"] == "user-alpha") & (result.aggregated_df["item_id"] == "sku#1")
    ]
    assert alpha_sku1.iloc[0]["weight"] == 4.0

    # Ensure indices are 0-based contiguous integers
    assert sorted(result.user_to_idx.values()) == [0, 1, 2]
    assert sorted(result.item_to_idx.values()) == [0, 1, 2]

    # Verify column presence
    assert list(result.aggregated_df.columns) == [
        "user_id",
        "item_id",
        "weight",
        "user_idx",
        "item_idx",
    ]


def test_preprocess_empty_dataframe():
    """Test error when empty DataFrame is passed."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty DataFrame"):
        preprocess_interactions(empty_df)


def test_preprocess_missing_columns():
    """Test error when required columns are absent."""
    df = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_interactions(df)


def test_popularity_recommender_sample_data():
    """Test popularity scores and top-N ranking on sample data."""
    raw_df = load_interactions("data/sample_interactions.csv")
    recommender = PopularityRecommender()
    recommender.fit(raw_df)

    assert recommender.is_fitted
    # PROD_002: U101(5) + U102(1) + U105(2) = 8.0
    # PROD_004: U103(1 + 5) = 6.0
    # PROD_003: U102(5) + U104(1) = 6.0
    # PROD_005: U104(5) + U105(1) = 6.0
    # PROD_001: U101(1 + 2) + U103(2) = 5.0
    assert recommender.item_scores["PROD_002"] == 8.0
    assert recommender.item_scores["PROD_001"] == 5.0

    # Top 1
    top1 = recommender.recommend(1)
    assert top1 == [("PROD_002", 8.0)]

    # Top 3
    top3 = recommender.recommend(3)
    assert len(top3) == 3
    assert top3[0] == ("PROD_002", 8.0)


def test_popularity_recommender_deterministic_ties():
    """Test deterministic tie-breaking by item ID."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["item_C", "item_A", "item_B"],
            "weight": [10.0, 10.0, 10.0],
        }
    )
    recommender = PopularityRecommender().fit(df)
    results = recommender.recommend(3)

    # Weights are equal; sorted alphabetically by item_id
    assert results == [
        ("item_A", 10.0),
        ("item_B", 10.0),
        ("item_C", 10.0),
    ]


def test_popularity_recommender_edge_cases():
    """Test n <= 0, n > total available items, and unfitted call."""
    df = pd.DataFrame(
        {
            "item_id": ["prod_A", "prod_B"],
            "weight": [2.0, 5.0],
        }
    )
    unfitted = PopularityRecommender()
    with pytest.raises(RuntimeError, match="must be fitted"):
        unfitted.recommend(5)

    recommender = PopularityRecommender().fit(df)

    # n = 0
    assert recommender.recommend(0) == []
    # n < 0
    assert recommender.recommend(-3) == []
    # n > available items
    recommendations = recommender.recommend(100)
    assert len(recommendations) == 2
    assert recommendations[0] == ("prod_B", 5.0)
    assert recommendations[1] == ("prod_A", 2.0)


def test_popularity_recommender_invalid_fit():
    """Test validation when fitting empty or malformed data."""
    with pytest.raises(ValueError, match="empty data"):
        PopularityRecommender().fit(pd.DataFrame())

    with pytest.raises(ValueError, match="Missing required columns"):
        PopularityRecommender().fit(pd.DataFrame({"item_id": ["p1"]}))

