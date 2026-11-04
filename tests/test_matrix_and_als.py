"""Tests for matrix_builder and als_recommender modules."""

import pytest
import numpy as np
import scipy.sparse as sp
import pandas as pd

from app.data_loader import load_interactions
from app.preprocessor import preprocess_interactions, PreprocessedData
from app.matrix_builder import build_user_item_matrix
from app.als_recommender import ImplicitALSRecommender


def test_build_user_item_matrix_sample_data():
    """Test user-item CSR matrix construction on sample dataset."""
    raw_df = load_interactions("data/sample_interactions.csv")
    prep_data = preprocess_interactions(raw_df)
    csr = build_user_item_matrix(prep_data)

    assert sp.isspmatrix_csr(csr)
    assert csr.dtype == np.float32
    # 5 unique users and 5 unique items
    assert csr.shape == (5, 5)
    # Aggregated unique user-item interactions count = 10
    assert csr.nnz == 10

    # Validate orientation: rows are users, columns are items
    # Check U101 (user_idx for U101) and PROD_001 (item_idx for PROD_001)
    u101_idx = prep_data.user_to_idx["U101"]
    prod1_idx = prep_data.item_to_idx["PROD_001"]
    prod2_idx = prep_data.item_to_idx["PROD_002"]

    # In sample data, U101 interacted with PROD_001 (1.0 + 2.0 = 3.0) and PROD_002 (5.0)
    assert csr[u101_idx, prod1_idx] == 3.0
    assert csr[u101_idx, prod2_idx] == 5.0


def test_build_user_item_matrix_invalid_inputs():
    """Test validation errors for invalid or empty inputs."""
    with pytest.raises(ValueError, match="cannot be None"):
        build_user_item_matrix(None)

    # Empty user/item mappings
    empty_prep = PreprocessedData(
        aggregated_df=pd.DataFrame(),
        user_to_idx={},
        idx_to_user={},
        item_to_idx={},
        idx_to_item={},
    )
    with pytest.raises(ValueError, match="zero users or zero items"):
        build_user_item_matrix(empty_prep)

    # Missing required columns in DataFrame
    malformed_prep = PreprocessedData(
        aggregated_df=pd.DataFrame({"user_idx": [0]}),
        user_to_idx={"u1": 0},
        idx_to_user={0: "u1"},
        item_to_idx={"i1": 0},
        idx_to_item={0: "i1"},
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        build_user_item_matrix(malformed_prep)


def test_build_user_item_matrix_out_of_bounds_index():
    """Test error when index values exceed user/item mapping bounds."""
    bad_df = pd.DataFrame(
        {
            "user_id": ["u1"],
            "item_id": ["i1"],
            "weight": [1.0],
            "user_idx": [5],  # out of bounds since len(user_to_idx) is 1
            "item_idx": [0],
        }
    )
    bad_prep = PreprocessedData(
        aggregated_df=bad_df,
        user_to_idx={"u1": 0},
        idx_to_user={0: "u1"},
        item_to_idx={"i1": 0},
        idx_to_item={0: "i1"},
    )
    with pytest.raises(ValueError, match="out of bounds"):
        build_user_item_matrix(bad_prep)


def test_als_recommender_tiny_synthetic_dataset():
    """Test ALS training and recommendation properties on tiny synthetic dataset."""
    # 4 users, 4 items
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4"],
            "item_id": ["i1", "i2", "i2", "i3", "i3", "i4", "i1"],
            "weight": [5.0, 3.0, 4.0, 2.0, 5.0, 1.0, 4.0],
        }
    )
    prep_data = preprocess_interactions(df)
    matrix = build_user_item_matrix(prep_data)

    als = ImplicitALSRecommender(factors=4, iterations=10, random_state=42)
    als.fit(matrix, prep_data)

    assert als.is_fitted
    assert als.model is not None

    # u1 interacted with i1 and i2
    # Recommendation should only return unseen items (i3, i4)
    recs = als.recommend("u1", n=2)
    assert len(recs) <= 2
    rec_items = [item for item, _ in recs]
    assert "i1" not in rec_items
    assert "i2" not in rec_items
    for item in rec_items:
        assert item in ["i3", "i4"]


def test_als_recommender_sample_dataset():
    """Test ALS training on actual sample data."""
    raw_df = load_interactions("data/sample_interactions.csv")
    prep_data = preprocess_interactions(raw_df)
    matrix = build_user_item_matrix(prep_data)

    als = ImplicitALSRecommender(factors=8, iterations=15, random_state=42)
    als.fit(matrix, prep_data)

    # U101 has interacted with PROD_001 and PROD_002
    u101_recs = als.recommend("U101", n=3)
    rec_item_ids = [item_id for item_id, _ in u101_recs]

    # Neither PROD_001 nor PROD_002 should appear
    assert "PROD_001" not in rec_item_ids
    assert "PROD_002" not in rec_item_ids
    assert len(u101_recs) <= 3


def test_als_recommender_edge_cases():
    """Test edge cases: unknown user, n <= 0, and n > unobserved items."""
    raw_df = load_interactions("data/sample_interactions.csv")
    prep_data = preprocess_interactions(raw_df)
    matrix = build_user_item_matrix(prep_data)

    als = ImplicitALSRecommender(factors=4, iterations=5, random_state=42)

    # Recommend before fit raises RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted"):
        als.recommend("U101", n=2)

    als.fit(matrix, prep_data)

    # Unknown user returns empty list
    assert als.recommend("NON_EXISTENT_USER", n=5) == []

    # n = 0 or n < 0 returns empty list
    assert als.recommend("U101", n=0) == []
    assert als.recommend("U101", n=-2) == []

    # Requesting n larger than unobserved catalog size returns all available unobserved items
    # U101 has 2 interacted items out of 5 total, so at most 3 unseen items exist
    recs_large_n = als.recommend("U101", n=100)
    assert len(recs_large_n) <= 3
    assert all(item not in ["PROD_001", "PROD_002"] for item, _ in recs_large_n)


def test_als_recommender_invalid_fit():
    """Test ALS fit input validation."""
    prep = PreprocessedData(
        aggregated_df=pd.DataFrame({"user_idx": [0], "item_idx": [0], "weight": [1.0]}),
        user_to_idx={"u1": 0},
        idx_to_user={0: "u1"},
        item_to_idx={"i1": 0},
        idx_to_item={0: "i1"},
    )
    als = ImplicitALSRecommender()

    # Non-CSR matrix
    with pytest.raises(ValueError, match="scipy.sparse.csr_matrix"):
        als.fit(np.array([[1.0]]), prep)

    # Shape mismatch with user mapping (non-empty matrix)
    matrix_wrong_users = sp.csr_matrix([[1.0], [1.0]])
    with pytest.raises(ValueError, match="Matrix row count"):
        als.fit(matrix_wrong_users, prep)

    # Shape mismatch with item mapping (non-empty matrix)
    matrix_wrong_items = sp.csr_matrix([[1.0, 1.0]])
    with pytest.raises(ValueError, match="Matrix column count"):
        als.fit(matrix_wrong_items, prep)

    # Empty matrix
    matrix_empty = sp.csr_matrix(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="Cannot train ALS on an empty matrix"):
        als.fit(matrix_empty, prep)
