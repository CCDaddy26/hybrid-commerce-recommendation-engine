"""Sparse matrix construction module for interaction data.

Constructs user-item CSR matrices from preprocessed interaction data.
"""

from typing import Tuple
import numpy as np
import scipy.sparse as sp
from app.preprocessor import PreprocessedData


def build_user_item_matrix(data: PreprocessedData) -> sp.csr_matrix:
    """Build a user-item CSR sparse matrix from PreprocessedData.

    Matrix Orientation:
    - Rows represent users (shape index 0: number of unique users).
    - Columns represent items (shape index 1: number of unique items).
    - Entry (u, i) represents the aggregated interaction weight for user_idx u and item_idx i.

    Parameters
    ----------
    data : PreprocessedData
        Container with aggregated interactions and contiguous 0-based index mappings.

    Returns
    -------
    sp.csr_matrix
        Sparse CSR matrix of shape (num_users, num_items) with dtype float32.

    Raises
    ------
    ValueError
        If data is empty, has zero users/items, or indices are out of bounds.
    """
    if data is None:
        raise ValueError("PreprocessedData cannot be None.")

    num_users = len(data.user_to_idx)
    num_items = len(data.item_to_idx)

    if num_users == 0 or num_items == 0:
        raise ValueError("Cannot build user-item matrix with zero users or zero items.")

    df = data.aggregated_df
    if df.empty:
        raise ValueError("Aggregated interactions DataFrame is empty.")

    required_cols = {"user_idx", "item_idx", "weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in aggregated DataFrame: {sorted(missing)}")

    user_indices = df["user_idx"].to_numpy()
    item_indices = df["item_idx"].to_numpy()
    weights = df["weight"].to_numpy(dtype=np.float32)

    # Validate bounds
    if (user_indices < 0).any() or (user_indices >= num_users).any():
        raise ValueError("User indices in DataFrame are out of bounds [0, num_users - 1].")

    if (item_indices < 0).any() or (item_indices >= num_items).any():
        raise ValueError("Item indices in DataFrame are out of bounds [0, num_items - 1].")

    # Build CSR matrix
    # shape: (num_users, num_items)
    matrix = sp.csr_matrix(
        (weights, (user_indices, item_indices)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    return matrix

