"""Preprocessing module for interaction data.

Aggregates duplicate interactions and builds contiguous user/item index mappings.
"""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Tuple
import pandas as pd


@dataclass
class PreprocessedData:
    """Container for preprocessed interaction data and index mappings.

    Attributes
    ----------
    aggregated_df : pd.DataFrame
        DataFrame with unique (user_id, item_id) rows, contiguous indices, and summed weights.
        Columns: ['user_id', 'item_id', 'weight', 'user_idx', 'item_idx']
    user_to_idx : Dict[Hashable, int]
        Mapping from raw user ID to contiguous integer index (0-based).
    idx_to_user : Dict[int, Hashable]
        Mapping from contiguous integer index back to raw user ID.
    item_to_idx : Dict[Hashable, int]
        Mapping from raw item ID to contiguous integer index (0-based).
    idx_to_item : Dict[int, Hashable]
        Mapping from contiguous integer index back to raw item ID.
    """

    aggregated_df: pd.DataFrame
    user_to_idx: Dict[Hashable, int]
    idx_to_user: Dict[int, Hashable]
    item_to_idx: Dict[Hashable, int]
    idx_to_item: Dict[int, Hashable]


def preprocess_interactions(df: pd.DataFrame) -> PreprocessedData:
    """Aggregate duplicate interactions and create contiguous index mappings.

    Parameters
    ----------
    df : pd.DataFrame
        Interaction DataFrame containing at least 'user_id', 'item_id', and 'weight'.

    Returns
    -------
    PreprocessedData
        Dataclass containing aggregated interactions and bidirectional mappings.

    Raises
    ------
    ValueError
        If DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("Cannot preprocess an empty DataFrame.")

    required = {"user_id", "item_id", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for preprocessing: {sorted(missing)}")

    # Aggregate weights for duplicate user_id, item_id pairs
    agg_df = (
        df.groupby(["user_id", "item_id"], as_index=False)["weight"]
        .sum()
        .sort_values(by=["user_id", "item_id"])
        .reset_index(drop=True)
    )

    # Build stable, deterministic, contiguous 0-based user mappings
    unique_users = sorted(agg_df["user_id"].unique(), key=lambda x: str(x))
    user_to_idx: Dict[Hashable, int] = {user: idx for idx, user in enumerate(unique_users)}
    idx_to_user: Dict[int, Hashable] = {idx: user for user, idx in user_to_idx.items()}

    # Build stable, deterministic, contiguous 0-based item mappings
    unique_items = sorted(agg_df["item_id"].unique(), key=lambda x: str(x))
    item_to_idx: Dict[Hashable, int] = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_item: Dict[int, Hashable] = {idx: item for item, idx in item_to_idx.items()}

    # Add index columns
    agg_df["user_idx"] = agg_df["user_id"].map(user_to_idx)
    agg_df["item_idx"] = agg_df["item_id"].map(item_to_idx)

    return PreprocessedData(
        aggregated_df=agg_df,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )

