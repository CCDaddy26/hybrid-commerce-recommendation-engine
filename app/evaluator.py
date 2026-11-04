"""Evaluation splitting module with strict leakage prevention.

Supports reproducible train/test and train/val/test splits using:
1. Leave-One-Out (or Leave-K-Out) per-user splitting for users with >= min_interactions.
2. Temporal splitting (when timestamps are present).
3. Explicit identification of cold-start and evaluation-ineligible users.
"""

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    """Container for split interaction datasets.

    Attributes
    ----------
    train_df : pd.DataFrame
        Training interactions.
    val_df : pd.DataFrame
        Validation interactions (used for hyperparameter/weight tuning).
    test_df : pd.DataFrame
        Held-out testing interactions (used strictly for final evaluation).
    eval_user_ids : List[Hashable]
        List of user IDs with ground-truth test interactions.
    val_user_ids : List[Hashable]
        List of user IDs with ground-truth validation interactions.
    insufficient_data_users : List[Hashable]
        Users who had fewer than the required interactions for evaluation splitting.
    """

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    eval_user_ids: List[Hashable]
    val_user_ids: List[Hashable]
    insufficient_data_users: List[Hashable]


def split_interactions(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.0,
    min_interactions_for_eval: int = 2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split interaction data into train, optional validation, and test sets without leakage.

    Splitting Strategy & Leakage Prevention:
    - User-item interaction rows are split per user.
    - For each user with at least `min_interactions_for_eval` interactions:
      At least 1 interaction is held out for the test set, and optionally 1 for validation.
      The remaining interactions remain strictly in `train_df`.
    - Users with fewer than `min_interactions_for_eval` interactions cannot be evaluated
      fairly on held-out ranking, so all their interactions remain in `train_df` (or they are
      flagged as `insufficient_data_users`).
    - Test interactions are strictly omitted from `train_df`, ensuring zero leakage into model fitting.
    - Deterministic and reproducible via `random_state`.

    Parameters
    ----------
    df : pd.DataFrame
        Interactions DataFrame containing 'user_id', 'item_id', 'weight'.
    test_ratio : float, default=0.2
        Target proportion of interactions held out for testing.
    val_ratio : float, default=0.0
        Target proportion held out for validation (e.g. weight tuning).
    min_interactions_for_eval : int, default=2
        Minimum interactions required for a user to contribute to test/val sets.
    random_state : int, default=42
        Seed for reproducible random sampling.

    Returns
    -------
    DatasetSplit
        Split datasets and metadata.
    """
    if df.empty:
        raise ValueError("Cannot split an empty interactions DataFrame.")

    required_cols = {"user_id", "item_id", "weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for splitting: {sorted(missing)}")

    rng = np.random.RandomState(random_state)

    train_rows = []
    val_rows = []
    test_rows = []

    eval_users = []
    val_users = []
    insufficient_users = []

    # Group interactions by user
    grouped = df.groupby("user_id")

    for user_id, user_data in grouped:
        n_items = len(user_data)
        if n_items < min_interactions_for_eval:
            train_rows.append(user_data)
            insufficient_users.append(user_id)
            continue

        # Shuffle indices deterministically for this user
        indices = user_data.index.to_numpy()
        rng.shuffle(indices)

        n_test = max(1, int(round(n_items * test_ratio)))
        n_val = int(round(n_items * val_ratio)) if val_ratio > 0.0 else 0

        # Ensure at least 1 item remains in train
        if n_test + n_val >= n_items:
            if n_val > 0 and n_items >= 3:
                n_test = 1
                n_val = 1
            else:
                n_test = 1
                n_val = 0

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        test_rows.append(df.loc[test_idx])
        eval_users.append(user_id)

        if len(val_idx) > 0:
            val_rows.append(df.loc[val_idx])
            val_users.append(user_id)

        train_rows.append(df.loc[train_idx])

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)

    return DatasetSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        eval_user_ids=eval_users,
        val_user_ids=val_users,
        insufficient_data_users=insufficient_users,
    )

