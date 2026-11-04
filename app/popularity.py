"""Popularity-based recommender module.

Ranks items based on total interaction weight with deterministic tie-breaking.
"""

from typing import Any, Dict, Hashable, List, Tuple
import pandas as pd


class PopularityRecommender:
    """Popularity fallback recommender.

    Scores items by sum of interaction weights and resolves ties
    deterministically using item ID as a secondary sort key.
    """

    def __init__(self) -> None:
        self.item_scores: Dict[Hashable, float] = {}
        self._ranked_items: List[Tuple[Hashable, float]] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PopularityRecommender":
        """Compute item popularity scores from interaction data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'item_id' and 'weight'.

        Returns
        -------
        PopularityRecommender
            The fitted recommender instance.

        Raises
        ------
        ValueError
            If df is empty or missing required columns.
        """
        if df.empty:
            raise ValueError("Cannot fit PopularityRecommender on empty data.")

        required = {"item_id", "weight"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for popularity recommender: {sorted(missing)}")

        grouped = (
            df.groupby("item_id")["weight"]
            .sum()
            .reset_index()
        )

        # Store dict of item_id -> total_weight
        self.item_scores = dict(zip(grouped["item_id"], grouped["weight"]))

        # Rank items: primary key = weight descending (-weight), secondary key = str(item_id) ascending
        # Python's sort is stable and tuples handle multi-key sorting cleanly
        raw_list = list(self.item_scores.items())
        # Sort by (-score, str(item_id))
        self._ranked_items = sorted(
            raw_list,
            key=lambda pair: (-pair[1], str(pair[0]))
        )
        self.is_fitted = True
        return self

    def recommend(self, n: int = 5) -> List[Tuple[Hashable, float]]:
        """Return the top-N recommended items with their popularity scores.

        Parameters
        ----------
        n : int, default=5
            Number of top items to return. If n <= 0, returns an empty list.
            If n > available items, returns all available items.

        Returns
        -------
        List[Tuple[Hashable, float]]
            List of (item_id, score) tuples ordered by popularity.

        Raises
        ------
        RuntimeError
            If called before `fit`.
        """
        if not self.is_fitted:
            raise RuntimeError("PopularityRecommender must be fitted before calling recommend().")

        if n <= 0:
            return []

        return self._ranked_items[:n]

