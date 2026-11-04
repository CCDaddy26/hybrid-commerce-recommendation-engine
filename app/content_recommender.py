"""Content-based recommender module using TF-IDF and Cosine Similarity.

Computes item representations from text metadata and recommends similar items.
Supports:
- Item-to-item recommendations from one or more seed items.
- Cold-start items: New items with metadata but zero prior interactions.
- User cold-start transparency: Explicitly distinguishes between users with interaction
  history versus users with no history (requiring explicit seed items or preference text).
"""

from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """Content-based recommender using TF-IDF vectorization and cosine similarity."""

    def __init__(self, stop_words: Optional[str] = "english") -> None:
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.item_ids: List[Hashable] = []
        self.item_to_idx: Dict[Hashable, int] = {}
        self.idx_to_item: Dict[int, Hashable] = {}
        self.is_fitted: bool = False

    def fit(self, metadata_df: pd.DataFrame) -> "ContentRecommender":
        """Fit TF-IDF representations on item metadata.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            DataFrame containing 'item_id' and 'content_text' columns.

        Returns
        -------
        ContentRecommender
            The fitted recommender instance.

        Raises
        ------
        ValueError
            If metadata_df is empty or missing required columns.
        """
        if metadata_df.empty:
            raise ValueError("Cannot fit ContentRecommender on empty metadata.")

        required = {"item_id", "content_text"}
        missing = required - set(metadata_df.columns)
        if missing:
            raise ValueError(f"Metadata DataFrame missing required columns: {sorted(missing)}")

        # Maintain deterministic order
        clean_df = metadata_df.copy().reset_index(drop=True)

        self.item_ids = list(clean_df["item_id"])
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.idx_to_item = {idx: item_id for idx, item_id in enumerate(self.item_ids)}

        # Fit TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(clean_df["content_text"])
        self.is_fitted = True
        return self

    def recommend_similar_items(
        self,
        seed_item_ids: Union[Hashable, Iterable[Hashable]],
        n: int = 5,
        exclude_items: Optional[Iterable[Hashable]] = None,
    ) -> List[Tuple[Hashable, float]]:
        """Recommend items most similar to one or more seed items.

        Parameters
        ----------
        seed_item_ids : Union[Hashable, Iterable[Hashable]]
            One or more item IDs whose content representations are averaged to form the query vector.
        n : int, default=5
            Number of top similar items to return. If n <= 0, returns [].
        exclude_items : Optional[Iterable[Hashable]], default=None
            Additional item IDs to exclude from results. Seed items are always excluded.

        Returns
        -------
        List[Tuple[Hashable, float]]
            List of (item_id, cosine_similarity_score) ordered by similarity descending.

        Raises
        ------
        RuntimeError
            If called before fitting.
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("ContentRecommender must be fitted before calling recommend_similar_items().")

        if n <= 0:
            return []

        # Normalize seeds to a list
        if isinstance(seed_item_ids, (str, int, float)) or not isinstance(seed_item_ids, Iterable):
            raw_seeds = [seed_item_ids]
        else:
            raw_seeds = list(seed_item_ids)

        valid_seed_indices = [
            self.item_to_idx[sid] for sid in raw_seeds if sid in self.item_to_idx
        ]

        # If no valid seed items are found in catalog
        if not valid_seed_indices:
            return []

        # Build query vector: mean of seed item vectors
        query_vector = self.tfidf_matrix[valid_seed_indices].mean(axis=0)
        query_vector = np.asarray(query_vector)

        # Compute cosine similarity between query vector and all item vectors
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Build exclusion set: seeds are always excluded
        to_exclude: Set[Hashable] = set(raw_seeds)
        if exclude_items is not None:
            to_exclude.update(exclude_items)

        # Rank candidates deterministically:
        # primary: -score descending, secondary: str(item_id) ascending
        candidates = []
        for idx, score in enumerate(similarities):
            item_id = self.idx_to_item[idx]
            if item_id in to_exclude:
                continue
            candidates.append((item_id, float(score)))

        candidates.sort(key=lambda pair: (-pair[1], str(pair[0])))

        return candidates[:n]

    def recommend_for_user(
        self,
        user_history_items: Optional[Iterable[Hashable]],
        n: int = 5,
        preference_text: Optional[str] = None,
    ) -> List[Tuple[Hashable, float]]:
        """Generate content-based recommendations for a user.

        Cold-Start Logic:
        1. User WITH history:
           Uses their historical interacted items as seeds to find similar unseen items.
           All items in `user_history_items` are excluded from the recommendations.
        2. User WITHOUT history (Cold-Start User):
           Item metadata alone cannot infer personal preferences for a brand-new user.
           The caller must either provide `preference_text` (e.g. search query, category)
           or explicit seeds. If neither is available, an empty list is returned so the caller
           can fall back to popularity recommendations rather than guessing.

        Parameters
        ----------
        user_history_items : Optional[Iterable[Hashable]]
            List of item IDs the user previously interacted with.
        n : int, default=5
            Number of recommendations to return.
        preference_text : Optional[str], default=None
            Explicit query or category interest string for cold-start users without interaction history.

        Returns
        -------
        List[Tuple[Hashable, float]]
            List of (item_id, similarity_score).
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("ContentRecommender must be fitted before calling recommend_for_user().")

        if n <= 0:
            return []

        history_list = list(user_history_items) if user_history_items else []

        # Case 1: User has interaction history
        if history_list:
            return self.recommend_similar_items(
                seed_item_ids=history_list,
                n=n,
                exclude_items=history_list,
            )

        # Case 2: Cold-start user with explicit preference query / category
        if preference_text and preference_text.strip():
            query_vec = self.vectorizer.transform([preference_text.strip()])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            candidates = [
                (self.idx_to_item[idx], float(score))
                for idx, score in enumerate(similarities)
            ]
            candidates.sort(key=lambda pair: (-pair[1], str(pair[0])))
            return candidates[:n]

        # Case 3: Cold-start user with no history and no explicit preference
        # We do not pretend we can infer personalized recommendations without data.
        return []

