"""Hybrid recommendation engine combining Collaborative Filtering (ALS),
Content-Based filtering (TF-IDF), and Popularity fallback with weighted ensembling
and robust score normalization.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Hashable, Iterable, List, Optional, Set, Tuple, Union

from app.als_recommender import ImplicitALSRecommender
from app.content_recommender import ContentRecommender
from app.popularity import PopularityRecommender


@dataclass
class HybridRecommendation:
    """Detailed recommendation outcome with score contribution breakdown.

    Attributes
    ----------
    item_id : Hashable
        Identifier of the recommended item.
    score : float
        Final ensembled score.
    breakdown : Dict[str, float]
        Dictionary of algorithm contribution (weighted normalized score per algorithm).
    """

    item_id: Hashable
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)


def normalize_scores(items_with_scores: List[Tuple[Hashable, float]]) -> List[Tuple[Hashable, float]]:
    """Normalize candidate scores to [0.0, 1.0] using min-max normalization.

    Handles:
    - Empty list -> returns empty list.
    - Non-finite values (NaN, Inf, -Inf) -> sanitized/filtered to finite numbers.
    - Single item or all equal scores -> assigned 1.0 (indicating top relevance in the candidate set).
    - Negative scores -> shifted cleanly by subtracting min_val.

    Parameters
    ----------
    items_with_scores : List[Tuple[Hashable, float]]
        List of (item_id, raw_score) tuples.

    Returns
    -------
    List[Tuple[Hashable, float]]
        List of (item_id, normalized_score) tuples in [0.0, 1.0].
    """
    if not items_with_scores:
        return []

    # Clean and sanitize scores
    sanitized: List[Tuple[Hashable, float]] = []
    for item_id, score in items_with_scores:
        try:
            val = float(score)
            if math.isnan(val) or math.isinf(val):
                continue
            sanitized.append((item_id, val))
        except (TypeError, ValueError):
            continue

    if not sanitized:
        return []

    scores = [s for _, s in sanitized]
    min_score = min(scores)
    max_score = max(scores)

    # If all scores are equal (including single item case), assign 1.0 to all
    if math.isclose(min_score, max_score, abs_tol=1e-9):
        return [(item_id, 1.0) for item_id, _ in sanitized]

    span = max_score - min_score
    normalized = [
        (item_id, float((s - min_score) / span))
        for item_id, s in sanitized
    ]
    return normalized


class HybridRecommender:
    """Ensemble recommender combining ALS, Content-Based, and Popularity models.

    Parameters
    ----------
    als_recommender : Optional[ImplicitALSRecommender], default=None
        Fitted collaborative filtering model.
    content_recommender : Optional[ContentRecommender], default=None
        Fitted content-based model.
    popularity_recommender : Optional[PopularityRecommender], default=None
        Fitted popularity fallback model.
    weights : Optional[Dict[str, float]], default=None
        Configurable algorithm weights for 'als', 'content', 'popularity'.
        Defaults to {'als': 0.50, 'content': 0.35, 'popularity': 0.15}.
    candidate_pool_size : int, default=50
        Number of raw candidates requested from each individual recommender before ensembling.
    """

    DEFAULT_WEIGHTS = {
        "als": 0.50,
        "content": 0.35,
        "popularity": 0.15,
    }

    def __init__(
        self,
        als_recommender: Optional[ImplicitALSRecommender] = None,
        content_recommender: Optional[ContentRecommender] = None,
        popularity_recommender: Optional[PopularityRecommender] = None,
        weights: Optional[Dict[str, float]] = None,
        candidate_pool_size: int = 50,
    ) -> None:
        self.als_recommender = als_recommender
        self.content_recommender = content_recommender
        self.popularity_recommender = popularity_recommender
        self.candidate_pool_size = max(candidate_pool_size, 1)

        raw_weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()
        self.weights = self._validate_and_normalize_weights(raw_weights)

    @staticmethod
    def _validate_and_normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Validate that weights are non-negative, finite, and sum to a positive value,
        then normalize them so the total equals 1.0.
        """
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Weights must be a non-empty dictionary.")

        normalized = {}
        total = 0.0
        for key in ["als", "content", "popularity"]:
            w = weights.get(key, 0.0)
            if not isinstance(w, (int, float)):
                raise ValueError(f"Weight for '{key}' must be a numeric value, got {type(w)}.")
            if math.isnan(w) or math.isinf(w):
                raise ValueError(f"Weight for '{key}' must be finite.")
            if w < 0.0:
                raise ValueError(f"Weight for '{key}' must be non-negative (>= 0.0), got {w}.")
            normalized[key] = float(w)
            total += float(w)

        if total <= 0.0:
            raise ValueError("Total weight across all algorithms must be strictly positive (> 0.0).")

        return {k: v / total for k, v in normalized.items()}

    def _is_recommender_available(self, name: str) -> bool:
        """Check if a specific recommender is present and fitted."""
        if name == "als":
            return bool(self.als_recommender and getattr(self.als_recommender, "is_fitted", False))
        if name == "content":
            return bool(self.content_recommender and getattr(self.content_recommender, "is_fitted", False))
        if name == "popularity":
            return bool(self.popularity_recommender and getattr(self.popularity_recommender, "is_fitted", False))
        return False

    def recommend(
        self,
        user_id: Optional[Hashable] = None,
        user_history: Optional[Iterable[Hashable]] = None,
        preference_text: Optional[str] = None,
        seed_item_ids: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        exclude_items: Optional[Iterable[Hashable]] = None,
        n: int = 5,
        return_breakdown: bool = False,
    ) -> Union[List[Tuple[Hashable, float]], List[HybridRecommendation]]:
        """Generate hybrid recommendations combining available signals.

        Cold-Start & Fallback Decision Flow:
        1. Known user with history / active ALS:
           - Generates candidates from ALS (collaborative signals).
           - Generates candidates from ContentRecommender (using history items or explicit seeds).
           - Generates candidates from PopularityRecommender (popularity signal).
        2. Cold-start user with preference text (or seed items):
           - Content-based candidates are retrieved via preference_text or seed items.
           - Popularity candidates are combined as fallback signal.
           - ALS contributes no candidates (user unknown in training matrix).
        3. Cold-start user without history and without preference:
           - Personalization algorithms yield no candidates.
           - System falls back cleanly to PopularityRecommender.
        4. Partial / Missing recommenders:
           - Any unavailable or un-fitted recommender has its weight redistributed
             proportionally across active recommenders that generate candidates.

        Parameters
        ----------
        user_id : Optional[Hashable], default=None
            User ID for ALS lookup.
        user_history : Optional[Iterable[Hashable]], default=None
            Item IDs previously interacted with by this user (excluded from results).
        preference_text : Optional[str], default=None
            Search text or category interest for new users without interaction history.
        seed_item_ids : Optional[Union[Hashable, Iterable[Hashable]]], default=None
            Explicit seed items for content similarity.
        exclude_items : Optional[Iterable[Hashable]], default=None
            Additional item IDs to exclude from recommendation output.
        n : int, default=5
            Number of recommendations to return. If n <= 0, returns [].
        return_breakdown : bool, default=False
            If True, returns a list of `HybridRecommendation` objects containing score breakdowns.
            If False, returns a list of `(item_id, score)` tuples.

        Returns
        -------
        Union[List[Tuple[Hashable, float]], List[HybridRecommendation]]
            Ranked recommendations.

        Raises
        ------
        RuntimeError
            If all recommenders are unavailable or not fitted.
        """
        if n <= 0:
            return []

        # Check if at least one recommender is available
        has_any = any(
            self._is_recommender_available(name)
            for name in ["als", "content", "popularity"]
        )
        if not has_any:
            raise RuntimeError(
                "All recommenders are unavailable or un-fitted. Cannot generate hybrid recommendations."
            )

        # Build comprehensive exclusion set
        exclusions: Set[Hashable] = set()
        if user_history:
            exclusions.update(user_history)
        if exclude_items:
            exclusions.update(exclude_items)
        if seed_item_ids:
            if isinstance(seed_item_ids, (str, int, float)) or not isinstance(seed_item_ids, Iterable):
                exclusions.add(seed_item_ids)
            else:
                exclusions.update(seed_item_ids)

        # Step 1: Collect raw candidate lists from available recommenders
        raw_candidates: Dict[str, List[Tuple[Hashable, float]]] = {}

        # 1a. ALS Candidates
        if self._is_recommender_available("als") and user_id is not None:
            try:
                als_recs = self.als_recommender.recommend(
                    user_id=user_id,
                    n=self.candidate_pool_size,
                    filter_already_liked_items=True,
                )
                if isinstance(als_recs, list):
                    raw_candidates["als"] = als_recs
            except Exception:
                raw_candidates["als"] = []

        # 1b. Content Candidates
        if self._is_recommender_available("content"):
            content_recs: List[Tuple[Hashable, float]] = []
            try:
                # If explicit seed item IDs were provided
                if seed_item_ids is not None:
                    content_recs = self.content_recommender.recommend_similar_items(
                        seed_item_ids=seed_item_ids,
                        n=self.candidate_pool_size,
                        exclude_items=exclusions,
                    )
                # Else use user history or preference text
                else:
                    content_recs = self.content_recommender.recommend_for_user(
                        user_history_items=user_history,
                        n=self.candidate_pool_size,
                        preference_text=preference_text,
                    )
                if isinstance(content_recs, list):
                    raw_candidates["content"] = content_recs
            except Exception:
                raw_candidates["content"] = []

        # 1c. Popularity Candidates
        if self._is_recommender_available("popularity"):
            try:
                pop_recs = self.popularity_recommender.recommend(n=self.candidate_pool_size)
                if isinstance(pop_recs, list):
                    raw_candidates["popularity"] = pop_recs
            except Exception:
                raw_candidates["popularity"] = []

        # Step 2: Normalize candidate scores per algorithm and filter exclusions
        norm_candidates: Dict[str, Dict[Hashable, float]] = {}
        for algo_name, candidate_list in raw_candidates.items():
            normalized_list = normalize_scores(candidate_list)
            # Filter exclusions and store dict: item_id -> normalized_score
            filtered_dict = {
                item_id: score
                for item_id, score in normalized_list
                if item_id not in exclusions
            }
            if filtered_dict:
                norm_candidates[algo_name] = filtered_dict

        # If after filtering no candidates exist anywhere
        if not norm_candidates:
            return []

        # Step 3: Dynamic weight redistribution among algorithms that produced valid candidates
        active_algos = list(norm_candidates.keys())
        active_weights_sum = sum(self.weights.get(algo, 0.0) for algo in active_algos)

        effective_weights: Dict[str, float] = {}
        if active_weights_sum > 0.0:
            effective_weights = {
                algo: self.weights.get(algo, 0.0) / active_weights_sum
                for algo in active_algos
            }
        else:
            # If all active algorithms had 0 configured weight, divide equally
            equal_w = 1.0 / len(active_algos)
            effective_weights = {algo: equal_w for algo in active_algos}

        # Step 4: Aggregate weighted scores across candidate pool
        all_candidate_items: Set[Hashable] = set()
        for cdict in norm_candidates.values():
            all_candidate_items.update(cdict.keys())

        aggregated_results: List[HybridRecommendation] = []
        for item_id in all_candidate_items:
            total_score = 0.0
            breakdown: Dict[str, float] = {}
            for algo, cdict in norm_candidates.items():
                if item_id in cdict:
                    contrib = effective_weights[algo] * cdict[item_id]
                    total_score += contrib
                    breakdown[algo] = contrib
            aggregated_results.append(
                HybridRecommendation(item_id=item_id, score=total_score, breakdown=breakdown)
            )

        # Step 5: Deterministic ranking
        # Primary sort: score descending (-score)
        # Secondary sort: item_id ascending (string representation)
        aggregated_results.sort(key=lambda rec: (-rec.score, str(rec.item_id)))

        top_n = aggregated_results[:n]

        if return_breakdown:
            return top_n
        return [(rec.item_id, rec.score) for rec in top_n]

