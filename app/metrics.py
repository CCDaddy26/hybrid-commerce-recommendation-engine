"""Evaluation metrics module for recommendation systems.

Computes standard ranking and retrieval metrics:
- Precision@K
- Recall@K
- MAP@K (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
- Coverage@K
- Catalog Coverage
"""

import math
from typing import Dict, Hashable, List, Optional, Sequence, Set, Union


def validate_k(k: int) -> None:
    """Validate that K is a strictly positive integer."""
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"K must be a strictly positive integer (> 0), got: {k}")


def precision_at_k(recommended: Sequence[Hashable], ground_truth: Set[Hashable], k: int) -> float:
    """Calculate Precision@K.

    Precision@K = (# of relevant items in top-K) / K
    """
    validate_k(k)
    if not ground_truth:
        return 0.0

    top_k = recommended[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for item in top_k if item in ground_truth)
    return float(hits / k)


def recall_at_k(recommended: Sequence[Hashable], ground_truth: Set[Hashable], k: int) -> float:
    """Calculate Recall@K.

    Recall@K = (# of relevant items in top-K) / len(ground_truth)
    """
    validate_k(k)
    if not ground_truth:
        return 0.0

    top_k = recommended[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for item in top_k if item in ground_truth)
    return float(hits / len(ground_truth))


def average_precision_at_k(recommended: Sequence[Hashable], ground_truth: Set[Hashable], k: int) -> float:
    """Calculate Average Precision@K (AP@K) for a single user."""
    validate_k(k)
    if not ground_truth:
        return 0.0

    top_k = recommended[:k]
    if not top_k:
        return 0.0

    score = 0.0
    num_hits = 0

    for idx, item in enumerate(top_k):
        if item in ground_truth:
            num_hits += 1
            score += num_hits / (idx + 1.0)

    denominator = min(len(ground_truth), k)
    if denominator == 0:
        return 0.0

    return float(score / denominator)


def map_at_k(
    all_recommendations: Sequence[Sequence[Hashable]],
    all_ground_truth: Sequence[Set[Hashable]],
    k: int,
) -> float:
    """Calculate Mean Average Precision@K (MAP@K) across multiple users."""
    validate_k(k)
    if not all_recommendations or not all_ground_truth:
        return 0.0

    if len(all_recommendations) != len(all_ground_truth):
        raise ValueError("Lengths of recommendations and ground truths must match.")

    ap_scores = [
        average_precision_at_k(recs, gt, k)
        for recs, gt in zip(all_recommendations, all_ground_truth)
    ]
    return float(sum(ap_scores) / len(ap_scores)) if ap_scores else 0.0


def ndcg_at_k(recommended: Sequence[Hashable], ground_truth: Set[Hashable], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K (NDCG@K) with binary relevance."""
    validate_k(k)
    if not ground_truth:
        return 0.0

    top_k = recommended[:k]
    if not top_k:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(idx + 2.0)

    # Ideal DCG (IDCG)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(idx + 2.0) for idx in range(ideal_hits))

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def hit_rate_at_k(recommended: Sequence[Hashable], ground_truth: Set[Hashable], k: int) -> float:
    """Calculate Hit Rate@K (1.0 if at least one relevant item is in top-K, else 0.0)."""
    validate_k(k)
    if not ground_truth:
        return 0.0

    top_k = recommended[:k]
    for item in top_k:
        if item in ground_truth:
            return 1.0
    return 0.0


def catalog_coverage(
    all_recommendations: Sequence[Sequence[Hashable]],
    catalog_items: Set[Hashable],
    k: Optional[int] = None,
) -> float:
    """Calculate Catalog Coverage: fraction of catalog items recommended at least once.

    Parameters
    ----------
    all_recommendations : Sequence[Sequence[Hashable]]
        Lists of recommended item IDs for multiple queries/users.
    catalog_items : Set[Hashable]
        Full set of eligible catalog item IDs.
    k : Optional[int], default=None
        If specified, considers only the top-K items per user.

    Returns
    -------
    float
        Coverage ratio in [0.0, 1.0].
    """
    if not catalog_items:
        return 0.0

    recommended_unique: Set[Hashable] = set()
    for recs in all_recommendations:
        top_items = recs[:k] if k is not None else recs
        recommended_unique.update(top_items)

    valid_recommended = recommended_unique.intersection(catalog_items)
    return float(len(valid_recommended) / len(catalog_items))


def evaluate_user_recommendations(
    recommended: Sequence[Hashable],
    ground_truth: Set[Hashable],
    k: int = 5,
) -> Dict[str, float]:
    """Calculate all standard single-user ranking metrics at cutoff K."""
    validate_k(k)
    return {
        f"precision@{k}": precision_at_k(recommended, ground_truth, k),
        f"recall@{k}": recall_at_k(recommended, ground_truth, k),
        f"ap@{k}": average_precision_at_k(recommended, ground_truth, k),
        f"ndcg@{k}": ndcg_at_k(recommended, ground_truth, k),
        f"hit_rate@{k}": hit_rate_at_k(recommended, ground_truth, k),
    }

