"""Benchmark and evaluation pipeline for recommendation systems.

Runs systematic comparisons across Popularity, ALS, Content, and Hybrid models,
optimizes ensemble weights using validation data, performs cold-start evaluation,
and benchmarks execution runtime and latency.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple
import pandas as pd

from app.als_recommender import ImplicitALSRecommender
from app.content_recommender import ContentRecommender
from app.evaluator import DatasetSplit, split_interactions
from app.hybrid_recommender import HybridRecommender
from app.item_metadata import load_item_metadata
from app.matrix_builder import build_user_item_matrix
from app.metrics import (
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    validate_k,
)
from app.popularity import PopularityRecommender
from app.preprocessor import PreprocessedData, preprocess_interactions


@dataclass
class BenchmarkResult:
    """Benchmark evaluation summary across models."""

    summary_df: pd.DataFrame
    raw_results: Dict[str, Dict[str, float]]
    timing_stats: Dict[str, float]


def evaluate_model_on_split(
    model: Any,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_users: List[Hashable],
    all_catalog_items: Set[Hashable],
    k: int = 3,
) -> Dict[str, float]:
    """Evaluate a single model instance across a specified test set.

    Computes: Precision@K, Recall@K, MAP@K, NDCG@K, Hit_Rate@K, Catalog_Coverage@K.
    """
    validate_k(k)
    precisions = []
    recalls = []
    aps = []
    ndcgs = []
    hit_rates = []
    all_recommendations = []

    # Map user to historical training items for exclusion
    train_history: Dict[Hashable, Set[Hashable]] = (
        train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )
    # Map user to ground truth test items
    test_ground_truth: Dict[Hashable, Set[Hashable]] = (
        test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    for user in eval_users:
        gt = test_ground_truth.get(user, set())
        if not gt:
            continue

        history = train_history.get(user, set())

        # Generate recommendations based on model type
        rec_ids: List[Hashable] = []
        if isinstance(model, HybridRecommender):
            raw = model.recommend(user_id=user, user_history=history, n=k)
            rec_ids = [item_id for item_id, _ in raw]
        elif isinstance(model, ImplicitALSRecommender):
            raw = model.recommend(user_id=user, n=k, filter_already_liked_items=True)
            rec_ids = [item_id for item_id, _ in raw]
        elif isinstance(model, ContentRecommender):
            raw = model.recommend_for_user(user_history_items=history, n=k)
            rec_ids = [item_id for item_id, _ in raw]
        elif isinstance(model, PopularityRecommender):
            raw = model.recommend(n=len(all_catalog_items))
            # Exclude history manually for popularity
            filtered = [item_id for item_id, _ in raw if item_id not in history]
            rec_ids = filtered[:k]

        all_recommendations.append(rec_ids)
        precisions.append(precision_at_k(rec_ids, gt, k))
        recalls.append(recall_at_k(rec_ids, gt, k))
        aps.append(average_precision_at_k(rec_ids, gt, k))
        ndcgs.append(ndcg_at_k(rec_ids, gt, k))
        hit_rates.append(hit_rate_at_k(rec_ids, gt, k))

    n_users = len(precisions)
    if n_users == 0:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"map@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            f"catalog_coverage@{k}": 0.0,
        }

    return {
        f"precision@{k}": float(sum(precisions) / n_users),
        f"recall@{k}": float(sum(recalls) / n_users),
        f"map@{k}": float(sum(aps) / n_users),
        f"ndcg@{k}": float(sum(ndcgs) / n_users),
        f"hit_rate@{k}": float(sum(hit_rates) / n_users),
        f"catalog_coverage@{k}": catalog_coverage(all_recommendations, all_catalog_items, k=k),
    }


def run_benchmark_pipeline(
    interactions_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    k: int = 3,
    random_state: int = 42,
) -> BenchmarkResult:
    """Run full benchmark comparing Popularity, ALS, Content, and Hybrid on held-out test data.

    Follows rigorous protocol:
    1. Splits interaction data into Train (80%) and Test (20%).
    2. Fits all models strictly on `train_df`.
    3. Evaluates all models on identical test users and unseen ground truth.
    4. Measures model fitting time and recommendation generation latency.
    """
    timing_stats: Dict[str, float] = {}

    # Split dataset
    split = split_interactions(
        interactions_df,
        test_ratio=0.25,
        val_ratio=0.0,
        min_interactions_for_eval=2,
        random_state=random_state,
    )

    all_catalog_items = set(metadata_df["item_id"])

    # 1. Preprocess train data only
    t0 = time.perf_counter()
    train_prep = preprocess_interactions(split.train_df)
    train_matrix = build_user_item_matrix(train_prep)

    # 2. Fit Popularity
    t_pop_start = time.perf_counter()
    pop_model = PopularityRecommender().fit(train_prep.aggregated_df)
    timing_stats["fit_popularity_ms"] = (time.perf_counter() - t_pop_start) * 1000

    # 3. Fit ALS
    t_als_start = time.perf_counter()
    als_model = ImplicitALSRecommender(factors=8, iterations=15, random_state=random_state)
    als_model.fit(train_matrix, train_prep)
    timing_stats["fit_als_ms"] = (time.perf_counter() - t_als_start) * 1000

    # 4. Fit Content
    t_cnt_start = time.perf_counter()
    content_model = ContentRecommender().fit(metadata_df)
    timing_stats["fit_content_ms"] = (time.perf_counter() - t_cnt_start) * 1000

    # 5. Build Hybrid
    hybrid_model = HybridRecommender(
        als_recommender=als_model,
        content_recommender=content_model,
        popularity_recommender=pop_model,
        weights={"als": 0.50, "content": 0.35, "popularity": 0.15},
    )

    models = {
        "Popularity": pop_model,
        "Implicit_ALS": als_model,
        "Content_Based": content_model,
        "Hybrid_Ensemble": hybrid_model,
    }

    raw_results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        t_eval_start = time.perf_counter()
        metrics = evaluate_model_on_split(
            model=model,
            model_name=name,
            train_df=split.train_df,
            test_df=split.test_df,
            eval_users=split.eval_user_ids,
            all_catalog_items=all_catalog_items,
            k=k,
        )
        timing_stats[f"eval_{name.lower()}_ms"] = (time.perf_counter() - t_eval_start) * 1000
        raw_results[name] = metrics

    # Format into DataFrame
    summary_df = pd.DataFrame(raw_results).T
    return BenchmarkResult(summary_df=summary_df, raw_results=raw_results, timing_stats=timing_stats)


def optimize_ensemble_weights(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_users: List[Hashable],
    metadata_df: pd.DataFrame,
    candidate_weight_configs: Optional[List[Dict[str, float]]] = None,
    k: int = 3,
    random_state: int = 42,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Tune hybrid ensemble weights using validation data strictly (no test set leakage).

    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame]
        The best performing weight configuration and the comparison DataFrame.
    """
    if candidate_weight_configs is None:
        candidate_weight_configs = [
            {"als": 0.50, "content": 0.35, "popularity": 0.15},  # Default baseline
            {"als": 0.70, "content": 0.20, "popularity": 0.10},  # ALS-heavy
            {"als": 0.20, "content": 0.70, "popularity": 0.10},  # Content-heavy
            {"als": 0.20, "content": 0.20, "popularity": 0.60},  # Popularity-heavy
            {"als": 0.33, "content": 0.33, "popularity": 0.34},  # Balanced equal
        ]

    train_prep = preprocess_interactions(train_df)
    train_matrix = build_user_item_matrix(train_prep)

    als = ImplicitALSRecommender(factors=8, iterations=15, random_state=random_state).fit(train_matrix, train_prep)
    content = ContentRecommender().fit(metadata_df)
    pop = PopularityRecommender().fit(train_prep.aggregated_df)

    all_catalog_items = set(metadata_df["item_id"])

    records = []
    best_config = candidate_weight_configs[0]
    best_ndcg = -1.0

    for idx, cfg in enumerate(candidate_weight_configs):
        hybrid = HybridRecommender(
            als_recommender=als,
            content_recommender=content,
            popularity_recommender=pop,
            weights=cfg,
        )
        res = evaluate_model_on_split(
            model=hybrid,
            model_name=f"Config_{idx}",
            train_df=train_df,
            test_df=val_df,
            eval_users=val_users,
            all_catalog_items=all_catalog_items,
            k=k,
        )
        record = {
            "config_idx": idx,
            "weights": str(cfg),
            f"ndcg@{k}": res[f"ndcg@{k}"],
            f"map@{k}": res[f"map@{k}"],
            f"precision@{k}": res[f"precision@{k}"],
            f"recall@{k}": res[f"recall@{k}"],
            f"hit_rate@{k}": res[f"hit_rate@{k}"],
            f"catalog_coverage@{k}": res[f"catalog_coverage@{k}"],
        }
        records.append(record)

        if res[f"ndcg@{k}"] > best_ndcg:
            best_ndcg = res[f"ndcg@{k}"]
            best_config = cfg

    opt_df = pd.DataFrame(records)
    return best_config, opt_df

