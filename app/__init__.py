"""App package exports."""

from app.data_loader import load_interactions
from app.preprocessor import preprocess_interactions, PreprocessedData
from app.popularity import PopularityRecommender
from app.matrix_builder import build_user_item_matrix
from app.als_recommender import ImplicitALSRecommender
from app.item_metadata import load_item_metadata
from app.content_recommender import ContentRecommender
from app.hybrid_recommender import HybridRecommender, HybridRecommendation, normalize_scores
from app.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    map_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    catalog_coverage,
    evaluate_user_recommendations,
)
from app.evaluator import split_interactions, DatasetSplit
from app.benchmark import run_benchmark_pipeline, optimize_ensemble_weights, BenchmarkResult

__all__ = [
    "load_interactions",
    "preprocess_interactions",
    "PreprocessedData",
    "PopularityRecommender",
    "build_user_item_matrix",
    "ImplicitALSRecommender",
    "load_item_metadata",
    "ContentRecommender",
    "HybridRecommender",
    "HybridRecommendation",
    "normalize_scores",
    "precision_at_k",
    "recall_at_k",
    "average_precision_at_k",
    "map_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "catalog_coverage",
    "evaluate_user_recommendations",
    "split_interactions",
    "DatasetSplit",
    "run_benchmark_pipeline",
    "optimize_ensemble_weights",
    "BenchmarkResult",
]
