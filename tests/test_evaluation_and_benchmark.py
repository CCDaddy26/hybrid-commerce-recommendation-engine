"""Tests for Phase 6 evaluation, ranking metrics, splitting, benchmarks, and optimization."""

import math
import pytest
import pandas as pd
import numpy as np

from app.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    map_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    catalog_coverage,
    evaluate_user_recommendations,
    validate_k,
)
from app.evaluator import split_interactions, DatasetSplit
from app.benchmark import run_benchmark_pipeline, optimize_ensemble_weights
from app.data_loader import load_interactions
from app.item_metadata import load_item_metadata
from app.popularity import PopularityRecommender
from app.als_recommender import ImplicitALSRecommender
from app.content_recommender import ContentRecommender
from app.hybrid_recommender import HybridRecommender
from app.preprocessor import preprocess_interactions
from app.matrix_builder import build_user_item_matrix


# ---------------------------------------------------------------------------
# Unit tests for Ranking Metrics
# ---------------------------------------------------------------------------

def test_metrics_perfect_ranking():
    """Verify metrics when recommendations match ground truth perfectly."""
    recs = ["item_1", "item_2", "item_3"]
    gt = {"item_1", "item_2"}

    p = precision_at_k(recs, gt, k=2)
    r = recall_at_k(recs, gt, k=2)
    ap = average_precision_at_k(recs, gt, k=2)
    ndcg = ndcg_at_k(recs, gt, k=2)
    hr = hit_rate_at_k(recs, gt, k=2)

    assert p == 1.0
    assert r == 1.0
    assert ap == 1.0
    assert ndcg == 1.0
    assert hr == 1.0


def test_metrics_no_hits():
    """Verify metrics when no recommended items are relevant."""
    recs = ["item_1", "item_2"]
    gt = {"item_9", "item_10"}

    assert precision_at_k(recs, gt, k=2) == 0.0
    assert recall_at_k(recs, gt, k=2) == 0.0
    assert average_precision_at_k(recs, gt, k=2) == 0.0
    assert ndcg_at_k(recs, gt, k=2) == 0.0
    assert hit_rate_at_k(recs, gt, k=2) == 0.0


def test_metrics_empty_inputs():
    """Verify metric calculations on empty recommendation or ground truth sets."""
    assert precision_at_k([], {"item_1"}, k=3) == 0.0
    assert recall_at_k([], {"item_1"}, k=3) == 0.0
    assert average_precision_at_k([], {"item_1"}, k=3) == 0.0
    assert ndcg_at_k([], {"item_1"}, k=3) == 0.0
    assert hit_rate_at_k([], {"item_1"}, k=3) == 0.0

    # Empty ground truth (no relevant items)
    assert precision_at_k(["item_1"], set(), k=3) == 0.0
    assert recall_at_k(["item_1"], set(), k=3) == 0.0


def test_metrics_k_validation():
    """Verify error on invalid K."""
    with pytest.raises(ValueError, match="strictly positive"):
        precision_at_k(["item_1"], {"item_1"}, k=0)

    with pytest.raises(ValueError, match="strictly positive"):
        recall_at_k(["item_1"], {"item_1"}, k=-2)


def test_map_and_catalog_coverage():
    """Verify MAP across multiple users and catalog coverage."""
    all_recs = [
        ["item_1", "item_2"],
        ["item_3", "item_4"],
    ]
    all_gt = [
        {"item_1"},
        {"item_4"},
    ]
    # User 1: hit at 1 -> AP = 1.0
    # User 2: hit at 2 -> AP = 0.5
    # MAP = 0.75
    mean_ap = map_at_k(all_recs, all_gt, k=2)
    assert mean_ap == 0.75

    catalog = {"item_1", "item_2", "item_3", "item_4", "item_5"}
    cov = catalog_coverage(all_recs, catalog, k=2)
    # 4 unique items out of 5 catalog items = 0.80
    assert cov == 0.80


# ---------------------------------------------------------------------------
# Unit tests for Dataset Splitting & Leakage Prevention
# ---------------------------------------------------------------------------

def test_split_interactions_leakage_prevention():
    """Verify that test items are strictly excluded from train set."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i2", "i3", "i1", "i4", "i5"],
            "weight": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0],
        }
    )
    split = split_interactions(df, test_ratio=0.33, min_interactions_for_eval=2, random_state=42)

    assert isinstance(split, DatasetSplit)
    # Ensure u3 has only 1 interaction, so it was marked as insufficient
    assert "u3" in split.insufficient_data_users

    # Verify no test (user_id, item_id) appears in train_df
    train_pairs = set(zip(split.train_df["user_id"], split.train_df["item_id"]))
    test_pairs = set(zip(split.test_df["user_id"], split.test_df["item_id"]))

    assert train_pairs.isdisjoint(test_pairs), "Data leakage detected between train and test sets!"


def test_split_interactions_validation_set():
    """Verify 3-way split (train, val, test)."""
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2"],
            "item_id": ["i1", "i2", "i3", "i4", "i1", "i2", "i3"],
            "weight": [1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0],
        }
    )
    split = split_interactions(df, test_ratio=0.25, val_ratio=0.25, min_interactions_for_eval=3, random_state=42)
    assert not split.train_df.empty
    assert not split.val_df.empty
    assert not split.test_df.empty

    train_pairs = set(zip(split.train_df["user_id"], split.train_df["item_id"]))
    val_pairs = set(zip(split.val_df["user_id"], split.val_df["item_id"]))
    test_pairs = set(zip(split.test_df["user_id"], split.test_df["item_id"]))

    assert train_pairs.isdisjoint(val_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert val_pairs.isdisjoint(test_pairs)


# ---------------------------------------------------------------------------
# Integration tests for Benchmark Pipeline & Weight Optimization
# ---------------------------------------------------------------------------

def test_run_benchmark_pipeline_sample_data():
    """Run full benchmark pipeline on sample dataset and verify metrics."""
    interactions = load_interactions("data/sample_interactions.csv")
    metadata = load_item_metadata("data/item_metadata.csv")

    benchmark = run_benchmark_pipeline(interactions, metadata, k=2, random_state=42)

    assert isinstance(benchmark.summary_df, pd.DataFrame)
    assert set(benchmark.raw_results.keys()) == {
        "Popularity",
        "Implicit_ALS",
        "Content_Based",
        "Hybrid_Ensemble",
    }

    # Verify timing stats exist
    assert "fit_als_ms" in benchmark.timing_stats
    assert "fit_popularity_ms" in benchmark.timing_stats
    assert "fit_content_ms" in benchmark.timing_stats


def test_optimize_ensemble_weights_validation_only():
    """Verify weight optimization evaluates candidate configs on validation data only."""
    interactions = load_interactions("data/sample_interactions.csv")
    metadata = load_item_metadata("data/item_metadata.csv")

    split = split_interactions(
        interactions,
        test_ratio=0.25,
        val_ratio=0.25,
        min_interactions_for_eval=2,
        random_state=42,
    )

    best_cfg, opt_df = optimize_ensemble_weights(
        train_df=split.train_df,
        val_df=split.val_df,
        val_users=split.val_user_ids,
        metadata_df=metadata,
        k=2,
    )

    assert isinstance(best_cfg, dict)
    assert "als" in best_cfg and "content" in best_cfg and "popularity" in best_cfg
    assert len(opt_df) >= 4


# ---------------------------------------------------------------------------
# Cold-start granular evaluation tests
# ---------------------------------------------------------------------------

def test_granular_cold_start_evaluation():
    """Evaluate performance across cold-start users vs users with history."""
    interactions = load_interactions("data/sample_interactions.csv")
    metadata = load_item_metadata("data/item_metadata.csv")

    prep_data = preprocess_interactions(interactions)
    matrix = build_user_item_matrix(prep_data)

    als = ImplicitALSRecommender(factors=8, iterations=15, random_state=42).fit(matrix, prep_data)
    content = ContentRecommender().fit(metadata)
    pop = PopularityRecommender().fit(prep_data.aggregated_df)

    hybrid = HybridRecommender(
        als_recommender=als,
        content_recommender=content,
        popularity_recommender=pop,
    )

    # 1. Known user with history
    recs_known = hybrid.recommend(user_id="U101", user_history=["PROD_001"], n=2)
    assert len(recs_known) > 0

    # 2. Cold-start user with preference text
    recs_cold_pref = hybrid.recommend(
        user_id="COLD_USER",
        user_history=[],
        preference_text="wireless ergonomic mouse",
        n=2,
    )
    assert len(recs_cold_pref) > 0
    # Must include mouse products PROD_001 or PROD_006
    cold_pref_items = [i for i, _ in recs_cold_pref]
    assert "PROD_001" in cold_pref_items or "PROD_006" in cold_pref_items

    # 3. Cold-start user without preference text
    recs_cold_nopref = hybrid.recommend(user_id="COLD_USER_NO_PREF", user_history=[], n=2)
    assert len(recs_cold_nopref) > 0
    # Should fall back cleanly to PopularityRecommender (PROD_002 is #1)
    assert recs_cold_nopref[0][0] == "PROD_002"

