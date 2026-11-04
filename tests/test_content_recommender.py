"""Tests for item_metadata and content_recommender modules."""

import pytest
import pandas as pd
from app.item_metadata import load_item_metadata, REQUIRED_METADATA_COLUMNS
from app.content_recommender import ContentRecommender


def test_load_item_metadata_sample_file():
    """Verify loading and validating the sample item metadata CSV."""
    df = load_item_metadata("data/item_metadata.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    for col in REQUIRED_METADATA_COLUMNS:
        assert col in df.columns
    assert "content_text" in df.columns
    assert not df["content_text"].isna().any()

    # Validate catalog includes PROD_001 to PROD_006
    item_ids = list(df["item_id"])
    assert "PROD_001" in item_ids
    assert "PROD_006" in item_ids  # Cold-start item with zero interactions in sample_interactions.csv


def test_load_item_metadata_file_not_found():
    """Verify FileNotFoundError when metadata file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_item_metadata("data/non_existent_metadata.csv")


def test_load_item_metadata_empty_file(tmp_path):
    """Verify error on completely empty metadata file."""
    f = tmp_path / "empty.csv"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_item_metadata(f)


def test_load_item_metadata_missing_columns(tmp_path):
    """Verify error when required columns are missing."""
    f = tmp_path / "bad_cols.csv"
    f.write_text("item_id,title\nP1,Sample\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required metadata column"):
        load_item_metadata(f)


def test_load_item_metadata_duplicate_ids(tmp_path):
    """Verify error on duplicate item IDs."""
    f = tmp_path / "dups.csv"
    f.write_text(
        "item_id,title,category,tags\nP1,Title 1,Cat1,tag1\nP1,Title 2,Cat2,tag2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate item IDs"):
        load_item_metadata(f)


def test_load_item_metadata_blank_fields(tmp_path):
    """Verify error when text fields are blank or whitespace-only."""
    f = tmp_path / "blank.csv"
    f.write_text(
        "item_id,title,category,tags\nP1,   ,Cat1,tag1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Empty or whitespace-only values"):
        load_item_metadata(f)


def test_content_recommender_basic_similarity():
    """Verify TF-IDF fitting and similarity ranking."""
    metadata = load_item_metadata("data/item_metadata.csv")
    recommender = ContentRecommender().fit(metadata)
    assert recommender.is_fitted
    assert recommender.tfidf_matrix.shape[0] == 6

    # PROD_001 is "Wireless Ergonomic Mouse"
    # The most similar item should be PROD_006 ("Ergonomic Vertical Mouse")
    recs = recommender.recommend_similar_items("PROD_001", n=2)
    assert len(recs) == 2
    top_item_id, top_score = recs[0]
    assert top_item_id == "PROD_006"
    assert top_score > 0.0

    # Ensure seed is excluded
    rec_ids = [item_id for item_id, _ in recs]
    assert "PROD_001" not in rec_ids


def test_content_recommender_recommends_cold_start_item():
    """Verify content recommendations can surface a new item (PROD_006) with zero interaction history."""
    metadata = load_item_metadata("data/item_metadata.csv")
    recommender = ContentRecommender().fit(metadata)

    # For user whose history includes PROD_001 (mouse), recommend_for_user should recommend PROD_006
    user_recs = recommender.recommend_for_user(user_history_items=["PROD_001"], n=3)
    rec_ids = [item_id for item_id, _ in user_recs]

    # PROD_006 is the vertical mouse (uninteracted in interaction CSV)
    assert "PROD_006" in rec_ids
    assert "PROD_001" not in rec_ids


def test_content_recommender_multiple_seeds():
    """Verify recommendations from multiple seed items."""
    metadata = load_item_metadata("data/item_metadata.csv")
    recommender = ContentRecommender().fit(metadata)

    # Seeds PROD_003 (Headphones) and PROD_004 (Speaker)
    recs = recommender.recommend_similar_items(["PROD_003", "PROD_004"], n=3)
    rec_ids = [item_id for item_id, _ in recs]

    assert "PROD_003" not in rec_ids
    assert "PROD_004" not in rec_ids
    assert len(recs) <= 3


def test_content_recommender_cold_start_user_behavior():
    """Verify that a cold-start user without history does not get pseudo-personalized recs."""
    metadata = load_item_metadata("data/item_metadata.csv")
    recommender = ContentRecommender().fit(metadata)

    # 1. User with no history and no preference text returns empty list
    # (signals to caller to use popularity fallback)
    empty_recs = recommender.recommend_for_user(user_history_items=[], preference_text=None)
    assert empty_recs == []

    # 2. User with no history but explicit query text gets relevant category recommendations
    audio_recs = recommender.recommend_for_user(
        user_history_items=[], preference_text="noise cancelling audio headphones"
    )
    assert len(audio_recs) > 0
    top_id, _ = audio_recs[0]
    assert top_id == "PROD_003"


def test_content_recommender_edge_cases():
    """Verify edge cases: unfitted model, unknown seeds, n <= 0, large n."""
    unfitted = ContentRecommender()
    with pytest.raises(RuntimeError, match="must be fitted"):
        unfitted.recommend_similar_items("PROD_001")

    metadata = load_item_metadata("data/item_metadata.csv")
    fitted = ContentRecommender().fit(metadata)

    # Unknown seed ID returns empty list
    assert fitted.recommend_similar_items("NON_EXISTENT_ID") == []

    # n <= 0 returns empty list
    assert fitted.recommend_similar_items("PROD_001", n=0) == []
    assert fitted.recommend_similar_items("PROD_001", n=-5) == []

    # n > available catalog returns all remaining items
    all_recs = fitted.recommend_similar_items("PROD_001", n=100)
    assert len(all_recs) == 5  # 6 items total minus 1 seed
    assert "PROD_001" not in [i for i, _ in all_recs]


def test_content_recommender_invalid_fit():
    """Verify error handling on invalid fit inputs."""
    with pytest.raises(ValueError, match="empty metadata"):
        ContentRecommender().fit(pd.DataFrame())

    with pytest.raises(ValueError, match="missing required columns"):
        ContentRecommender().fit(pd.DataFrame({"item_id": ["P1"]}))

