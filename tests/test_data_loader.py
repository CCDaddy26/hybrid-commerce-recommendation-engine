"""Tests for data_loader module."""

import pytest
import pandas as pd
from app.data_loader import load_interactions, REQUIRED_COLUMNS


def test_load_sample_interactions_success():
    """Test loading the default sample interactions CSV file."""
    df = load_interactions("data/sample_interactions.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert (df["weight"] > 0).all()
    assert len(df) == 12


def test_file_not_found():
    """Test error when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_interactions("data/non_existent_file.csv")


def test_empty_file(tmp_path):
    """Test error when CSV file is completely empty."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_interactions(empty_file)


def test_missing_required_columns(tmp_path):
    """Test error when required columns are absent."""
    csv_file = tmp_path / "missing_cols.csv"
    csv_file.write_text("user_id,item_id\nU1,PROD_1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required column"):
        load_interactions(csv_file)


def test_invalid_weight_values(tmp_path):
    """Test error when weights contain negative numbers or zero."""
    csv_file = tmp_path / "invalid_weights.csv"
    csv_file.write_text(
        "user_id,item_id,event_type,weight\nU1,PROD_1,view,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly positive"):
        load_interactions(csv_file)


def test_non_numeric_weight(tmp_path):
    """Test error when weight is non-numeric."""
    csv_file = tmp_path / "non_numeric_weight.csv"
    csv_file.write_text(
        "user_id,item_id,event_type,weight\nU1,PROD_1,view,five\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-numeric or missing"):
        load_interactions(csv_file)


def test_missing_identifier_values(tmp_path):
    """Test error when user_id or item_id has null values."""
    csv_file = tmp_path / "null_ids.csv"
    csv_file.write_text(
        "user_id,item_id,event_type,weight\n,PROD_1,view,1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing values found"):
        load_interactions(csv_file)

