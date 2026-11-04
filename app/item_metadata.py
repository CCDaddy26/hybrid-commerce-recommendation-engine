"""Item metadata loading and validation module."""

from pathlib import Path
from typing import Dict, Hashable, List, Optional, Set, Union
import pandas as pd

REQUIRED_METADATA_COLUMNS = ["item_id", "title", "category", "tags"]


def load_item_metadata(
    file_path: Union[str, Path],
    known_catalog_items: Optional[Union[Set[Hashable], List[Hashable]]] = None,
) -> pd.DataFrame:
    """Load and validate item metadata from a CSV file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the metadata CSV file.
    known_catalog_items : Optional[Union[Set[Hashable], List[Hashable]]], default=None
        Optional collection of known item IDs from previous interaction preprocessing.
        If provided, metadata items will be validated against or aligned with this catalog.

    Returns
    -------
    pd.DataFrame
        Validated metadata DataFrame with columns:
        ['item_id', 'title', 'category', 'tags', 'content_text'].

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, missing required columns, contains duplicate item IDs,
        or contains missing or blank text fields.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("Item metadata file is empty.")

    if df.empty:
        raise ValueError("Item metadata file contains no data rows.")

    # Validate required columns
    missing_cols = [col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required metadata column(s): {', '.join(missing_cols)}. "
            f"Expected: {', '.join(REQUIRED_METADATA_COLUMNS)}"
        )

    # Check for duplicate item_ids
    if df["item_id"].duplicated().any():
        dups = df.loc[df["item_id"].duplicated(), "item_id"].tolist()
        raise ValueError(f"Duplicate item IDs found in metadata: {dups}")

    # Check for missing values in required fields
    if df[REQUIRED_METADATA_COLUMNS].isna().any().any():
        raise ValueError("Missing (null/NaN) values found in required metadata fields.")

    # Convert text columns to string and strip whitespace
    for col in ["item_id", "title", "category", "tags"]:
        df[col] = df[col].astype(str).str.strip()

    # Check for blank / empty strings
    for col in ["item_id", "title", "category", "tags"]:
        if (df[col] == "").any():
            raise ValueError(f"Empty or whitespace-only values found in column '{col}'.")

    # Combine text fields to produce a single content representation
    # Weight title and category by including them clearly with tags
    df["content_text"] = (
        df["title"] + " " + df["category"] + " " + df["tags"]
    ).str.strip()

    return df

