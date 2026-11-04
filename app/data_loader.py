"""Data loading and validation module for commerce interactions."""

from pathlib import Path
from typing import Union
import pandas as pd

REQUIRED_COLUMNS = ["user_id", "item_id", "event_type", "weight"]


def load_interactions(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate interaction data from a CSV file.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the interactions CSV file.

    Returns
    -------
    pd.DataFrame
        Validated interactions DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, missing required columns, or contains invalid weights.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("Interactions file is empty.")

    if df.empty:
        raise ValueError("Interactions file contains no data rows.")

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )

    # Validate weight values (must be numeric and strictly positive)
    numeric_weights = pd.to_numeric(df["weight"], errors="coerce")
    if numeric_weights.isna().any():
        raise ValueError("The 'weight' column contains non-numeric or missing values.")

    if (numeric_weights <= 0).any():
        raise ValueError("The 'weight' column must contain strictly positive values (> 0).")

    # Clean missing values in IDs or event type
    if df[["user_id", "item_id", "event_type"]].isna().any().any():
        raise ValueError("Missing values found in user_id, item_id, or event_type columns.")

    df["weight"] = numeric_weights
    return df

