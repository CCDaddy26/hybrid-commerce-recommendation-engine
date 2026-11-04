"""Implicit-feedback ALS collaborative filtering recommender module.

Trains an Alternating Least Squares (ALS) model on user-item interaction data
using the `implicit` library.
"""

from typing import Any, Dict, Hashable, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import implicit.als

from app.preprocessor import PreprocessedData


class ImplicitALSRecommender:
    """Collaborative filtering recommender using implicit Alternating Least Squares.

    Parameters
    ----------
    factors : int, default=32
        Number of latent factors to compute.
    regularization : float, default=0.05
        Regularization parameter to prevent overfitting.
    iterations : int, default=15
        Number of ALS iterations to run during fitting.
    random_state : Optional[int], default=42
        Seed for random initialization of latent factor matrices.
    alpha : float, default=1.0
        Linear confidence scaling parameter (C = 1 + alpha * r).
    """

    def __init__(
        self,
        factors: int = 32,
        regularization: float = 0.05,
        iterations: int = 15,
        random_state: Optional[int] = 42,
        alpha: float = 1.0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.alpha = alpha

        # Internal model and mapping references
        self.model: Optional[implicit.als.AlternatingLeastSquares] = None
        self.user_to_idx: Dict[Hashable, int] = {}
        self.idx_to_item: Dict[int, Hashable] = {}
        self.user_item_matrix: Optional[sp.csr_matrix] = None
        self.is_fitted: bool = False

    def fit(
        self,
        user_item_matrix: sp.csr_matrix,
        preprocessed_data: PreprocessedData,
        show_progress: bool = False,
    ) -> "ImplicitALSRecommender":
        """Train the ALS model on a user-item CSR matrix.

        Matrix Orientation Note:
        - `user_item_matrix` is expected in (num_users, num_items) orientation where
          rows are users and columns are items.
        - In modern versions of the `implicit` library (v0.6+ and v0.7+), `model.fit()`
          expects `user_items` directly with shape (users, items). If an older version
          or different orientation convention is encountered, transpose accordingly.
          Here, `user_item_matrix` of shape (users, items) is passed directly.

        Parameters
        ----------
        user_item_matrix : sp.csr_matrix
            Sparse CSR matrix of shape (num_users, num_items).
        preprocessed_data : PreprocessedData
            Data container holding user_to_idx and idx_to_item mappings.
        show_progress : bool, default=False
            Whether to display the progress bar during training.

        Returns
        -------
        ImplicitALSRecommender
            The fitted recommender instance.

        Raises
        ------
        ValueError
            If matrix is empty, not CSR format, or has shape mismatch with mappings.
        """
        if not sp.isspmatrix_csr(user_item_matrix):
            raise ValueError("user_item_matrix must be a scipy.sparse.csr_matrix.")

        num_users, num_items = user_item_matrix.shape
        if num_users == 0 or num_items == 0 or user_item_matrix.nnz == 0:
            raise ValueError("Cannot train ALS on an empty matrix.")

        if num_users != len(preprocessed_data.user_to_idx):
            raise ValueError(
                f"Matrix row count ({num_users}) does not match number of users in mappings ({len(preprocessed_data.user_to_idx)})."
            )

        if num_items != len(preprocessed_data.idx_to_item):
            raise ValueError(
                f"Matrix column count ({num_items}) does not match number of items in mappings ({len(preprocessed_data.idx_to_item)})."
            )

        # Store references
        self.user_to_idx = preprocessed_data.user_to_idx
        self.idx_to_item = preprocessed_data.idx_to_item
        self.user_item_matrix = user_item_matrix

        # Instantiate implicit's ALS model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            alpha=self.alpha,
        )

        # Fit on user_items matrix
        self.model.fit(self.user_item_matrix, show_progress=show_progress)
        self.is_fitted = True
        return self

    def recommend(
        self,
        user_id: Hashable,
        n: int = 5,
        filter_already_liked_items: bool = True,
    ) -> List[Tuple[Hashable, float]]:
        """Generate top-N recommendations for a specified user.

        Items the user has already interacted with are excluded by default.
        Handles unknown users gracefully by returning an empty list (caller can fall back to popularity).
        Safe against n <= 0 and n larger than available unobserved items.

        Parameters
        ----------
        user_id : Hashable
            Identifier of the user to recommend items for.
        n : int, default=5
            Maximum number of recommendations to return.
        filter_already_liked_items : bool, default=True
            Whether to exclude items present in user's interaction history.

        Returns
        -------
        List[Tuple[Hashable, float]]
            List of (item_id, predicted_score) tuples ordered by score descending.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted.
        """
        if not self.is_fitted or self.model is None or self.user_item_matrix is None:
            raise RuntimeError("ImplicitALSRecommender must be fitted before calling recommend().")

        if n <= 0:
            return []

        # Cold-start or unknown user: return empty list
        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        user_row = self.user_item_matrix[user_idx]

        num_items = len(self.idx_to_item)
        seen_items_count = user_row.nnz if filter_already_liked_items else 0
        unseen_items_count = num_items - seen_items_count

        if unseen_items_count <= 0:
            return []

        request_n = min(n, unseen_items_count)

        item_indices, scores = self.model.recommend(
            userid=user_idx,
            user_items=user_row,
            N=request_n,
            filter_already_liked_items=filter_already_liked_items,
        )

        results: List[Tuple[Hashable, float]] = []
        for raw_item_idx, score in zip(item_indices, scores):
            item_idx = int(raw_item_idx)
            # Filter out implicit's sentinel unranked scores (-3.4e38) if present
            if score < -1e30:
                continue
            item_id = self.idx_to_item.get(item_idx)
            if item_id is not None:
                results.append((item_id, float(score)))

        return results

