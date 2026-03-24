"""
preprocessing.py — BioScreen-QSAR
Feature matrix preparation: variance filtering, train/test split,
and optional feature selection prior to model training.

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.feature_selection import VarianceThreshold

from scripts.utils import configure_logger

logger = configure_logger(__name__)


# ── Preprocessing class ───────────────────────────────────────────────────────

class FeaturePreprocessor:
    """
    Prepares the ECFP descriptor matrix for model training.

    Steps:
      1. Remove constant / near-constant bit columns (variance threshold).
      2. Optionally apply feature selection (low-variance removal is default).
      3. Split dataset into training and held-out test sets.
      4. Provide cross-validation splitter objects.

    Parameters
    ----------
    variance_threshold : float
        Minimum variance required to retain a bit column. Columns at or
        below this value are removed. Default=0.0 removes zero-variance
        (constant) columns only.
    test_size : float
        Proportion of the dataset reserved for the held-out test set.
    n_splits : int
        Number of folds for k-fold cross-validation.
    random_state : int
        Seed for reproducibility.
    stratify : bool
        Whether to use stratified splitting (classification only).
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        test_size: float = 0.20,
        n_splits: int = 5,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.stratify = stratify

        self._var_filter: VarianceThreshold | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply variance filtering and produce train / test splits.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            ECFP fingerprint matrix.
        y : np.ndarray, shape (n_samples,)
            Biological activity labels or values.

        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
        """
        logger.info(f"Input feature matrix: {X.shape}")

        # Step 1: Remove near-constant features
        X = self._apply_variance_filter(X, fit=True)

        # Step 2: Train / test split
        stratify_labels = y if self.stratify else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_labels,
            )
        except ValueError:
            # Fall back to non-stratified split if stratification fails
            logger.warning(
                "Stratified split failed (possibly too few samples per class). "
                "Falling back to random split."
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None,
            )

        logger.info(
            f"Train: {X_train.shape[0]} samples | "
            f"Test: {X_test.shape[0]} samples | "
            f"Features after filtering: {X_train.shape[1]}"
        )
        return X_train, X_test, y_train, y_test

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the already-fitted variance filter to a new feature matrix
        (e.g., a virtual screening library).

        Parameters
        ----------
        X : np.ndarray
            Raw fingerprint matrix for new compounds.

        Returns
        -------
        np.ndarray
            Filtered feature matrix with the same columns as the training set.
        """
        if self._var_filter is None:
            raise RuntimeError("Call fit_transform() before transform().")
        return self._var_filter.transform(X)

    def get_cv_splitter(self, task: str = "classification"):
        """
        Return a scikit-learn cross-validation splitter.

        Parameters
        ----------
        task : {'classification', 'regression'}
            For classification, StratifiedKFold is returned to maintain
            class-balance across folds; KFold otherwise.

        Returns
        -------
        StratifiedKFold or KFold
        """
        if task == "classification":
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_variance_filter(
        self, X: np.ndarray, fit: bool = True
    ) -> np.ndarray:
        """Remove columns with variance ≤ self.variance_threshold."""
        if fit:
            self._var_filter = VarianceThreshold(
                threshold=self.variance_threshold
            )
            X_filtered = self._var_filter.fit_transform(X)
        else:
            X_filtered = self._var_filter.transform(X)

        n_removed = X.shape[1] - X_filtered.shape[1]
        logger.info(
            f"Variance filter: removed {n_removed} near-constant columns. "
            f"Remaining features: {X_filtered.shape[1]}."
        )
        return X_filtered


# ── Convenience function ──────────────────────────────────────────────────────

def prepare_matrices(
    descriptor_path: str,
    smiles_col: str = "curated_SMILES",
    activity_col: str = "activity",
    test_size: float = 0.20,
    n_splits: int = 5,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    FeaturePreprocessor
]:
    """
    Full preprocessing pipeline: load descriptors → filter → split.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor
    """
    df = pd.read_csv(descriptor_path)

    # Isolate bit columns (all columns that are not metadata)
    bit_cols = [c for c in df.columns if c not in [smiles_col, activity_col]]
    X = df[bit_cols].values.astype(np.float32)
    y = df[activity_col].values

    preprocessor = FeaturePreprocessor(
        test_size=test_size,
        n_splits=n_splits,
        random_state=random_state,
        stratify=stratify,
    )
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
    return X_train, X_test, y_train, y_test, preprocessor
