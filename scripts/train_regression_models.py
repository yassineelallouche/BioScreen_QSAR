"""
train_regression_models.py — BioScreen-QSAR
Training pipeline for continuous regression QSAR models (pMIC prediction).

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate, KFold

from scripts.validation_metrics import regression_metrics, metrics_to_dataframe
from scripts.hyperparameter_optimization import BayesianOptimiser
from scripts.model_serialization import save_model
from scripts.utils import configure_logger

logger = configure_logger(__name__)


# ── Default model catalogue ───────────────────────────────────────────────────

DEFAULT_REGRESSORS = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    ),
    "SVM": SVR(
        C=1.0,
        kernel="rbf",
        gamma="scale",
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    ),
}


def train_regressors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    algorithms: list[str] | None = None,
    optimise: bool = True,
    n_trials: int = 50,
    n_splits: int = 5,
    model_dir: str = "models",
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Train, optimise, and evaluate regression QSAR models.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices.
    y_train, y_test : np.ndarray
        Continuous activity values (e.g., pMIC).
    algorithms : list[str], optional
        Which algorithms to train; defaults to all three.
    optimise : bool
        Whether to perform Bayesian hyperparameter search.
    n_trials : int
        Number of Optuna trials.
    n_splits : int
        Cross-validation fold count.
    model_dir : str
        Directory for serialised model files.
    results_dir : str
        Directory for result CSVs.

    Returns
    -------
    pd.DataFrame
        Summary of regression metrics per model.
    """
    if algorithms is None:
        algorithms = list(DEFAULT_REGRESSORS.keys())

    all_metrics = []

    for algo_name in algorithms:
        logger.info(f"\n{'='*55}")
        logger.info(f"  Training regressor: {algo_name}")
        logger.info(f"{'='*55}")

        # ── Hyperparameter optimisation ───────────────────────────────────
        if optimise:
            opt = BayesianOptimiser(
                algorithm=algo_name.lower().replace("lightgbm", "lgbm"),
                task="regression",
                n_trials=n_trials,
                cv_folds=n_splits,
                scoring="r2",
            )
            opt.optimise(X_train, y_train)
            model = opt.build_best_model()
        else:
            model = DEFAULT_REGRESSORS.get(algo_name)
            if model is None:
                logger.warning(f"Unknown algorithm '{algo_name}'. Skipping.")
                continue

        # ── Cross-validation ──────────────────────────────────────────────
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
            return_train_score=False,
            n_jobs=-1,
        )
        cv_r2   = cv_scores["test_r2"].mean()
        cv_rmse = -cv_scores["test_neg_root_mean_squared_error"].mean()
        logger.info(
            f"  CV R²   = {cv_r2:.4f} ± {cv_scores['test_r2'].std():.4f}"
        )
        logger.info(f"  CV RMSE = {cv_rmse:.4f}")

        # ── Final fit ─────────────────────────────────────────────────────
        model.fit(X_train, y_train)

        # ── External test evaluation ──────────────────────────────────────
        y_pred = model.predict(X_test)
        ext_metrics = regression_metrics(y_test, y_pred)
        logger.info(
            f"  External Test — R²={ext_metrics['r2']}, "
            f"RMSE={ext_metrics['rmse']}, MAE={ext_metrics['mae']}"
        )

        # ── Save model ────────────────────────────────────────────────────
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/regressor_{algo_name.lower()}.pkl"
        save_model(model, model_path)

        # ── Accumulate results ────────────────────────────────────────────
        row_metrics = {
            "cv_r2_mean":   round(cv_r2, 4),
            "cv_r2_std":    round(cv_scores["test_r2"].std(), 4),
            "cv_rmse_mean": round(cv_rmse, 4),
        }
        row_metrics.update(ext_metrics)
        all_metrics.append(metrics_to_dataframe(row_metrics, model_name=algo_name))

    # ── Save aggregated results ───────────────────────────────────────────────
    if all_metrics:
        results_df = pd.concat(all_metrics, ignore_index=True)
        os.makedirs(results_dir, exist_ok=True)
        out_path = f"{results_dir}/regression_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"Regression results saved to '{out_path}'.")
        return results_df

    return pd.DataFrame()
