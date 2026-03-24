"""
hyperparameter_optimization.py — BioScreen-QSAR
Bayesian hyperparameter optimisation using Optuna (Tree-structured Parzen Estimator).

Supported algorithms: Random Forest, SVM, LightGBM
Supported tasks:      Classification, Regression

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier, LGBMRegressor

from scripts.utils import configure_logger

logger = configure_logger(__name__)

# Suppress Optuna's verbose per-trial output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Search space definitions ──────────────────────────────────────────────────

def _rf_params(trial: optuna.Trial, task: str) -> dict:
    """Random Forest hyperparameter search space."""
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 30),
        "min_samples_split":trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight":     trial.suggest_categorical("class_weight", [None, "balanced"])
                            if task == "classification" else None,
        "n_jobs":           -1,
        "random_state":     42,
    }


def _svm_params(trial: optuna.Trial, task: str) -> dict:
    """Support Vector Machine hyperparameter search space."""
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    params: dict = {
        "C":       trial.suggest_float("C", 1e-3, 1e3, log=True),
        "kernel":  kernel,
    }
    if kernel in ("rbf", "poly"):
        params["gamma"] = trial.suggest_float("gamma", 1e-5, 1e1, log=True)
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    if task == "classification":
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        )
        params["probability"] = True
    return params


def _lgbm_params(trial: optuna.Trial, task: str) -> dict:
    """LightGBM hyperparameter search space."""
    return {
        "n_estimators":    trial.suggest_int("n_estimators", 50, 1000, step=50),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
        "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth":       trial.suggest_int("max_depth", 3, 12),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 1.0),
        "n_jobs":          -1,
        "random_state":    42,
        "verbosity":       -1,
        "is_unbalance":    True if task == "classification" else False,
    }


# ── Optimiser class ───────────────────────────────────────────────────────────

class BayesianOptimiser:
    """
    Bayesian hyperparameter optimiser wrapping Optuna.

    Parameters
    ----------
    algorithm : {'rf', 'svm', 'lgbm'}
        Which ML algorithm to optimise.
    task : {'classification', 'regression'}
        Whether the endpoint is binary or continuous.
    n_trials : int
        Number of Optuna trials (evaluations of the objective function).
    cv_folds : int
        Number of cross-validation folds used to evaluate each trial.
    scoring : str
        scikit-learn scoring string. Defaults are 'roc_auc' (classification)
        and 'r2' (regression).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        algorithm: str = "lgbm",
        task: str = "classification",
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.task = task.lower()
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

        if scoring is None:
            self.scoring = "roc_auc" if self.task == "classification" else "r2"
        else:
            self.scoring = scoring

        self.best_params_: dict = {}
        self.best_score_: float = float("-inf")
        self.study_: optuna.Study | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def optimise(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict:
        """
        Run Bayesian optimisation and return the best hyperparameters found.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training labels/values.

        Returns
        -------
        dict
            Best hyperparameter configuration.
        """
        logger.info(
            f"Starting Bayesian optimisation: algorithm={self.algorithm}, "
            f"task={self.task}, n_trials={self.n_trials}, scoring={self.scoring}."
        )

        self.study_ = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        self.study_.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        logger.info(
            f"Optimisation complete. Best {self.scoring} = {self.best_score_:.4f}. "
            f"Best params: {self.best_params_}"
        )
        return self.best_params_

    def build_best_model(self):
        """
        Instantiate and return the best model using the optimised parameters.

        Returns
        -------
        Fitted-ready scikit-learn / LightGBM estimator.

        Raises
        ------
        RuntimeError
            If optimise() has not been called yet.
        """
        if not self.best_params_:
            raise RuntimeError("Call optimise() before build_best_model().")

        params = dict(self.best_params_)  # copy to avoid modifying study params

        if self.algorithm == "rf":
            cls = RandomForestClassifier if self.task == "classification" else RandomForestRegressor
            params.setdefault("n_jobs", -1)
            params.setdefault("random_state", self.random_state)
            params.pop("class_weight", None) if self.task == "regression" else None
            return cls(**{k: v for k, v in params.items() if k in cls().get_params()})

        if self.algorithm == "svm":
            cls = SVC if self.task == "classification" else SVR
            if self.task == "classification":
                params["probability"] = True
            return cls(**{k: v for k, v in params.items() if k in cls().get_params()})

        if self.algorithm == "lgbm":
            cls = LGBMClassifier if self.task == "classification" else LGBMRegressor
            params.setdefault("random_state", self.random_state)
            params.setdefault("verbosity", -1)
            return cls(**{k: v for k, v in params.items() if k in cls().get_params()})

        raise ValueError(f"Unknown algorithm: '{self.algorithm}'.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Optuna objective function — returns mean CV score for a trial."""
        param_fn = {
            "rf":   _rf_params,
            "svm":  _svm_params,
            "lgbm": _lgbm_params,
        }.get(self.algorithm)

        if param_fn is None:
            raise ValueError(f"Unknown algorithm: '{self.algorithm}'.")

        params = param_fn(trial, self.task)

        # Build a temporary model for cross-validation
        if self.algorithm == "rf":
            cls = RandomForestClassifier if self.task == "classification" else RandomForestRegressor
        elif self.algorithm == "svm":
            cls = SVC if self.task == "classification" else SVR
            if self.task == "classification":
                params["probability"] = True
        else:
            cls = LGBMClassifier if self.task == "classification" else LGBMRegressor

        model = cls(**{k: v for k, v in params.items() if k in cls().get_params()})

        try:
            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=-1,
            )
            return float(np.mean(scores))
        except Exception as exc:
            logger.debug(f"Trial failed: {exc}")
            return float("-inf")
