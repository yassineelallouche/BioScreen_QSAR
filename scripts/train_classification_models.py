"""
train_classification_models.py — BioScreen-QSAR
Training pipeline for binary classification QSAR models.

Supported algorithms: Random Forest (RF), Support Vector Machine (SVM),
                      LightGBM (LGBM)
Validation strategy : 5-fold stratified cross-validation (inner loop) +
                      held-out external test set (outer evaluation)

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

from scripts.validation_metrics import classification_metrics, metrics_to_dataframe
from scripts.hyperparameter_optimization import BayesianOptimiser
from scripts.model_serialization import save_model
from scripts.utils import configure_logger

logger = configure_logger(__name__)


# ── Default model catalogue ───────────────────────────────────────────────────

DEFAULT_CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "SVM": SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        is_unbalance=True,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    ),
}


# ── Training function ─────────────────────────────────────────────────────────

def train_classifiers(
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
    Train, optimise (optionally), and evaluate binary classification models.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Training and test feature matrices.
    y_train, y_test : np.ndarray
        Training and test binary labels (0/1).
    algorithms : list[str], optional
        Subset of algorithm keys to train. Defaults to all three.
    optimise : bool
        If True, apply Bayesian hyperparameter optimisation before training.
    n_trials : int
        Number of Optuna trials per algorithm.
    n_splits : int
        Number of cross-validation folds.
    model_dir : str
        Directory where serialised models are saved.
    results_dir : str
        Directory where result CSVs are saved.

    Returns
    -------
    pd.DataFrame
        Summary table of classification metrics for each model.
    """
    if algorithms is None:
        algorithms = list(DEFAULT_CLASSIFIERS.keys())

    all_metrics = []

    for algo_name in algorithms:
        logger.info(f"\n{'='*55}")
        logger.info(f"  Training classifier: {algo_name}")
        logger.info(f"{'='*55}")

        # ── Hyperparameter optimisation ───────────────────────────────────
        if optimise:
            logger.info(f"  Running Bayesian optimisation ({n_trials} trials) …")
            opt = BayesianOptimiser(
                algorithm=algo_name.lower().replace("lightgbm", "lgbm"),
                task="classification",
                n_trials=n_trials,
                cv_folds=n_splits,
            )
            opt.optimise(X_train, y_train)
            model = opt.build_best_model()
        else:
            model = DEFAULT_CLASSIFIERS.get(algo_name)
            if model is None:
                logger.warning(f"Unknown algorithm '{algo_name}'. Skipping.")
                continue

        # ── Cross-validation on training set ─────────────────────────────
        cv_splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=["roc_auc", "f1", "accuracy"],
            return_train_score=False,
            n_jobs=-1,
        )
        logger.info(
            f"  CV AUC  = {cv_scores['test_roc_auc'].mean():.4f} ± "
            f"{cv_scores['test_roc_auc'].std():.4f}"
        )
        logger.info(
            f"  CV F1   = {cv_scores['test_f1'].mean():.4f} ± "
            f"{cv_scores['test_f1'].std():.4f}"
        )

        # ── Final fit on full training set ────────────────────────────────
        model.fit(X_train, y_train)

        # ── External test-set evaluation ──────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        ext_metrics = classification_metrics(y_test, y_pred, y_prob)

        logger.info(
            f"  External Test — "
            f"BACC={ext_metrics['bacc']}, MCC={ext_metrics['mcc']}, "
            f"AUC={ext_metrics['auc']}"
        )

        # ── Serialise model ───────────────────────────────────────────────
        model_path = f"{model_dir}/classifier_{algo_name.lower()}.pkl"
        save_model(model, model_path)

        # ── Accumulate metrics ────────────────────────────────────────────
        row_metrics = {
            "cv_auc_mean": round(cv_scores["test_roc_auc"].mean(), 4),
            "cv_auc_std":  round(cv_scores["test_roc_auc"].std(), 4),
            "cv_f1_mean":  round(cv_scores["test_f1"].mean(), 4),
        }
        row_metrics.update(ext_metrics)
        df_row = metrics_to_dataframe(row_metrics, model_name=algo_name)
        all_metrics.append(df_row)

    # ── Aggregate and save results ────────────────────────────────────────────
    if all_metrics:
        results_df = pd.concat(all_metrics, ignore_index=True)
        import os
        os.makedirs(results_dir, exist_ok=True)
        out_path = f"{results_dir}/classification_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"Classification results saved to '{out_path}'.")
        return results_df

    return pd.DataFrame()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from scripts.preprocessing import prepare_matrices

    parser = argparse.ArgumentParser(
        description="BioScreen-QSAR — Classification Training"
    )
    parser.add_argument("--descriptors", required=True)
    parser.add_argument("--smiles",      default="curated_SMILES")
    parser.add_argument("--activity",    default="activity")
    parser.add_argument("--no_optimise", action="store_true")
    parser.add_argument("--n_trials",    default=50, type=int)
    args = parser.parse_args()

    X_tr, X_te, y_tr, y_te, _ = prepare_matrices(
        args.descriptors,
        smiles_col=args.smiles,
        activity_col=args.activity,
        stratify=True,
    )

    results = train_classifiers(
        X_tr, X_te, y_tr.astype(int), y_te.astype(int),
        optimise=not args.no_optimise,
        n_trials=args.n_trials,
    )
    print(results.to_string())
