"""
validation_metrics.py — BioScreen-QSAR
Comprehensive validation metric computation for QSAR models.

Classification metrics:
  Accuracy (ACC), Sensitivity (Se), Specificity (Sp),
  Positive Predictive Value (PPV), Negative Predictive Value (NPV),
  Matthews Correlation Coefficient (MCC), Area Under ROC Curve (AUC),
  F1-score, Balanced Accuracy (BACC).

Regression metrics:
  R², Mean Absolute Error (MAE), Mean Squared Error (MSE),
  Root Mean Squared Error (RMSE), Explained Variance Score.

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    roc_curve,
)

from scripts.utils import configure_logger

logger = configure_logger(__name__)


# ── Classification metrics ────────────────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """
    Compute a comprehensive set of classification performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = inactive, 1 = active).
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : np.ndarray, optional
        Predicted probabilities for the positive class (class 1).
        Required for AUC computation.

    Returns
    -------
    dict
        Keys: acc, bacc, sensitivity, specificity, ppv, npv, mcc, auc, f1.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / max(tp + fn, 1)   # Recall for positive class
    specificity = tn / max(tn + fp, 1)   # Recall for negative class
    ppv = tp / max(tp + fp, 1)           # Precision
    npv = tn / max(tn + fn, 1)

    auc = (
        roc_auc_score(y_true, y_prob)
        if y_prob is not None
        else float("nan")
    )

    metrics = {
        "acc":         round(accuracy_score(y_true, y_pred), 4),
        "bacc":        round(balanced_accuracy_score(y_true, y_pred), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv":         round(ppv, 4),
        "npv":         round(npv, 4),
        "mcc":         round(matthews_corrcoef(y_true, y_pred), 4),
        "auc":         round(auc, 4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    logger.info(
        f"Classification metrics — "
        f"ACC={metrics['acc']}, BACC={metrics['bacc']}, "
        f"Se={metrics['sensitivity']}, Sp={metrics['specificity']}, "
        f"MCC={metrics['mcc']}, AUC={metrics['auc']}"
    )
    return metrics


# ── Regression metrics ────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute a comprehensive set of regression performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Experimental activity values.
    y_pred : np.ndarray
        Model-predicted activity values.

    Returns
    -------
    dict
        Keys: r2, mae, mse, rmse, explained_variance.
    """
    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        "r2":               round(r2_score(y_true, y_pred), 4),
        "mae":              round(mean_absolute_error(y_true, y_pred), 4),
        "mse":              round(mse, 4),
        "rmse":             round(float(np.sqrt(mse)), 4),
        "explained_var":    round(explained_variance_score(y_true, y_pred), 4),
    }

    logger.info(
        f"Regression metrics — "
        f"R²={metrics['r2']}, MAE={metrics['mae']}, "
        f"RMSE={metrics['rmse']}, Explained Var={metrics['explained_var']}"
    )
    return metrics


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> None:
    """Plot the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1B6CA8", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"ROC curve saved to '{save_path}'.")
    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> None:
    """Plot a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Inactive", "Active"],
        yticklabels=["Inactive", "Active"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def plot_predicted_vs_experimental(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    activity_label: str = "pMIC (log units)",
    save_path: str | None = None,
) -> None:
    """Scatter plot of predicted vs. experimental values for regression."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="none", color="#E05A2B", s=25)

    lims = [min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Experimental {activity_label}")
    ax.set_ylabel(f"Predicted {activity_label}")
    ax.set_title(f"Predicted vs. Experimental — {model_name}")

    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
            fontsize=10, color="#333333")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()


def metrics_to_dataframe(metrics_dict: dict, model_name: str = "Model") -> pd.DataFrame:
    """Convert a metrics dictionary to a tidy single-row DataFrame."""
    row = {"Model": model_name}
    row.update(metrics_dict)
    return pd.DataFrame([row])
