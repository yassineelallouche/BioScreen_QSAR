"""
model_serialization.py — BioScreen-QSAR
Utilities for saving, loading, and versioning trained ML models using joblib.

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import os
import json
from datetime import datetime

import joblib

from scripts.utils import configure_logger

logger = configure_logger(__name__)


def save_model(model, filepath: str, metadata: dict | None = None) -> None:
    """
    Serialise a trained estimator to disk using joblib compression.

    Parameters
    ----------
    model : scikit-learn / LightGBM estimator
        Fitted model object.
    filepath : str
        Destination path for the .pkl file.
    metadata : dict, optional
        Additional information (e.g., algorithm name, task, metrics)
        stored alongside the model in a companion JSON file.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    joblib.dump(model, filepath, compress=3)
    logger.info(f"Model saved to '{filepath}'.")

    if metadata is not None:
        meta_path = filepath.replace(".pkl", "_metadata.json")
        metadata["saved_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info(f"Model metadata saved to '{meta_path}'.")


def load_model(filepath: str):
    """
    Load a serialised model from a .pkl file.

    Parameters
    ----------
    filepath : str
        Path to the saved model file.

    Returns
    -------
    Fitted estimator.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Model file not found: '{filepath}'.")

    model = joblib.load(filepath)
    logger.info(f"Model loaded from '{filepath}'.")
    return model


def list_saved_models(model_dir: str = "models") -> list[str]:
    """
    List all .pkl model files in a given directory.

    Parameters
    ----------
    model_dir : str
        Directory to search.

    Returns
    -------
    list[str]
        Paths of discovered .pkl files.
    """
    if not os.path.isdir(model_dir):
        return []
    return [
        os.path.join(model_dir, f)
        for f in sorted(os.listdir(model_dir))
        if f.endswith(".pkl")
    ]
