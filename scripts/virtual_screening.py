"""
virtual_screening.py — BioScreen-QSAR
Ligand-based virtual screening (LBVS) pipeline.

Workflow:
  1. Load a candidate compound library (CSV with SMILES).
  2. Standardise and validate each compound.
  3. Compute ECFP fingerprints.
  4. Apply the same variance filter used during training.
  5. Run predictions with the pre-trained model.
  6. Rank compounds by predicted probability (classification)
     or predicted value (regression).
  7. Export results with metadata.

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from scripts.data_curation import MolecularStandardiser
from scripts.descriptors_ecfp import ECFPGenerator
from scripts.model_serialization import load_model
from scripts.preprocessing import FeaturePreprocessor
from scripts.utils import configure_logger, save_csv

logger = configure_logger(__name__)

# Maximum library size for the cloud-hosted version
CLOUD_MAX_COMPOUNDS = 2000


def run_virtual_screening(
    library_path: str,
    model_path: str,
    preprocessor: FeaturePreprocessor | None = None,
    smiles_col: str = "SMILES",
    task: str = "classification",
    ecfp_radius: int = 2,
    ecfp_n_bits: int = 2048,
    output_path: str | None = None,
    cloud_mode: bool = False,
) -> pd.DataFrame:
    """
    Execute the full ligand-based virtual screening pipeline.

    Parameters
    ----------
    library_path : str
        Path to the compound library CSV file.
    model_path : str
        Path to the pre-trained model .pkl file.
    preprocessor : FeaturePreprocessor, optional
        Fitted preprocessor containing the variance filter from training.
        If None, no filtering is applied.
    smiles_col : str
        Name of the SMILES column in the library CSV.
    task : {'classification', 'regression'}
        Type of QSAR model.
    ecfp_radius : int
        ECFP fingerprint radius (must match training configuration).
    ecfp_n_bits : int
        Fingerprint bit-vector length (must match training configuration).
    output_path : str, optional
        If provided, save the results table to this CSV path.
    cloud_mode : bool
        If True, limit processing to CLOUD_MAX_COMPOUNDS compounds.

    Returns
    -------
    pd.DataFrame
        Results table with columns: curated_SMILES, rank, predicted_class /
        predicted_value, and (for classification) probability_active.
    """
    # ── Load library ──────────────────────────────────────────────────────────
    library_df = pd.read_csv(library_path)
    original_cols = [c for c in library_df.columns if c != smiles_col]
    logger.info(
        f"Virtual screening library loaded: {len(library_df)} compounds."
    )

    if cloud_mode and len(library_df) > CLOUD_MAX_COMPOUNDS:
        logger.warning(
            f"Cloud mode: truncating library to {CLOUD_MAX_COMPOUNDS} compounds. "
            f"For larger libraries, run BioScreen-QSAR locally."
        )
        library_df = library_df.iloc[:CLOUD_MAX_COMPOUNDS].copy()

    # ── Standardise ───────────────────────────────────────────────────────────
    standardiser = MolecularStandardiser()
    library_df["curated_SMILES"] = library_df[smiles_col].apply(
        standardiser.standardise_smiles
    )

    n_failed = library_df["curated_SMILES"].isna().sum()
    if n_failed:
        logger.warning(f"{n_failed} compounds failed standardisation and were excluded.")

    library_df = library_df.dropna(subset=["curated_SMILES"]).reset_index(drop=True)
    logger.info(f"{len(library_df)} compounds retained after standardisation.")

    # ── Compute fingerprints ──────────────────────────────────────────────────
    generator = ECFPGenerator(radius=ecfp_radius, n_bits=ecfp_n_bits)
    fps = []
    valid_idx = []

    for idx, row in library_df.iterrows():
        fp = generator.smiles_to_fp(row["curated_SMILES"])
        if fp is not None:
            fps.append(fp)
            valid_idx.append(idx)

    if not fps:
        logger.error("No valid fingerprints computed. Aborting virtual screening.")
        return pd.DataFrame()

    X = np.vstack(fps).astype(np.float32)
    library_df = library_df.loc[valid_idx].reset_index(drop=True)
    logger.info(f"Fingerprint matrix: {X.shape}")

    # ── Apply variance filter (if preprocessor provided) ─────────────────────
    if preprocessor is not None:
        try:
            X = preprocessor.transform(X)
            logger.info(f"Feature matrix after variance filter: {X.shape}")
        except Exception as exc:
            logger.warning(f"Variance filter failed: {exc}. Using unfiltered features.")

    # ── Load model and predict ────────────────────────────────────────────────
    model = load_model(model_path)

    if task == "classification":
        y_pred = model.predict(X)
        results_df = library_df[["curated_SMILES"] + [c for c in original_cols
                                                       if c in library_df.columns]].copy()
        results_df["predicted_class"] = y_pred.astype(int)
        results_df["predicted_label"] = results_df["predicted_class"].map(
            {0: "Inactive", 1: "Active"}
        )

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
            results_df["probability_active"] = np.round(prob, 4)
            # Rank: active compounds first, sorted by descending probability
            results_df = results_df.sort_values(
                by=["predicted_class", "probability_active"],
                ascending=[False, False],
            ).reset_index(drop=True)
        else:
            results_df = results_df.sort_values(
                by="predicted_class", ascending=False
            ).reset_index(drop=True)

    else:  # regression
        y_pred = model.predict(X)
        results_df = library_df[["curated_SMILES"] + [c for c in original_cols
                                                       if c in library_df.columns]].copy()
        results_df["predicted_pMIC"] = np.round(y_pred, 4)
        # Rank: highest predicted potency first
        results_df = results_df.sort_values(
            by="predicted_pMIC", ascending=False
        ).reset_index(drop=True)

    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    n_active = (results_df["predicted_class"] == 1).sum() if task == "classification" else None
    logger.info(
        f"Virtual screening complete. "
        f"Total screened: {len(results_df)}"
        + (f" | Predicted active: {n_active}" if n_active is not None else "")
    )

    if output_path:
        save_csv(results_df, output_path)

    return results_df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioScreen-QSAR — Virtual Screening"
    )
    parser.add_argument("--library",   required=True, help="Compound library CSV")
    parser.add_argument("--model",     required=True, help="Trained model .pkl")
    parser.add_argument("--output",    required=True, help="Output CSV path")
    parser.add_argument("--smiles",    default="SMILES")
    parser.add_argument("--task",      default="classification",
                        choices=["classification", "regression"])
    parser.add_argument("--radius",    default=2,    type=int)
    parser.add_argument("--n_bits",    default=2048, type=int)
    args = parser.parse_args()

    run_virtual_screening(
        library_path=args.library,
        model_path=args.model,
        smiles_col=args.smiles,
        task=args.task,
        ecfp_radius=args.radius,
        ecfp_n_bits=args.n_bits,
        output_path=args.output,
    )
