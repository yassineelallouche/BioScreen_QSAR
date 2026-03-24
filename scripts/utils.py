"""
utils.py — BioScreen-QSAR
Shared utility functions used across all modules.
Author: BioScreen-QSAR Project
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────

def configure_logger(name: str = "bioscreen_qsar", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a module-level logger with timestamped console output.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__ of the calling module).
    level : int
        Logging verbosity level (e.g., logging.DEBUG, logging.INFO).
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = configure_logger(__name__)


# ── File I/O helpers ──────────────────────────────────────────────────────────

def load_csv(filepath: str, smiles_col: str = "SMILES", activity_col: str = None) -> pd.DataFrame:
    """
    Load a CSV dataset and perform basic integrity checks.

    Parameters
    ----------
    filepath : str
        Path to the input CSV file.
    smiles_col : str
        Name of the column containing SMILES strings.
    activity_col : str, optional
        Name of the biological activity column.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records from '{filepath}'.")

    if smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if activity_col and activity_col not in df.columns:
        raise ValueError(
            f"Activity column '{activity_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows where SMILES is null
    n_before = len(df)
    df = df.dropna(subset=[smiles_col])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} rows with missing SMILES.")

    return df


def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save a DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : str
        Destination file path.
    index : bool
        Whether to write the row index.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved {len(df)} records to '{filepath}'.")


# ── SMILES sanitisation helpers ───────────────────────────────────────────────

def is_valid_smiles(smiles: str) -> bool:
    """
    Check whether a SMILES string can be parsed by RDKit into a valid molecule.

    Parameters
    ----------
    smiles : str
        SMILES string to validate.

    Returns
    -------
    bool
        True if the molecule is parseable, False otherwise.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonical_smiles(smiles: str) -> str | None:
    """
    Convert a SMILES string to its RDKit canonical form.

    Parameters
    ----------
    smiles : str
        Input SMILES.

    Returns
    -------
    str or None
        Canonical SMILES string, or None if the input is invalid.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ── Reporting ─────────────────────────────────────────────────────────────────

def generate_summary_report(
    n_input: int,
    n_after_standardisation: int,
    n_duplicates_removed: int,
    n_discordant: int,
    n_final: int,
    output_path: str = "reports/curation_report.txt",
) -> None:
    """
    Write a plain-text curation summary report to disk.

    Parameters
    ----------
    n_input : int
        Total compounds in the raw input dataset.
    n_after_standardisation : int
        Compounds remaining after structural standardisation.
    n_duplicates_removed : int
        Duplicate records removed (concordant averages kept, discordant dropped).
    n_discordant : int
        Discordant duplicate pairs discarded entirely.
    n_final : int
        Final compound count used for modelling.
    output_path : str
        Destination path for the text report.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        "  BioScreen-QSAR — Curation Summary Report",
        f"  Generated: {timestamp}",
        "=" * 60,
        f"  Input compounds              : {n_input:>8}",
        f"  After standardisation        : {n_after_standardisation:>8}",
        f"  Duplicate entries removed    : {n_duplicates_removed:>8}",
        f"  Discordant pairs discarded   : {n_discordant:>8}",
        f"  Final curated dataset        : {n_final:>8}",
        "=" * 60,
    ]
    report_text = "\n".join(lines)
    with open(output_path, "w") as fh:
        fh.write(report_text)
    logger.info(f"Curation report saved to '{output_path}'.")
    print(report_text)


# ── Miscellaneous ─────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create a directory (and all parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def timestamp_str() -> str:
    """Return a compact timestamp string suitable for use in filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
