"""
duplicate_handling.py — BioScreen-QSAR
Duplicate detection and resolution for QSAR datasets.

Rules (adapted from Fourches et al., 2010; 2016):

  Binary / Classification data:
    - Concordant duplicates  → keep one representative record.
    - Discordant duplicates  → discard all records for that structure.

  Continuous / Regression data:
    - Duplicates differing by ≤ 0.2 log units → average the values, keep one.
    - Duplicates differing by  > 0.2 log units → discard all records.

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal

from scripts.utils import configure_logger, save_csv

logger = configure_logger(__name__)

# Threshold (log units) below which continuous duplicates are averaged
CONTINUOUS_THRESHOLD = 0.2


def handle_duplicates(
    df: pd.DataFrame,
    smiles_col: str = "curated_SMILES",
    activity_col: str = "activity",
    data_type: Literal["binary", "continuous"] = "binary",
    output_curated: str | None = None,
    output_report: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Detect and resolve duplicate SMILES entries according to the BioScreen-QSAR
    duplicate-handling protocol.

    Parameters
    ----------
    df : pd.DataFrame
        Standardised dataset (output of data_curation.py).
    smiles_col : str
        Column containing canonical SMILES.
    activity_col : str
        Column containing the biological endpoint.
    data_type : {'binary', 'continuous'}
        Whether the endpoint is categorical (binary) or quantitative (continuous).
    output_curated : str, optional
        If provided, save the curated dataset to this path.
    output_report : str, optional
        If provided, save the duplicate analysis report to this path.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - Curated DataFrame with duplicates resolved.
        - Dictionary summarising duplicate statistics.
    """
    n_input = len(df)
    logger.info(
        f"Duplicate analysis started. Input: {n_input} records, "
        f"data type: '{data_type}'."
    )

    df = df.copy()

    # Identify duplicated canonical SMILES
    duplicated_mask = df.duplicated(subset=[smiles_col], keep=False)
    n_unique_structures_with_duplicates = (
        df[duplicated_mask][smiles_col].nunique()
    )

    if data_type == "binary":
        df_clean, n_removed, n_discordant = _resolve_binary_duplicates(
            df, smiles_col, activity_col
        )
    else:  # continuous
        df_clean, n_removed, n_discordant = _resolve_continuous_duplicates(
            df, smiles_col, activity_col
        )

    n_final = len(df_clean)
    stats = {
        "n_input": n_input,
        "n_duplicated_structures": n_unique_structures_with_duplicates,
        "n_removed_duplicate_records": n_removed,
        "n_discordant_structures": n_discordant,
        "n_final": n_final,
        "pct_removed": round(100 * (n_input - n_final) / max(n_input, 1), 2),
    }

    _log_stats(stats)

    if output_curated:
        save_csv(df_clean, output_curated)
    if output_report:
        _write_report(stats, output_report)

    return df_clean.reset_index(drop=True), stats


# ── Binary duplicate resolution ───────────────────────────────────────────────

def _resolve_binary_duplicates(
    df: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
) -> tuple[pd.DataFrame, int, int]:
    """
    Resolve binary duplicates.

    Concordant pairs (all copies share the same label) → keep one.
    Discordant pairs (contradictory labels)            → discard all.
    """
    n_input = len(df)
    keep_rows = []
    n_discordant = 0

    for smi, group in df.groupby(smiles_col):
        unique_labels = group[activity_col].unique()

        if len(unique_labels) == 1:
            # Concordant: keep the first occurrence
            keep_rows.append(group.iloc[0])
        else:
            # Discordant: discard all copies
            n_discordant += 1
            logger.debug(f"Discordant binary duplicate discarded: {smi}")

    df_clean = pd.DataFrame(keep_rows)
    n_removed = n_input - len(df_clean)
    return df_clean, n_removed, n_discordant


# ── Continuous duplicate resolution ──────────────────────────────────────────

def _resolve_continuous_duplicates(
    df: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
) -> tuple[pd.DataFrame, int, int]:
    """
    Resolve continuous duplicates.

    Activity spread ≤ CONTINUOUS_THRESHOLD log units → average values, keep one.
    Activity spread  > CONTINUOUS_THRESHOLD log units → discard all copies.
    """
    n_input = len(df)
    keep_rows = []
    n_discordant = 0

    for smi, group in df.groupby(smiles_col):
        values = group[activity_col].dropna().values.astype(float)

        if len(values) == 0:
            continue
        if len(values) == 1:
            keep_rows.append(group.iloc[0])
            continue

        spread = values.max() - values.min()
        if spread <= CONTINUOUS_THRESHOLD:
            # Average the values and keep metadata from the first row
            representative = group.iloc[0].copy()
            representative[activity_col] = float(np.mean(values))
            keep_rows.append(representative)
        else:
            # Discordant: spread too large → discard
            n_discordant += 1
            logger.debug(
                f"Discordant continuous duplicate (spread={spread:.3f}) discarded: {smi}"
            )

    df_clean = pd.DataFrame(keep_rows)
    n_removed = n_input - len(df_clean)
    return df_clean, n_removed, n_discordant


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _log_stats(stats: dict) -> None:
    logger.info(
        f"Duplicate analysis complete:\n"
        f"  Input records                    : {stats['n_input']}\n"
        f"  Unique structures with duplicates: {stats['n_duplicated_structures']}\n"
        f"  Records removed                  : {stats['n_removed_duplicate_records']}\n"
        f"  Discordant structures discarded  : {stats['n_discordant_structures']}\n"
        f"  Final dataset size               : {stats['n_final']}\n"
        f"  Percent removed                  : {stats['pct_removed']}%"
    )


def _write_report(stats: dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "BioScreen-QSAR — Duplicate Analysis Report",
        "=" * 50,
        f"Input records                    : {stats['n_input']}",
        f"Unique structures with duplicates: {stats['n_duplicated_structures']}",
        f"Records removed                  : {stats['n_removed_duplicate_records']}",
        f"Discordant structures discarded  : {stats['n_discordant_structures']}",
        f"Final dataset size               : {stats['n_final']}",
        f"Percent removed                  : {stats['pct_removed']}%",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    logger.info(f"Duplicate report saved to '{path}'.")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioScreen-QSAR — Duplicate Handling Module"
    )
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--smiles",   default="curated_SMILES")
    parser.add_argument("--activity", default="activity")
    parser.add_argument("--type",     default="binary", choices=["binary", "continuous"])
    args = parser.parse_args()

    from scripts.utils import load_csv
    df = load_csv(args.input, smiles_col=args.smiles, activity_col=args.activity)
    handle_duplicates(
        df,
        smiles_col=args.smiles,
        activity_col=args.activity,
        data_type=args.type,
        output_curated=args.output,
        output_report=args.output.replace(".csv", "_duplicate_report.txt"),
    )
