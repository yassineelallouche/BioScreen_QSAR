"""
data_curation.py — BioScreen-QSAR
Molecular standardisation pipeline using RDKit.

Steps applied to each compound:
  1. Parse SMILES → RDKit Mol object
  2. Remove molecules that cannot be parsed (invalid SMILES)
  3. Sanitise the molecule (valence correction, aromaticity)
  4. Remove mixture components (retain the largest fragment)
  5. Neutralise formal charges (simple rule-based neutralisation)
  6. Standardise tautomers (canonical tautomer selection via RDKit)
  7. Re-generate canonical SMILES for downstream use

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize

from scripts.utils import configure_logger, load_csv, save_csv

logger = configure_logger(__name__)


# ── Core standardiser class ───────────────────────────────────────────────────

class MolecularStandardiser:
    """
    Applies a sequential standardisation pipeline to a collection of SMILES
    strings, mirroring best-practice guidelines for QSAR data curation
    (Fourches et al., 2010, 2016).

    Parameters
    ----------
    remove_mixtures : bool
        If True, keep only the largest fragment of multi-component SMILES.
    neutralise_charges : bool
        If True, attempt to neutralise formal charges via RDKit's Uncharger.
    standardise_tautomers : bool
        If True, apply canonical tautomer enumeration and selection.
    """

    def __init__(
        self,
        remove_mixtures: bool = True,
        neutralise_charges: bool = True,
        standardise_tautomers: bool = True,
    ) -> None:
        self.remove_mixtures = remove_mixtures
        self.neutralise_charges = neutralise_charges
        self.standardise_tautomers = standardise_tautomers

        # RDKit standardisation utilities
        self._salt_remover = SaltRemover.SaltRemover()
        self._uncharger = rdMolStandardize.Uncharger()
        self._tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
        self._largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()

    # ── Public interface ──────────────────────────────────────────────────────

    def standardise_smiles(self, smiles: str) -> str | None:
        """
        Standardise a single SMILES string.

        Parameters
        ----------
        smiles : str
            Raw SMILES string.

        Returns
        -------
        str or None
            Canonical, standardised SMILES; None if the molecule is invalid
            or cannot be processed.
        """
        mol = self._parse(smiles)
        if mol is None:
            return None

        if self.remove_mixtures:
            mol = self._remove_largest_fragment(mol)
            if mol is None:
                return None

        if self.neutralise_charges:
            mol = self._neutralise(mol)
            if mol is None:
                return None

        if self.standardise_tautomers:
            mol = self._canonical_tautomer(mol)
            if mol is None:
                return None

        try:
            Chem.SanitizeMol(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as exc:
            logger.debug(f"Final sanitisation failed for '{smiles}': {exc}")
            return None

    def standardise_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        activity_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Apply standardisation to every row of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset containing at minimum a SMILES column.
        smiles_col : str
            Column name with raw SMILES strings.
        activity_col : str, optional
            Column name of the biological endpoint.

        Returns
        -------
        pd.DataFrame
            DataFrame with an added 'curated_SMILES' column;
            rows with failed standardisation are dropped.
        """
        logger.info(f"Starting standardisation of {len(df)} compounds …")

        df = df.copy()
        df["curated_SMILES"] = df[smiles_col].apply(self.standardise_smiles)

        # Report and remove failures
        n_failed = df["curated_SMILES"].isna().sum()
        if n_failed:
            logger.warning(
                f"{n_failed} compounds could not be standardised and will be excluded."
            )

        df = df.dropna(subset=["curated_SMILES"]).reset_index(drop=True)
        logger.info(f"Standardisation complete. {len(df)} compounds retained.")

        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse(smiles: str) -> Chem.Mol | None:
        """Parse a SMILES string into an RDKit Mol object."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def _remove_largest_fragment(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Keep only the largest organic fragment (removes salts/counter-ions)."""
        try:
            # First pass: salt remover with built-in list
            mol = self._salt_remover.StripMol(mol, dontRemoveEverything=True)
            # Second pass: largest fragment chooser handles edge cases
            mol = self._largest_fragment_chooser.choose(mol)
            return mol
        except Exception:
            return None

    def _neutralise(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Neutralise formal charges using RDKit's Uncharger."""
        try:
            return self._uncharger.uncharge(mol)
        except Exception:
            return mol  # Return original if neutralisation fails

    def _canonical_tautomer(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Select the canonical tautomeric form."""
        try:
            return self._tautomer_enumerator.Canonicalize(mol)
        except Exception:
            return mol  # Return original if tautomerisation fails


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_curation_pipeline(
    input_path: str,
    output_path: str,
    smiles_col: str = "SMILES",
    activity_col: str | None = None,
    remove_mixtures: bool = True,
    neutralise_charges: bool = True,
    standardise_tautomers: bool = True,
) -> pd.DataFrame:
    """
    End-to-end curation pipeline: load → standardise → save.

    Parameters
    ----------
    input_path : str
        Path to raw CSV dataset.
    output_path : str
        Path where the standardised (pre-deduplication) dataset is saved.
    smiles_col : str
        Column name of the SMILES strings.
    activity_col : str, optional
        Column name of the biological endpoint.
    remove_mixtures : bool
        Enable mixture/salt removal step.
    neutralise_charges : bool
        Enable charge neutralisation step.
    standardise_tautomers : bool
        Enable canonical tautomer selection.

    Returns
    -------
    pd.DataFrame
        Standardised dataset ready for duplicate analysis.
    """
    df = load_csv(input_path, smiles_col=smiles_col, activity_col=activity_col)
    n_input = len(df)

    standardiser = MolecularStandardiser(
        remove_mixtures=remove_mixtures,
        neutralise_charges=neutralise_charges,
        standardise_tautomers=standardise_tautomers,
    )
    df_std = standardiser.standardise_dataframe(df, smiles_col=smiles_col)

    save_csv(df_std, output_path)
    logger.info(
        f"Curation: {n_input} → {len(df_std)} compounds "
        f"({n_input - len(df_std)} removed)."
    )
    return df_std


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioScreen-QSAR — Molecular Standardisation Module"
    )
    parser.add_argument("--input",    required=True, help="Raw CSV input file")
    parser.add_argument("--output",   required=True, help="Standardised CSV output file")
    parser.add_argument("--smiles",   default="SMILES", help="SMILES column name")
    parser.add_argument("--activity", default=None, help="Activity column name")
    args = parser.parse_args()

    run_curation_pipeline(
        input_path=args.input,
        output_path=args.output,
        smiles_col=args.smiles,
        activity_col=args.activity,
    )
