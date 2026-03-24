"""
descriptors_ecfp.py — BioScreen-QSAR
Extended Connectivity Fingerprint (ECFP) calculation via RDKit.

ECFP (Rogers & Hahn, 2010) encodes the circular molecular environment
around each heavy atom up to a given radius r.  The resulting bit-vectors
are dense, rotation/translation-invariant, and well-suited for machine
learning classifiers and regressors.

Supported configurations:
  - Radius : 2 (ECFP4), 3 (ECFP6)
  - nBits  : 1024, 2048

Author: BioScreen-QSAR Project
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.utils import configure_logger, save_csv

logger = configure_logger(__name__)


# ── Core fingerprint generator ────────────────────────────────────────────────

class ECFPGenerator:
    """
    Compute Morgan / ECFP fingerprints for a list of SMILES strings.

    Parameters
    ----------
    radius : int
        Circular neighbourhood radius. radius=2 → ECFP4, radius=3 → ECFP6.
    n_bits : int
        Length of the binary bit-vector (1024 or 2048 recommended).
    use_chirality : bool
        Whether to include stereochemical information in the hashing.
    use_bond_types : bool
        Whether to encode bond-type information.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_chirality: bool = False,
        use_bond_types: bool = True,
    ) -> None:
        if radius not in (1, 2, 3, 4, 5, 6):
            raise ValueError("Radius must be an integer between 1 and 6.")
        if n_bits not in (512, 1024, 2048, 4096):
            raise ValueError("n_bits must be one of: 512, 1024, 2048, 4096.")

        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types

        logger.info(
            f"ECFPGenerator initialised: radius={radius} "
            f"(ECFP{2*radius}), n_bits={n_bits}."
        )

    def smiles_to_fp(self, smiles: str) -> np.ndarray | None:
        """
        Compute the ECFP fingerprint for a single SMILES string.

        Parameters
        ----------
        smiles : str
            Canonical SMILES string.

        Returns
        -------
        np.ndarray of shape (n_bits,) or None
            Binary integer array (0/1); None if the SMILES is invalid.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.n_bits,
                useChirality=self.use_chirality,
                useBondTypes=self.use_bond_types,
            )
            return np.array(fp, dtype=np.uint8)
        except Exception as exc:
            logger.debug(f"Fingerprint failed for '{smiles}': {exc}")
            return None

    def dataframe_to_fp_matrix(
        self,
        df: pd.DataFrame,
        smiles_col: str = "curated_SMILES",
        activity_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
        """
        Compute ECFP fingerprints for an entire dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing SMILES (and optionally activity) columns.
        smiles_col : str
            Column with canonical SMILES.
        activity_col : str, optional
            Column with biological activity values.

        Returns
        -------
        X : np.ndarray of shape (n_valid, n_bits)
            Fingerprint matrix.
        y : np.ndarray of shape (n_valid,) or None
            Activity values for valid compounds.
        valid_smiles : list[str]
            SMILES strings for which fingerprints were computed successfully.
        """
        fps = []
        labels = []
        valid_smiles = []

        n_failed = 0
        for _, row in df.iterrows():
            smi = row[smiles_col]
            fp = self.smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
                valid_smiles.append(smi)
                if activity_col:
                    labels.append(row[activity_col])
            else:
                n_failed += 1

        if n_failed:
            logger.warning(
                f"{n_failed} molecules failed fingerprint generation "
                f"and were excluded."
            )

        X = np.vstack(fps) if fps else np.empty((0, self.n_bits), dtype=np.uint8)
        y = np.array(labels) if labels else None

        logger.info(
            f"Fingerprint matrix: {X.shape[0]} compounds × {X.shape[1]} bits."
        )
        return X, y, valid_smiles

    def build_descriptor_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "curated_SMILES",
        activity_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a complete descriptor DataFrame including SMILES, activity, and
        all ECFP bit columns (Bit_0000 … Bit_XXXX).

        Parameters
        ----------
        df : pd.DataFrame
            Input curated dataset.
        smiles_col : str
            Column with canonical SMILES.
        activity_col : str, optional
            Column with biological activity.

        Returns
        -------
        pd.DataFrame
            Combined descriptor DataFrame.
        """
        X, y, valid_smiles = self.dataframe_to_fp_matrix(df, smiles_col, activity_col)

        bit_cols = [f"Bit_{i:04d}" for i in range(self.n_bits)]
        fp_df = pd.DataFrame(X, columns=bit_cols)
        fp_df.insert(0, "curated_SMILES", valid_smiles)

        if y is not None and activity_col:
            fp_df.insert(1, activity_col, y)

        return fp_df


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_descriptor_pipeline(
    input_path: str,
    output_path: str,
    smiles_col: str = "curated_SMILES",
    activity_col: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """
    Load a curated dataset, compute ECFP fingerprints, and save the result.

    Parameters
    ----------
    input_path : str
        Path to the curated CSV file.
    output_path : str
        Path for the output descriptor CSV.
    smiles_col : str
        SMILES column name.
    activity_col : str, optional
        Activity column name.
    radius : int
        ECFP radius (2 = ECFP4, 3 = ECFP6).
    n_bits : int
        Fingerprint bit-vector length.

    Returns
    -------
    pd.DataFrame
        Descriptor DataFrame.
    """
    from scripts.utils import load_csv

    df = load_csv(input_path, smiles_col=smiles_col, activity_col=activity_col)
    generator = ECFPGenerator(radius=radius, n_bits=n_bits)
    descriptor_df = generator.build_descriptor_dataframe(df, smiles_col, activity_col)
    save_csv(descriptor_df, output_path)
    return descriptor_df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioScreen-QSAR — ECFP Descriptor Generation"
    )
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--smiles",   default="curated_SMILES")
    parser.add_argument("--activity", default=None)
    parser.add_argument("--radius",   default=2, type=int)
    parser.add_argument("--n_bits",   default=2048, type=int)
    args = parser.parse_args()

    run_descriptor_pipeline(
        input_path=args.input,
        output_path=args.output,
        smiles_col=args.smiles,
        activity_col=args.activity,
        radius=args.radius,
        n_bits=args.n_bits,
    )
