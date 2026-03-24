"""
tests/test_curation.py — BioScreen-QSAR
Unit tests for the data curation module.

Run with:  pytest tests/ -v
"""

import pytest
import pandas as pd
from scripts.data_curation import MolecularStandardiser, run_curation_pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_SMILES   = "CC(=O)Nc1ccc(O)cc1"            # Paracetamol
SALT_SMILES    = "CC(=O)Nc1ccc(O)cc1.Na"          # Paracetamol sodium salt
INVALID_SMILES = "this_is_not_valid"
CHARGED_SMILES = "CC(=O)Nc1ccc([O-])cc1"          # Phenolate anion
MIXTURE_SMILES = "CC(=O)O.CCO"                     # Acetic acid + ethanol


@pytest.fixture
def standardiser():
    return MolecularStandardiser(
        remove_mixtures=True,
        neutralise_charges=True,
        standardise_tautomers=True,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSingleSMILES:
    """Tests for standardise_smiles on individual molecules."""

    def test_valid_smiles_returns_string(self, standardiser):
        result = standardiser.standardise_smiles(VALID_SMILES)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_smiles_returns_none(self, standardiser):
        result = standardiser.standardise_smiles(INVALID_SMILES)
        assert result is None

    def test_empty_string_returns_none(self, standardiser):
        result = standardiser.standardise_smiles("")
        assert result is None

    def test_salt_is_desalted(self, standardiser):
        """Salt form should produce same canonical SMILES as parent."""
        canonical_parent = standardiser.standardise_smiles(VALID_SMILES)
        canonical_salt   = standardiser.standardise_smiles(SALT_SMILES)
        assert canonical_parent is not None
        assert canonical_salt is not None
        assert canonical_parent == canonical_salt

    def test_mixture_retains_largest_fragment(self, standardiser):
        """From a mixture, the largest organic fragment is retained."""
        result = standardiser.standardise_smiles(MIXTURE_SMILES)
        assert result is not None
        # Should not contain '.' (mixture separator)
        assert "." not in result

    def test_charged_species_neutralised(self, standardiser):
        """Charged SMILES should produce a neutral canonical form."""
        result = standardiser.standardise_smiles(CHARGED_SMILES)
        assert result is not None
        # Phenolate [O-] → phenol OH; canonical SMILES should match parent
        parent = standardiser.standardise_smiles(VALID_SMILES)
        assert result == parent


class TestDataFrame:
    """Tests for standardise_dataframe on a DataFrame input."""

    def test_returns_dataframe(self, standardiser):
        df = pd.DataFrame({
            "SMILES":   [VALID_SMILES, SALT_SMILES, INVALID_SMILES],
            "activity": [1, 1, 0],
        })
        result = standardiser.standardise_dataframe(df, smiles_col="SMILES")
        assert isinstance(result, pd.DataFrame)

    def test_invalid_entries_removed(self, standardiser):
        df = pd.DataFrame({
            "SMILES":   [VALID_SMILES, INVALID_SMILES],
            "activity": [1, 0],
        })
        result = standardiser.standardise_dataframe(df, smiles_col="SMILES")
        assert len(result) == 1

    def test_curated_smiles_column_added(self, standardiser):
        df = pd.DataFrame({"SMILES": [VALID_SMILES], "activity": [1]})
        result = standardiser.standardise_dataframe(df, smiles_col="SMILES")
        assert "curated_SMILES" in result.columns

    def test_original_columns_preserved(self, standardiser):
        df = pd.DataFrame({
            "SMILES":      [VALID_SMILES],
            "activity":    [1],
            "compound_id": ["CMPD_001"],
        })
        result = standardiser.standardise_dataframe(df, smiles_col="SMILES")
        assert "compound_id" in result.columns

    def test_all_valid_preserved(self, standardiser):
        smiles_list = [VALID_SMILES, SALT_SMILES, MIXTURE_SMILES]
        df = pd.DataFrame({"SMILES": smiles_list, "activity": [1, 1, 0]})
        result = standardiser.standardise_dataframe(df, smiles_col="SMILES")
        assert len(result) == len(smiles_list)
