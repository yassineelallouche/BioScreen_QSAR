"""
tests/test_descriptors.py — BioScreen-QSAR
Unit tests for ECFP fingerprint generation.
"""

import pytest
import numpy as np
import pandas as pd
from scripts.descriptors_ecfp import ECFPGenerator


VALID_SMILES   = "CC(=O)Nc1ccc(O)cc1"
INVALID_SMILES = "not_valid_smiles"


@pytest.fixture
def gen2048():
    return ECFPGenerator(radius=2, n_bits=2048)

@pytest.fixture
def gen1024():
    return ECFPGenerator(radius=2, n_bits=1024)


class TestSingleFingerprint:

    def test_valid_smiles_returns_array(self, gen2048):
        fp = gen2048.smiles_to_fp(VALID_SMILES)
        assert fp is not None
        assert isinstance(fp, np.ndarray)

    def test_correct_length_2048(self, gen2048):
        fp = gen2048.smiles_to_fp(VALID_SMILES)
        assert fp.shape == (2048,)

    def test_correct_length_1024(self, gen1024):
        fp = gen1024.smiles_to_fp(VALID_SMILES)
        assert fp.shape == (1024,)

    def test_binary_values_only(self, gen2048):
        fp = gen2048.smiles_to_fp(VALID_SMILES)
        assert set(fp).issubset({0, 1})

    def test_invalid_smiles_returns_none(self, gen2048):
        fp = gen2048.smiles_to_fp(INVALID_SMILES)
        assert fp is None

    def test_distinct_molecules_different_fps(self, gen2048):
        fp1 = gen2048.smiles_to_fp("CC(=O)O")      # Acetic acid
        fp2 = gen2048.smiles_to_fp("c1ccccc1")     # Benzene
        assert not np.array_equal(fp1, fp2)

    def test_same_molecule_identical_fps(self, gen2048):
        fp1 = gen2048.smiles_to_fp(VALID_SMILES)
        fp2 = gen2048.smiles_to_fp(VALID_SMILES)
        assert np.array_equal(fp1, fp2)


class TestDataFrameMatrix:

    def test_matrix_shape(self, gen2048):
        df = pd.DataFrame({
            "curated_SMILES": ["CC(=O)O", "c1ccccc1", "CCO"],
            "activity":       [1, 0, 1],
        })
        X, y, smi = gen2048.dataframe_to_fp_matrix(df, "curated_SMILES", "activity")
        assert X.shape == (3, 2048)
        assert len(y) == 3
        assert len(smi) == 3

    def test_invalid_excluded_from_matrix(self, gen2048):
        df = pd.DataFrame({
            "curated_SMILES": ["CC(=O)O", INVALID_SMILES, "CCO"],
            "activity":       [1, 0, 1],
        })
        X, y, smi = gen2048.dataframe_to_fp_matrix(df, "curated_SMILES", "activity")
        assert X.shape[0] == 2   # One invalid removed

    def test_descriptor_dataframe_columns(self, gen2048):
        df = pd.DataFrame({
            "curated_SMILES": ["CC(=O)O"],
            "activity":       [1],
        })
        desc = gen2048.build_descriptor_dataframe(df, "curated_SMILES", "activity")
        assert "curated_SMILES" in desc.columns
        assert "activity" in desc.columns
        assert "Bit_0000" in desc.columns
        assert "Bit_2047" in desc.columns
        assert desc.shape[1] == 2 + 2048   # SMILES + activity + 2048 bits


class TestConfiguration:

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError):
            ECFPGenerator(radius=10)

    def test_invalid_nbits_raises(self):
        with pytest.raises(ValueError):
            ECFPGenerator(n_bits=100)
