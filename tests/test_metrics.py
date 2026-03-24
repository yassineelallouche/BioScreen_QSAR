"""
tests/test_metrics.py — BioScreen-QSAR
Unit tests for validation metrics computation.
"""

import pytest
import numpy as np
from scripts.validation_metrics import classification_metrics, regression_metrics


class TestClassificationMetrics:

    def test_perfect_prediction(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        m = classification_metrics(y, y)
        assert m["acc"]         == 1.0
        assert m["bacc"]        == 1.0
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0
        assert m["mcc"]         == 1.0
        assert m["f1"]          == 1.0

    def test_worst_prediction(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        m = classification_metrics(y_true, y_pred)
        assert m["acc"]  == 0.0
        assert m["mcc"]  == -1.0

    def test_all_keys_present(self):
        y = np.array([1, 0, 1, 0])
        m = classification_metrics(y, y)
        for key in ["acc", "bacc", "sensitivity", "specificity",
                    "ppv", "npv", "mcc", "auc", "f1"]:
            assert key in m

    def test_auc_nan_without_proba(self):
        y = np.array([1, 0, 1, 0])
        m = classification_metrics(y, y, y_prob=None)
        assert np.isnan(m["auc"])

    def test_auc_with_proba(self):
        y    = np.array([1, 1, 0, 0])
        prob = np.array([0.9, 0.8, 0.2, 0.1])
        m = classification_metrics(y, (prob > 0.5).astype(int), y_prob=prob)
        assert m["auc"] == pytest.approx(1.0, abs=0.001)

    def test_values_bounded_zero_one(self):
        rng    = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.integers(0, 2, 100)
        m = classification_metrics(y_true, y_pred)
        for key in ["acc", "bacc", "sensitivity", "specificity", "ppv", "npv", "f1"]:
            assert 0.0 <= m[key] <= 1.0


class TestRegressionMetrics:

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = regression_metrics(y, y)
        assert m["r2"]   == pytest.approx(1.0)
        assert m["mae"]  == pytest.approx(0.0)
        assert m["mse"]  == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)

    def test_all_keys_present(self):
        y = np.array([1.0, 2.0, 3.0])
        m = regression_metrics(y, y)
        for key in ["r2", "mae", "mse", "rmse", "explained_var"]:
            assert key in m

    def test_rmse_equals_sqrt_mse(self):
        y_true = np.array([1.0, 2.5, 3.1])
        y_pred = np.array([1.2, 2.3, 3.4])
        m = regression_metrics(y_true, y_pred)
        assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]), abs=1e-6)

    def test_mae_lower_bound(self):
        y = np.array([1.0, 2.0])
        m = regression_metrics(y, y)
        assert m["mae"] >= 0.0

    def test_r2_negative_for_bad_model(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])   # reversed
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] < 0.0
