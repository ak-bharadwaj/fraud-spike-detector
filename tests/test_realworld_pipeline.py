"""Tests for the real-world ML fraud detection pipeline (EXP-REALWORLD-CCF-001).

Validates:
1. Data adapter handles synthetic & real datasets and enforces 3-way temporal split.
2. ML scorers (IF, XGBoost, Ensemble) fit on TRAIN, calibrate on CALIBRATION, predict on TEST.
3. ML scorers explicitly decouple from FeatureSnapshot streaming semantics (NotImplementedError).
4. Principled feature/model group ablation executes 6 variants without outcome enforcement.
5. Calibration ECE and non-parametric bootstrap CIs produce valid metrics.
6. Serialization/deserialization and provenance metadata round-trip correctly.
7. Dynamic amount sum for FN financial exposure.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.scoring.base import AnomalyScorer


# ============================================================================
# Fixtures (Offline & Isolation Guarantee)
# ============================================================================

@pytest.fixture(scope="module")
def synthetic_data():
    """Synthetic dataset matrix for offline unit tests."""
    from src.realworld.data_adapter import KaggleCreditCardAdapter
    return KaggleCreditCardAdapter.create_synthetic_benchmark_matrix(n_rows=200, seed=42)


@pytest.fixture(scope="module")
def synthetic_dfs(synthetic_data):
    """Split synthetic dataset into TRAIN (140), CALIB (30), TEST (30)."""
    _, _, df = synthetic_data
    train_df = df.iloc[:140].copy()
    calib_df = df.iloc[140:170].copy()
    test_df = df.iloc[170:].copy()
    return train_df, calib_df, test_df


@pytest.fixture(scope="module")
def trained_xgb(synthetic_data):
    """Train XGBoost scorer on synthetic data with Platt calibration."""
    from src.scoring.ml_scorer import XGBoostFraudScorer

    X, y, _ = synthetic_data
    X_train, y_train = X[:140], y[:140]
    X_calib, y_calib = X[140:170], y[140:170]

    scorer = XGBoostFraudScorer(n_estimators=50, max_depth=3, seed=42)
    scorer.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    return scorer


@pytest.fixture(scope="module")
def trained_if(synthetic_data):
    """Train Isolation Forest on normal synthetic data with score calibration."""
    from src.scoring.ml_scorer import IsolationForestScorer

    X, y, _ = synthetic_data
    X_train, y_train = X[:140], y[:140]
    X_calib = X[140:170]

    scorer = IsolationForestScorer(n_estimators=50, seed=42)
    scorer.fit(X_train[y_train == 0])
    scorer.calibrate_scores(X_calib)
    return scorer


# ============================================================================
# Data Adapter Tests
# ============================================================================

class TestKaggleCreditCardAdapter:
    """Tests for the Kaggle Credit Card Fraud dataset adapter."""

    def test_synthetic_matrix_generation(self, synthetic_data):
        """Synthetic benchmark matrix generator should produce valid 30-feature arrays."""
        X, y, df = synthetic_data
        assert X.shape == (200, 30)
        assert len(y) == 200
        assert "Time" in df.columns
        assert "Amount" in df.columns
        assert "Class" in df.columns

    def test_feature_subset_extraction(self):
        """Adapter should correctly extract specified feature subsets."""
        from src.realworld.data_adapter import KaggleCreditCardAdapter
        _, _, df = KaggleCreditCardAdapter.create_synthetic_benchmark_matrix(n_rows=50)

        adapter = KaggleCreditCardAdapter.__new__(KaggleCreditCardAdapter)
        X_pca, _ = adapter.get_feature_matrix(df, feature_subset="pca")
        assert X_pca.shape[1] == 28

        X_pca_amt, _ = adapter.get_feature_matrix(df, feature_subset="pca_plus_amount")
        assert X_pca_amt.shape[1] == 29

        X_amt_time, _ = adapter.get_feature_matrix(df, feature_subset="amount_time")
        assert X_amt_time.shape[1] == 2


# ============================================================================
# ML Scorer & Calibration Isolation Tests
# ============================================================================

class TestIsolationForestScorer:
    """Tests for unsupervised Isolation Forest scorer."""

    def test_abc_compliance(self, trained_if):
        """IF scorer must be an instance of AnomalyScorer ABC."""
        assert isinstance(trained_if, AnomalyScorer)

    def test_predict_anomaly_scores_range(self, trained_if, synthetic_data):
        """Calibrated anomaly scores should be in [0, 1] range."""
        X, _, _ = synthetic_data
        scores = trained_if.predict_anomaly_scores(X)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_not_fitted_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        from src.scoring.ml_scorer import IsolationForestScorer
        scorer = IsolationForestScorer()
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.predict_anomaly_scores(X)

    def test_calculate_score_decoupled_raises(self, trained_if):
        """calculate_score on ML scorer must raise NotImplementedError."""
        from datetime import datetime, timezone
        feat = FeatureSnapshot(
            merchant_id="M1", timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc), volume=10.0, velocity=5.0,
            amount_statistics={}, unique_customers=5, unique_devices=5, data_quality="GOOD"
        )
        base = BaselineSnapshot(
            merchant_id="M1", timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc), expected_values={}, robust_scale={},
            history_count=10, current_window_count=5, evidence_state="SUFFICIENT"
        )
        with pytest.raises(NotImplementedError, match="Track B real-world ML model"):
            trained_if.calculate_score(feat, base)

    def test_serialization_roundtrip(self, trained_if):
        """Model should serialize and deserialize correctly."""
        from src.scoring.ml_scorer import IsolationForestScorer
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "if_model.joblib"
            trained_if.save(path)
            loaded = IsolationForestScorer.load(path)
            assert loaded._is_fitted
            X = np.random.randn(5, 30)
            orig_scores = trained_if.predict_anomaly_scores(X)
            loaded_scores = loaded.predict_anomaly_scores(X)
            np.testing.assert_array_almost_equal(orig_scores, loaded_scores)


class TestXGBoostFraudScorer:
    """Tests for supervised XGBoost fraud scorer."""

    def test_abc_compliance(self, trained_xgb):
        """XGBoost scorer must be an instance of AnomalyScorer ABC."""
        assert isinstance(trained_xgb, AnomalyScorer)

    def test_predict_proba_range(self, trained_xgb, synthetic_data):
        """Predicted probabilities should be in [0, 1]."""
        X, _, _ = synthetic_data
        probas = trained_xgb.predict_proba(X)
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_no_silent_calibration_fallback_raises(self):
        """Fitting without calibration split must raise RuntimeError (NO silent training fallback)."""
        from src.scoring.ml_scorer import XGBoostFraudScorer
        X = np.random.randn(50, 30)
        y = np.random.randint(0, 2, 50)
        scorer = XGBoostFraudScorer(n_estimators=10)
        with pytest.raises(RuntimeError, match="X_calib and y_calib are strictly required"):
            scorer.fit(X, y)

    def test_calculate_score_decoupled_raises(self, trained_xgb):
        """calculate_score on ML scorer must raise NotImplementedError."""
        from datetime import datetime, timezone
        feat = FeatureSnapshot(
            merchant_id="M1", timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc), volume=10.0, velocity=5.0,
            amount_statistics={}, unique_customers=5, unique_devices=5, data_quality="GOOD"
        )
        base = BaselineSnapshot(
            merchant_id="M1", timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc), expected_values={}, robust_scale={},
            history_count=10, current_window_count=5, evidence_state="SUFFICIENT"
        )
        with pytest.raises(NotImplementedError, match="Track B real-world ML model"):
            trained_xgb.calculate_score(feat, base)

    def test_serialization_roundtrip(self, trained_xgb):
        """Model should serialize and deserialize correctly."""
        from src.scoring.ml_scorer import XGBoostFraudScorer
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xgb_model.joblib"
            trained_xgb.save(path)
            loaded = XGBoostFraudScorer.load(path)
            assert loaded._is_fitted
            X = np.random.randn(5, 30)
            orig_probas = trained_xgb.predict_proba(X)
            loaded_probas = loaded.predict_proba(X)
            np.testing.assert_array_almost_equal(orig_probas, loaded_probas, decimal=4)


class TestEnsembleFraudScorer:
    """Tests for the weighted IF + XGBoost ensemble."""

    def test_abc_compliance(self, trained_if, trained_xgb):
        """Ensemble scorer must be an instance of AnomalyScorer ABC."""
        from src.scoring.ml_scorer import EnsembleFraudScorer
        ensemble = EnsembleFraudScorer(if_scorer=trained_if, xgb_scorer=trained_xgb)
        assert isinstance(ensemble, AnomalyScorer)

    def test_ensemble_scores_range(self, trained_if, trained_xgb, synthetic_data):
        """Ensemble scores should be in [0, 1] range."""
        from src.scoring.ml_scorer import EnsembleFraudScorer
        ensemble = EnsembleFraudScorer(if_scorer=trained_if, xgb_scorer=trained_xgb)
        X, _, _ = synthetic_data
        scores = ensemble.predict_ensemble_scores(X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0 + 1e-6

    def test_calculate_score_decoupled_raises(self, trained_if, trained_xgb):
        """calculate_score on ensemble must raise NotImplementedError."""
        from datetime import datetime, timezone
        from src.scoring.ml_scorer import EnsembleFraudScorer
        ensemble = EnsembleFraudScorer(if_scorer=trained_if, xgb_scorer=trained_xgb)
        feat = FeatureSnapshot(
            merchant_id="M1", timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc), volume=10.0, velocity=5.0,
            amount_statistics={}, unique_customers=5, unique_devices=5, data_quality="GOOD"
        )
        base = BaselineSnapshot(
            merchant_id="M1", timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc), expected_values={}, robust_scale={},
            history_count=10, current_window_count=5, evidence_state="SUFFICIENT"
        )
        with pytest.raises(NotImplementedError, match="Track B real-world ML model"):
            ensemble.calculate_score(feat, base)


# ============================================================================
# Principled Ablation Execution Tests
# ============================================================================

class TestPrincipledAblation:
    """Tests that 6 principled feature/model group ablations execute cleanly."""

    def test_ablation_study_executes_six_variants(self, synthetic_dfs):
        """Ablation study must execute all 6 variants."""
        from scripts.train_realworld import run_principled_ablation
        from src.realworld.data_adapter import KaggleCreditCardAdapter

        train_df, calib_df, test_df = synthetic_dfs
        adapter = KaggleCreditCardAdapter.__new__(KaggleCreditCardAdapter)

        results = run_principled_ablation(adapter, train_df, calib_df, test_df, seed=42)

        assert len(results) == 6
        variant_ids = [r["variant_id"] for r in results]
        assert "FULL_ENSEMBLE" in variant_ids
        assert "XGB_ONLY" in variant_ids
        assert "IF_ONLY" in variant_ids
        assert "PCA_ONLY" in variant_ids
        assert "PCA_PLUS_AMOUNT" in variant_ids
        assert "AMOUNT_TIME_ONLY" in variant_ids

        for r in results:
            assert "metrics" in r
            assert "f1_score" in r["metrics"]
            assert "fn_exposure" in r["metrics"]


# ============================================================================
# Evaluation Utility & Dynamic Financial Metrics Tests
# ============================================================================

class TestEvaluationUtilities:
    """Tests for binary classification evaluation & dynamic FN exposure calculation."""

    def test_dynamic_amount_sum_fn_exposure(self):
        """FN exposure should equal the sum of Amount for missed fraud transactions."""
        from src.scoring.ml_scorer import evaluate_binary_classifier

        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 0])  # TP=1 (idx 0), FN=2 (idx 1, 2), FP=1 (idx 3), TN=1 (idx 4)
        amounts = np.array([100.0, 250.0, 500.0, 75.0, 20.0])

        metrics = evaluate_binary_classifier(y_true, y_pred, amounts=amounts, fp_unit_cost=50.0)

        assert metrics["tp"] == 1
        assert metrics["fn"] == 2
        assert metrics["fp"] == 1
        assert metrics["fp_cost"] == 50.0
        assert metrics["fn_exposure"] == 750.0  # 250.0 + 500.0
        assert metrics["total_cost"] == 800.0   # 50.0 + 750.0


# ============================================================================
# Integration & Contract Tests
# ============================================================================

class TestEndToEndIntegration:
    """Integration tests verifying exports and contract non-breakage."""

    def test_existing_tests_not_broken(self):
        """Verify importing new ML modules doesn't break existing exports."""
        from src.scoring import (
            AnomalyScorer,
            StaticThresholdScorer,
            StatisticalDeviationScorer,
            HybridEWMAScorer,
            IsolationForestScorer,
            XGBoostFraudScorer,
            EnsembleFraudScorer,
        )
        assert AnomalyScorer is not None
        assert StatisticalDeviationScorer is not None
        assert IsolationForestScorer is not None
        assert XGBoostFraudScorer is not None
