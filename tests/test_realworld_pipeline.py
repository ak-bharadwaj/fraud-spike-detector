"""Tests for the real-world ML fraud detection pipeline (EXP-REALWORLD-CCF-001).

Validates:
1. Data adapter loads dataset and enforces 3-way temporal split (TRAIN 70% / CALIB 15% / TEST 15%).
2. Evaluates raw transaction features without artificial merchant clustering.
3. ML scorers (IF, XGBoost, Ensemble) fit on TRAIN, calibrate on CALIBRATION, predict on TEST.
4. All ML scorers conform to AnomalyScorer ABC interface.
5. Principled feature/model group ablation executes and records honest ΔF1 metrics.
6. Calibration ECE on locked TEST set is recorded.
7. Non-parametric bootstrap CIs produce valid 95% intervals.
8. Serialization/deserialization and provenance manifest generation round-trip correctly.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.scoring.base import AnomalyScorer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def adapter():
    """Load the Kaggle dataset adapter (module-scoped for performance)."""
    from src.realworld.data_adapter import KaggleCreditCardAdapter
    return KaggleCreditCardAdapter(seed=42)


@pytest.fixture(scope="module")
def dataset(adapter):
    """Load the full dataset."""
    return adapter.load()


@pytest.fixture(scope="module")
def three_way_split(adapter):
    """Create 3-way temporal split (TRAIN 70% / CALIB 15% / TEST 15%)."""
    return adapter.temporal_three_way_split(train_ratio=0.70, calib_ratio=0.15)


@pytest.fixture(scope="module")
def trained_xgb(adapter, three_way_split):
    """Train XGBoost scorer on TRAIN data and calibrate on CALIBRATION split."""
    from src.scoring.ml_scorer import XGBoostFraudScorer

    train_df, calib_df, _ = three_way_split
    X_train, y_train = adapter.get_feature_matrix(train_df)
    X_calib, y_calib = adapter.get_feature_matrix(calib_df)
    feature_names = adapter.get_feature_names()

    scorer = XGBoostFraudScorer(n_estimators=100, max_depth=4, seed=42)
    scorer.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib, feature_names=feature_names)
    return scorer


@pytest.fixture(scope="module")
def trained_if(adapter, three_way_split):
    """Train Isolation Forest on normal TRAIN transactions and calibrate on CALIBRATION split."""
    from src.scoring.ml_scorer import IsolationForestScorer

    train_df, calib_df, _ = three_way_split
    X_train, y_train = adapter.get_feature_matrix(train_df)
    X_calib, _ = adapter.get_feature_matrix(calib_df)
    X_normal = X_train[y_train == 0]
    feature_names = adapter.get_feature_names()

    scorer = IsolationForestScorer(n_estimators=100, seed=42)
    scorer.fit(X_normal, feature_names=feature_names)
    scorer.calibrate_scores(X_calib)
    return scorer


# ============================================================================
# Data Adapter Tests
# ============================================================================

class TestKaggleCreditCardAdapter:
    """Tests for the ULB / Kaggle Credit Card Fraud dataset adapter."""

    def test_load_dataset_shape(self, dataset):
        """Dataset should have 284,807 rows and include required columns."""
        assert len(dataset) == 284_807
        assert "Class" in dataset.columns
        assert "Time" in dataset.columns
        assert "Amount" in dataset.columns
        assert "timestamp" in dataset.columns

    def test_fraud_count(self, dataset):
        """Dataset should contain exactly 492 fraud transactions."""
        assert int(dataset["Class"].sum()) == 492

    def test_fraud_rate(self, dataset):
        """Fraud rate should be approximately 0.17%."""
        rate = dataset["Class"].mean()
        assert 0.001 < rate < 0.003

    def test_no_fake_merchant_clustering(self, dataset):
        """Adapter must evaluate raw transactions without inventing fake merchant IDs."""
        assert "merchant_id" not in dataset.columns or dataset["merchant_id"].nunique() == 1

    def test_three_way_split_no_leakage(self, adapter, three_way_split):
        """Splits must be strictly temporal — zero time overlap between splits."""
        train_df, calib_df, test_df = three_way_split

        max_train_time = train_df["Time"].max()
        min_calib_time = calib_df["Time"].min()
        max_calib_time = calib_df["Time"].max()
        min_test_time = test_df["Time"].min()

        assert max_train_time <= min_calib_time, "Leakage between TRAIN and CALIBRATION"
        assert max_calib_time <= min_test_time, "Leakage between CALIBRATION and TEST"

    def test_three_way_split_ratios(self, three_way_split):
        """Ratios should be ~70% TRAIN, ~15% CALIB, ~15% TEST."""
        train_df, calib_df, test_df = three_way_split
        total = len(train_df) + len(calib_df) + len(test_df)

        assert 0.68 < len(train_df) / total < 0.72
        assert 0.13 < len(calib_df) / total < 0.17
        assert 0.13 < len(test_df) / total < 0.17

    def test_feature_matrix_shape(self, adapter, dataset):
        """Feature matrix should have 30 columns (Time + Amount + V1-V28)."""
        X, y = adapter.get_feature_matrix(dataset)
        assert X.shape[0] == 284_807
        assert X.shape[1] == 30
        assert y.shape[0] == 284_807

    def test_dataset_manifest(self, adapter):
        """Dataset manifest should contain complete metadata and SHA-256 hash."""
        manifest = adapter.get_dataset_manifest()
        assert manifest["total_transactions"] == 284_807
        assert manifest["total_fraud"] == 492
        assert "dataset_sha256" in manifest
        assert "splits" in manifest
        assert "train" in manifest["splits"]
        assert "calibration" in manifest["splits"]
        assert "test" in manifest["splits"]


# ============================================================================
# ML Scorer Tests
# ============================================================================

class TestIsolationForestScorer:
    """Tests for unsupervised Isolation Forest scorer."""

    def test_abc_compliance(self, trained_if):
        """IF scorer must be an instance of AnomalyScorer ABC."""
        assert isinstance(trained_if, AnomalyScorer)

    def test_predict_anomaly_scores_range(self, trained_if, adapter, three_way_split):
        """Calibrated anomaly scores should be in [0, 1] range."""
        _, _, test_df = three_way_split
        X_test, _ = adapter.get_feature_matrix(test_df)
        scores = trained_if.predict_anomaly_scores(X_test)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(test_df)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_predict_labels(self, trained_if, adapter, three_way_split):
        """Labels should be binary (0 or 1)."""
        _, _, test_df = three_way_split
        X_test, _ = adapter.get_feature_matrix(test_df)
        labels = trained_if.predict_labels(X_test)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_not_fitted_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        from src.scoring.ml_scorer import IsolationForestScorer
        scorer = IsolationForestScorer()
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.predict_anomaly_scores(X)

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

    def test_predict_proba_range(self, trained_xgb, adapter, three_way_split):
        """Predicted probabilities should be in [0, 1]."""
        _, _, test_df = three_way_split
        X_test, _ = adapter.get_feature_matrix(test_df)
        probas = trained_xgb.predict_proba(X_test)
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_predict_labels_binary(self, trained_xgb, adapter, three_way_split):
        """Labels should be binary."""
        _, _, test_df = three_way_split
        X_test, _ = adapter.get_feature_matrix(test_df)
        labels = trained_xgb.predict_labels(X_test)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_feature_importance(self, trained_xgb):
        """Feature importance should be available after fitting."""
        importance = trained_xgb.get_feature_importance()
        assert len(importance) == 30
        assert all(v >= 0 for v in importance.values())

    def test_f1_above_minimum(self, trained_xgb, adapter, three_way_split):
        """XGBoost F1 on test set should be above a minimum threshold."""
        from src.scoring.ml_scorer import evaluate_binary_classifier
        _, _, test_df = three_way_split
        X_test, y_test = adapter.get_feature_matrix(test_df)
        y_pred = trained_xgb.predict_labels(X_test)
        y_proba = trained_xgb.predict_proba(X_test)
        metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
        assert metrics["f1_score"] > 0.5, f"F1 too low: {metrics['f1_score']:.4f}"

    def test_auc_roc_above_minimum(self, trained_xgb, adapter, three_way_split):
        """AUC-ROC should be above 0.9 on this well-studied dataset."""
        from sklearn.metrics import roc_auc_score
        _, _, test_df = three_way_split
        X_test, y_test = adapter.get_feature_matrix(test_df)
        y_proba = trained_xgb.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba)
        assert auc > 0.9, f"AUC-ROC too low: {auc:.4f}"

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

    def test_ensemble_scores_range(self, trained_if, trained_xgb, adapter, three_way_split):
        """Ensemble scores should be in [0, 1] range."""
        from src.scoring.ml_scorer import EnsembleFraudScorer
        ensemble = EnsembleFraudScorer(if_scorer=trained_if, xgb_scorer=trained_xgb)
        _, _, test_df = three_way_split
        X_test, _ = adapter.get_feature_matrix(test_df)
        scores = ensemble.predict_ensemble_scores(X_test)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0 + 1e-6

    def test_serialization_roundtrip(self, trained_if, trained_xgb):
        """Ensemble should serialize and deserialize correctly."""
        from src.scoring.ml_scorer import EnsembleFraudScorer
        ensemble = EnsembleFraudScorer(if_scorer=trained_if, xgb_scorer=trained_xgb)
        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble.save(tmpdir)
            loaded = EnsembleFraudScorer.load(tmpdir)
            assert loaded.if_scorer is not None
            assert loaded.xgb_scorer is not None


# ============================================================================
# Principled Ablation Execution Tests
# ============================================================================

class TestPrincipledAblation:
    """Tests that principled feature/model group ablation executes and produces honest metrics."""

    def test_ablation_study_executes_honestly(self, adapter, three_way_split):
        """Ablation variants should execute and report metrics without outcome enforcement."""
        from scripts.train_realworld import run_principled_ablation

        train_df, calib_df, test_df = three_way_split
        results = run_principled_ablation(adapter, train_df, calib_df, test_df, seed=42)

        assert len(results) == 5
        variant_ids = [r["variant_id"] for r in results]
        assert "FULL_ENSEMBLE" in variant_ids
        assert "XGB_ONLY" in variant_ids
        assert "IF_ONLY" in variant_ids
        assert "PCA_ONLY" in variant_ids
        assert "AMOUNT_TIME_ONLY" in variant_ids

        # Verify metrics structure for each variant
        for r in results:
            assert "metrics" in r
            assert "f1_score" in r["metrics"]
            assert "precision" in r["metrics"]
            assert "recall" in r["metrics"]


# ============================================================================
# Calibration Tests
# ============================================================================

class TestCalibration:
    """Tests for probability calibration quality."""

    def test_calibration_ece_computed(self, trained_xgb, adapter, three_way_split):
        """ECE should be computed on locked TEST split."""
        from src.scoring.ml_scorer import compute_calibration_curve
        _, _, test_df = three_way_split
        X_test, y_test = adapter.get_feature_matrix(test_df)
        y_proba = trained_xgb.predict_proba(X_test)
        cal = compute_calibration_curve(y_test, y_proba, n_bins=10)
        assert cal["ece"] >= 0.0
        assert cal["ece"] < 0.30

    def test_calibration_bins_populated(self, trained_xgb, adapter, three_way_split):
        """At least some calibration bins should be populated."""
        from src.scoring.ml_scorer import compute_calibration_curve
        _, _, test_df = three_way_split
        X_test, y_test = adapter.get_feature_matrix(test_df)
        y_proba = trained_xgb.predict_proba(X_test)
        cal = compute_calibration_curve(y_test, y_proba, n_bins=10)
        populated = sum(1 for b in cal["bins"] if b["n_samples"] > 0)
        assert populated >= 2


# ============================================================================
# Bootstrap CI Tests
# ============================================================================

class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_produces_valid_intervals(self, trained_xgb, adapter, three_way_split):
        """Bootstrap CIs should have lower <= upper and be in [0, 1]."""
        _, _, test_df = three_way_split
        X_test, y_test = adapter.get_feature_matrix(test_df)
        y_pred = trained_xgb.predict_labels(X_test)
        y_proba = trained_xgb.predict_proba(X_test)

        from scripts.train_realworld import run_bootstrap_ci
        bootstrap = run_bootstrap_ci(y_test, y_pred, y_proba, n_bootstrap=100, seed=42)

        for metric in ["precision", "recall", "f1"]:
            ci = bootstrap[metric]
            assert ci["ci_lower"] <= ci["ci_upper"]
            assert ci["ci_lower"] >= 0.0
            assert ci["ci_upper"] <= 1.0


# ============================================================================
# Evaluation Utility Tests
# ============================================================================

class TestEvaluationUtilities:
    """Tests for binary classification evaluation functions."""

    def test_evaluate_perfect_classifier(self):
        """Perfect classifier should get precision=1, recall=1, F1=1."""
        from src.scoring.ml_scorer import evaluate_binary_classifier
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0])
        metrics = evaluate_binary_classifier(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_cost_model(self):
        """FP cost and FN exposure should be calculated correctly."""
        from src.scoring.ml_scorer import evaluate_binary_classifier
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0])  # 1 FP, 1 FN
        metrics = evaluate_binary_classifier(y_true, y_pred, fp_unit_cost=50.0, fn_exposure_factor=800.0)
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1
        assert metrics["fp_cost"] == 50.0
        assert metrics["fn_exposure"] == 800.0
        assert metrics["total_cost"] == 850.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """Integration tests for the full pipeline."""

    def test_existing_tests_not_broken(self):
        """Verify importing new ML modules doesn't break existing imports."""
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

    def test_anomaly_scorer_abc_calculate_score(self, trained_xgb):
        """ML scorers should accept FeatureSnapshot/BaselineSnapshot via ABC interface."""
        from datetime import datetime, timezone

        feat = FeatureSnapshot(
            merchant_id="TEST-001",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            volume=100.0,
            velocity=50.0,
            amount_statistics={
                "total_amount": 5000.0, "mean_amount": 50.0, "std_amount": 10.0,
                "median_amount": 45.0, "mad_amount": 5.0, "min_amount": 10.0, "max_amount": 200.0,
            },
            unique_customers=20,
            unique_devices=15,
            data_quality="GOOD",
        )
        base = BaselineSnapshot(
            merchant_id="TEST-001",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            expected_values={"volume": 50.0, "velocity": 25.0},
            robust_scale={"volume": 10.0, "velocity": 5.0},
            history_count=10,
            current_window_count=5,
            evidence_state="SUFFICIENT",
        )

        result = trained_xgb.calculate_score(feat, base)
        assert isinstance(result, RiskScore)
        assert result.score is not None or result.confidence == 0.0
