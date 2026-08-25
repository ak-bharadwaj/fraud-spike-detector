"""Comprehensive behavioral unit tests for Day 5 HybridEWMAScorer.

Validates all 15 required scoring behavioral dimensions:
1. Exact standardized magnitude calculation (M_k = |f_k - expected_k| / scale_k).
2. Max-feature aggregation (S_raw = max_k M_k across all 11 features).
3. Zero-scale handling (raises ValueError if scale <= 0.0).
4. Explicit feature mapping (all 11 features mapped).
5. Missing baseline feature failure (raises KeyError if baseline expectation/scale missing).
6. EWMA progression (S_ewma,t = alpha * S_raw,t + (1 - alpha) * S_ewma,t-1).
7. EWMA merchant isolation (Merchant A vs B independence).
8. INSUFFICIENT evidence -> EWMA state reset.
9. DEGRADED evidence state mapping (confidence = 0.5, data_quality = "DEGRADED").
10. SUFFICIENT evidence state mapping (confidence = 1.0, data_quality = "GOOD").
11. Persistence ownership (documented for Day 6 AlertStateMachine).
12. Deterministic scoring replay.
13. GroundTruth isolation AST check.
14. Holdout isolation AST check.
15. RiskScore Pydantic schema compliance.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import numpy as np
import pytest

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.contracts.config_schemas import DetectorConfig, ScorerConfig, EvidenceConfig, StateMachineConfig
from src.scoring.hybrid_ewma import HybridEWMAScorer, FEATURE_BASELINE_MAP


# =====================================================================
# Helpers to create FeatureSnapshot and BaselineSnapshot
# =====================================================================

def make_dummy_feature(
    merchant_id: str,
    ts: datetime,
    volume: float = 10.0,
    velocity: float = 2.0,
    mean_amount: float = 100.0,
    data_quality: str = "GOOD",
) -> FeatureSnapshot:
    return FeatureSnapshot(
        merchant_id=merchant_id,
        timestamp=ts,
        volume=volume,
        velocity=velocity,
        amount_statistics={
            "total_amount": volume * mean_amount,
            "mean_amount": mean_amount,
            "std_amount": 10.0,
            "median_amount": mean_amount,
            "mad_amount": 5.0,
            "min_amount": mean_amount - 15.0,
            "max_amount": mean_amount + 15.0,
        },
        unique_customers=int(volume * 0.8),
        unique_devices=int(volume * 0.5),
        data_quality=data_quality,
    )


def make_dummy_baseline(
    merchant_id: str,
    ts: datetime,
    evidence_state: str = "SUFFICIENT",
    exp_volume: float = 10.0,
    scale_volume: float = 2.0,
) -> BaselineSnapshot:
    return BaselineSnapshot(
        merchant_id=merchant_id,
        timestamp=ts,
        expected_values={
            "volume": exp_volume,
            "velocity": exp_volume / 5.0,
            "unique_customers": exp_volume * 0.8,
            "unique_devices": exp_volume * 0.5,
            "amount_total_amount": exp_volume * 100.0,
            "amount_mean_amount": 100.0,
            "amount_std_amount": 10.0,
            "amount_median_amount": 100.0,
            "amount_mad_amount": 5.0,
            "amount_min_amount": 85.0,
            "amount_max_amount": 115.0,
        },
        robust_scale={
            "volume": scale_volume,
            "velocity": scale_volume / 5.0,
            "unique_customers": scale_volume * 0.8,
            "unique_devices": scale_volume * 0.5,
            "amount_total_amount": scale_volume * 100.0,
            "amount_mean_amount": 10.0,
            "amount_std_amount": 5.0,
            "amount_median_amount": 10.0,
            "amount_mad_amount": 5.0,
            "amount_min_amount": 10.0,
            "amount_max_amount": 10.0,
        },
        history_count=50,
        current_window_count=int(exp_volume),
        evidence_state=evidence_state,
    )


# =====================================================================
# 1. Exact Standardized Magnitude & 2. Max Aggregation
# =====================================================================

def test_exact_standardized_magnitude_and_max_aggregation():
    """Verify exact M_k calculation and max aggregation S_raw = max_k M_k."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Volume: f_val = 30.0, exp = 10.0, scale = 2.0 -> M_vol = (30-10)/2 = 10.0
    feat = make_dummy_feature("M1", st, volume=30.0, velocity=6.0)
    base = make_dummy_baseline("M1", st, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)

    risk = scorer.calculate_score(feat, base)

    # Max magnitude is M_vol = 10.0
    assert math.isclose(risk.score, 10.0, abs_tol=1e-4)


# =====================================================================
# 3. Zero-Scale Handling (Raises ValueError)
# =====================================================================

def test_zero_robust_scale_raises_value_error():
    """Verify zero or negative robust scale raises explicit ValueError."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    feat = make_dummy_feature("M1", st)
    base = make_dummy_baseline("M1", st)
    base.robust_scale["volume"] = 0.0  # Zero scale violation

    with pytest.raises(ValueError, match="Invalid non-positive robust scale for feature 'volume'"):
        scorer.calculate_score(feat, base)


# =====================================================================
# 4. Explicit Feature Mapping & 5. Missing Baseline Feature Failure
# =====================================================================

def test_explicit_feature_mapping_and_missing_baseline_feature_raises_key_error():
    """Verify all 11 required features are mapped and missing baseline feature raises KeyError."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    assert len(FEATURE_BASELINE_MAP) == 11

    feat = make_dummy_feature("M1", st)
    base = make_dummy_baseline("M1", st)
    del base.expected_values["amount_mean_amount"]  # Delete required baseline expectation

    with pytest.raises(KeyError, match="Missing required baseline feature expectation/scale for 'amount_mean_amount'"):
        scorer.calculate_score(feat, base)


# =====================================================================
# 6. EWMA Progression & 7. EWMA Merchant Isolation
# =====================================================================

def test_ewma_smoothing_progression_and_merchant_isolation():
    """Verify EWMA update equation and merchant state isolation."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Merchant A step 1 (raw = 10.0) -> EWMA_A = 10.0
    feat_a1 = make_dummy_feature("M_A", st, volume=30.0)
    base_a1 = make_dummy_baseline("M_A", st, exp_volume=10.0, scale_volume=2.0)
    risk_a1 = scorer.calculate_score(feat_a1, base_a1)
    assert math.isclose(risk_a1.score, 10.0, abs_tol=1e-4)

    # Merchant B step 1 (raw = 5.0) -> EWMA_B = 5.0 (isolated from Merchant A)
    feat_b1 = make_dummy_feature("M_B", st, volume=20.0)
    base_b1 = make_dummy_baseline("M_B", st, exp_volume=10.0, scale_volume=2.0)
    risk_b1 = scorer.calculate_score(feat_b1, base_b1)
    assert math.isclose(risk_b1.score, 5.0, abs_tol=1e-4)

    # Merchant A step 2 (raw = 0.0) -> EWMA_A = 0.3 * 0.0 + 0.7 * 10.0 = 7.0
    st2 = st + timedelta(minutes=1)
    feat_a2 = make_dummy_feature("M_A", st2, volume=10.0)
    base_a2 = make_dummy_baseline("M_A", st2, exp_volume=10.0, scale_volume=2.0)
    risk_a2 = scorer.calculate_score(feat_a2, base_a2)
    assert math.isclose(risk_a2.score, 7.0, abs_tol=1e-4)


# =====================================================================
# 8. INSUFFICIENT Evidence EWMA State Reset
# =====================================================================

def test_insufficient_evidence_resets_ewma_state():
    """Verify INSUFFICIENT evidence state resets merchant EWMA state preventing stale state leakage."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Step 1: Merchant A gets high EWMA score = 10.0
    feat1 = make_dummy_feature("M1", st, volume=30.0)
    base1 = make_dummy_baseline("M1", st, exp_volume=10.0, scale_volume=2.0)
    scorer.calculate_score(feat1, base1)
    assert "M1" in scorer._ewma_states

    # Step 2: Evidence gap (INSUFFICIENT) -> EWMA state for M1 must be reset/popped
    st2 = st + timedelta(minutes=1)
    feat2 = make_dummy_feature("M1", st2, volume=10.0)
    base2 = make_dummy_baseline("M1", st2, evidence_state="INSUFFICIENT")
    risk2 = scorer.calculate_score(feat2, base2)

    assert risk2.score is None
    assert risk2.confidence == 0.0
    assert "M1" not in scorer._ewma_states  # State reset!

    # Step 3: Evidence resumes (raw score = 5.0) -> EWMA starts fresh at 5.0, NOT 0.3*5 + 0.7*10 (6.5)
    st3 = st + timedelta(minutes=2)
    feat3 = make_dummy_feature("M1", st3, volume=20.0)
    base3 = make_dummy_baseline("M1", st3, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)
    risk3 = scorer.calculate_score(feat3, base3)

    assert math.isclose(risk3.score, 5.0, abs_tol=1e-4)


# =====================================================================
# 9. DEGRADED Evidence Mapping & 10. SUFFICIENT Evidence Mapping
# =====================================================================

def test_degraded_and_sufficient_evidence_state_mappings():
    """Verify DEGRADED state yields confidence=0.5 and data_quality='DEGRADED', SUFFICIENT yields confidence=1.0 and data_quality='GOOD'."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    feat = make_dummy_feature("M1", st, volume=20.0)

    # DEGRADED
    base_deg = make_dummy_baseline("M1", st, evidence_state="DEGRADED", exp_volume=10.0, scale_volume=2.0)
    risk_deg = scorer.calculate_score(feat, base_deg)
    assert risk_deg.confidence == 0.5
    assert risk_deg.data_quality == "DEGRADED"

    # Reset
    scorer.reset("M1")

    # SUFFICIENT
    base_suf = make_dummy_baseline("M1", st, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)
    risk_suf = scorer.calculate_score(feat, base_suf)
    assert risk_suf.confidence == 1.0
    assert risk_suf.data_quality == "GOOD"


# =====================================================================
# 11. Persistence Ownership Documentation Check
# =====================================================================

def test_persistence_ownership_documented_for_alert_state_machine():
    """Verify HybridEWMAScorer constructor does not accept dead persistence parameter."""
    with pytest.raises(TypeError):
        HybridEWMAScorer(alpha=0.3, static_threshold=3.5, persistence=2)  # persistence parameter removed from scorer


# =====================================================================
# 12. Deterministic Replay
# =====================================================================

def test_deterministic_scoring_replay():
    """Verify identical input sequence produces identical RiskScore outputs."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    scorer1 = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)
    feat = make_dummy_feature("M1", st, volume=20.0)
    base = make_dummy_baseline("M1", st, exp_volume=10.0, scale_volume=2.0)

    risk1 = scorer1.calculate_score(feat, base)

    scorer2 = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)
    risk2 = scorer2.calculate_score(feat, base)

    assert risk1 == risk2
    assert risk1.model_dump() == risk2.model_dump()


# =====================================================================
# 13. GroundTruth & 14. Holdout Isolation AST Check
# =====================================================================

def test_ground_truth_and_holdout_isolation_in_scoring_package():
    """Verify src/scoring and src/detector contain zero imports of ground_truth or holdout code."""
    for pkg_name in ["scoring", "detector"]:
        pkg_dir = Path(__file__).parent.parent / "src" / pkg_name
        py_files = list(pkg_dir.rglob("*.py"))

        assert len(py_files) > 0, f"No python files found in src/{pkg_name}"

        for file_path in py_files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "ground_truth" not in alias.name, f"GroundTruth import violation in {file_path}: {alias.name}"
                        assert "holdout" not in alias.name, f"Holdout import violation in {file_path}: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert "ground_truth" not in module, f"GroundTruth import violation in {file_path}: {module}"
                    assert "holdout" not in module, f"Holdout import violation in {file_path}: {module}"
                    for alias in node.names:
                        assert "GroundTruth" not in alias.name, f"GroundTruth element import violation in {file_path}: {alias.name}"
                        assert "holdout" not in alias.name and "Holdout" not in alias.name, f"Holdout element import violation in {file_path}: {alias.name}"


# =====================================================================
# 15. RiskScore Schema Validation
# =====================================================================

def test_risk_score_pydantic_schema_compliance():
    """Verify emitted RiskScore validates strictly against RiskScore Pydantic contract."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    feat = make_dummy_feature("M1", st, volume=20.0)
    base = make_dummy_baseline("M1", st, exp_volume=10.0, scale_volume=2.0)

    risk = scorer.calculate_score(feat, base)

    dumped = risk.model_dump()
    reconstructed = RiskScore(**dumped)

    assert reconstructed.score == risk.score
    assert reconstructed.confidence == 1.0
    assert reconstructed.data_quality == "GOOD"
