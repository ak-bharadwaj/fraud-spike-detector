"""Comprehensive behavioral unit tests for Day 5 HybridEWMAScorer.

Validates all scoring behavioral dimensions:
1. Exact standardized magnitude calculation (M_k = |f_k - expected_k| / scale_k).
2. Max-feature aggregation (S_raw = max_k M_k across all 11 features).
3. Zero-scale handling (raises ValueError if scale <= 0.0).
4. Explicit feature mapping (all 11 features mapped).
5. Missing baseline feature failure (raises KeyError if baseline expectation/scale missing).
6. EWMA progression (S_ewma,t = alpha * S_raw,t + (1 - alpha) * S_ewma,t-1).
7. EWMA merchant isolation (Merchant A vs B independence).
8. INSUFFICIENT evidence -> EWMA state reset and score=None.
9. DEGRADED evidence state mapping (confidence = 0.5, data_quality = "DEGRADED").
10. SUFFICIENT evidence state mapping (confidence = 1.0, data_quality = "GOOD").
11. RiskScore data_quality mapping matrix (EMPTY, INSUFFICIENT, DEGRADED, GOOD).
12. Persistence ownership (documented for Day 6 AlertStateMachine).
13. Deterministic scoring replay.
14. GroundTruth & Holdout isolation AST check.
15. RiskScore Pydantic schema compliance.
"""

import ast
from typing import Optional, List, Dict, Any, Union
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
    velocity: Optional[float] = None,
    mean_amount: float = 100.0,
    data_quality: str = "GOOD",
) -> FeatureSnapshot:
    vel = velocity if velocity is not None else (volume / 5.0)
    return FeatureSnapshot(
        merchant_id=merchant_id,
        timestamp=ts,
        volume=volume,
        velocity=vel,
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
# 1. Evidence-State Scoring Tests (INSUFFICIENT, DEGRADED, SUFFICIENT)
# =====================================================================

def test_evidence_state_insufficient_returns_none_score_and_resets_ewma():
    """Verify INSUFFICIENT evidence state returns score=None, confidence=0.0, triggered_signals=[], and resets EWMA state."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Establish initial EWMA state
    feat1 = make_dummy_feature("M1", st, volume=20.0)
    base1 = make_dummy_baseline("M1", st, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)
    scorer.calculate_score(feat1, base1)
    assert "M1" in scorer._ewma_states

    # INSUFFICIENT evidence gap
    st2 = st + timedelta(minutes=1)
    feat2 = make_dummy_feature("M1", st2, volume=100.0)
    base2 = make_dummy_baseline("M1", st2, evidence_state="INSUFFICIENT")

    risk = scorer.calculate_score(feat2, base2)

    assert risk.score is None
    assert risk.confidence == 0.0
    assert risk.triggered_signals == []
    assert risk.data_quality == "INSUFFICIENT"
    assert "M1" not in scorer._ewma_states  # EWMA state reset!


def test_evidence_state_degraded_returns_score_with_half_confidence():
    """Verify DEGRADED evidence state returns valid EWMA score with confidence=0.5 and data_quality='DEGRADED'."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    feat = make_dummy_feature("M1", st, volume=20.0)
    base = make_dummy_baseline("M1", st, evidence_state="DEGRADED", exp_volume=10.0, scale_volume=2.0)

    risk = scorer.calculate_score(feat, base)

    assert risk.score is not None
    assert math.isclose(risk.score, 5.0, abs_tol=1e-4)
    assert risk.confidence == 0.5
    assert risk.data_quality == "DEGRADED"


def test_evidence_state_sufficient_returns_score_with_full_confidence():
    """Verify SUFFICIENT evidence state returns valid EWMA score with confidence=1.0 and data_quality='GOOD'."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    feat = make_dummy_feature("M1", st, volume=20.0)
    base = make_dummy_baseline("M1", st, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)

    risk = scorer.calculate_score(feat, base)

    assert risk.score is not None
    assert math.isclose(risk.score, 5.0, abs_tol=1e-4)
    assert risk.confidence == 1.0
    assert risk.data_quality == "GOOD"


def test_risk_score_data_quality_mapping_matrix():
    """Verify all 4 rows of the RiskScore data_quality contract matrix."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # 1. GOOD + INSUFFICIENT -> INSUFFICIENT
    r1 = scorer.calculate_score(
        make_dummy_feature("M1", st, data_quality="GOOD"),
        make_dummy_baseline("M1", st, evidence_state="INSUFFICIENT"),
    )
    assert r1.data_quality == "INSUFFICIENT"
    assert r1.score is None

    # 2. EMPTY + INSUFFICIENT -> EMPTY
    r2 = scorer.calculate_score(
        make_dummy_feature("M1", st, volume=0.0, data_quality="EMPTY"),
        make_dummy_baseline("M1", st, evidence_state="INSUFFICIENT"),
    )
    assert r2.data_quality == "EMPTY"
    assert r2.score is None

    # 3. DEGRADED -> DEGRADED
    r3 = scorer.calculate_score(
        make_dummy_feature("M1", st, volume=2.0, data_quality="GOOD"),
        make_dummy_baseline("M1", st, evidence_state="DEGRADED"),
    )
    assert r3.data_quality == "DEGRADED"
    assert r3.confidence == 0.5

    # 4. SUFFICIENT -> GOOD
    r4 = scorer.calculate_score(
        make_dummy_feature("M1", st, volume=10.0, data_quality="GOOD"),
        make_dummy_baseline("M1", st, evidence_state="SUFFICIENT"),
    )
    assert r4.data_quality == "GOOD"
    assert r4.confidence == 1.0


# =====================================================================
# 2. Exact Standardized Magnitude & Max Aggregation
# =====================================================================

def test_exact_standardized_magnitude_and_max_aggregation():
    """Verify exact M_k calculation and max aggregation S_raw = max_k M_k."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    feat = make_dummy_feature("M1", st, volume=30.0, velocity=6.0)
    base = make_dummy_baseline("M1", st, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)

    risk = scorer.calculate_score(feat, base)

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
    base.robust_scale["volume"] = 0.0

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
    del base.expected_values["amount_mean_amount"]

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
# 8. Persistence Ownership Documentation Check
# =====================================================================

def test_persistence_ownership_documented_for_alert_state_machine():
    """Verify HybridEWMAScorer constructor does not accept dead persistence parameter."""
    with pytest.raises(TypeError):
        HybridEWMAScorer(alpha=0.3, static_threshold=3.5, persistence=2)


# =====================================================================
# 9. Deterministic Replay
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
# 10. GroundTruth & Holdout Isolation (AST Architectural Check)
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
# 11. RiskScore Schema Validation
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


# =====================================================================
# 12. Composite Confidence Contract Tests (Master Plan §17/§19)
# =====================================================================

def test_confidence_varies_independently_of_risk():
    """Verify confidence can vary independently of risk score (e.g. High Risk with Low Confidence)."""
    from src.scoring.statistical import StatisticalDeviationScorer

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = StatisticalDeviationScorer(static_threshold=3.5)

    base_suff = make_dummy_baseline("M1", st, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)
    base_deg = make_dummy_baseline("M1", st, evidence_state="DEGRADED", exp_volume=10.0, scale_volume=2.0)

    # 1. High Risk + High Confidence: true volume spike with good data quality, all signals agreeing
    feat_high_conf = make_dummy_feature("M1", st, volume=30.0, data_quality="GOOD")
    r1 = scorer.calculate_score(feat_high_conf, base_suff)
    assert r1.score == 10.0
    assert r1.confidence == 1.0

    # 2. High Risk + Low Confidence: same volume spike but degraded data quality
    feat_deg_quality = make_dummy_feature("M1", st, volume=30.0, data_quality="DEGRADED")
    r2 = scorer.calculate_score(feat_deg_quality, base_suff)
    assert r2.score == 10.0
    assert r2.confidence == 0.5

    # 3. High Risk + Partial Feature Availability: masked signals (1 of 4 groups active)
    r3 = scorer.calculate_score(feat_high_conf, base_suff, signal_mask=["volume"])
    assert r3.score == 10.0
    assert r3.confidence < 1.0  # Feature availability reduced!
    assert r3.confidence == 0.25

    # 4. High Risk + Weak Signal Agreement: isolated single-signal spike without corroboration
    # Only volume spikes, velocity and behavioral remain at expected baseline
    feat_isolated = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st,
        volume=30.0,
        velocity=2.0,  # nominal
        amount_statistics={
            "total_amount": 1000.0,
            "mean_amount": 100.0,
            "std_amount": 10.0,
            "median_amount": 100.0,
            "mad_amount": 5.0,
            "min_amount": 85.0,
            "max_amount": 115.0,
        },
        unique_customers=8,  # nominal
        unique_devices=5,  # nominal
        data_quality="GOOD",
    )
    r4 = scorer.calculate_score(feat_isolated, base_suff)
    assert r4.score == 10.0
    assert r4.confidence < 1.0  # Lower confidence due to lack of multi-signal agreement!
    assert r4.confidence == 0.75

    # 5. Low Risk + High Confidence: nominal unperturbed window
    feat_nominal = make_dummy_feature("M1", st, volume=10.0, data_quality="GOOD")
    r5 = scorer.calculate_score(feat_nominal, base_suff)
    assert r5.score == 0.0
    assert r5.confidence == 1.0
