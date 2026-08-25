"""Comprehensive behavioral unit tests for Day 5 HybridEWMAScorer.

Validates all scoring behavioral dimensions:
1. RiskScore Pydantic schema compliance (score: Optional[float] = None).
2. Evidence-state scoring (INSUFFICIENT -> score=None, confidence=0.0).
3. Degraded evidence scoring (DEGRADED -> score=S_ewma, confidence=0.5).
4. Sufficient evidence scoring (SUFFICIENT -> score=S_ewma, confidence=1.0).
5. Standardized magnitude calculation M_k = |f_k - expected_k| / scale_k.
6. EWMA smoothing progression (S_ewma,t = alpha * S_raw,t + (1 - alpha) * S_ewma,t-1).
7. Triggered signals thresholding (M_k >= static_threshold).
8. Merchant EWMA state isolation (Merchant A vs B independence).
9. Deterministic scoring replay.
10. GroundTruth & Holdout isolation (zero ground-truth/holdout imports in src/scoring and src/detector).
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import numpy as np
import pytest

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.contracts.config_schemas import DetectorConfig, ScorerConfig, EvidenceConfig, StateMachineConfig
from src.scoring.hybrid_ewma import HybridEWMAScorer


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
# 1. Evidence-State Scoring Tests (INSUFFICIENT, DEGRADED, SUFFICIENT)
# =====================================================================

def test_evidence_state_insufficient_returns_none_score():
    """Verify INSUFFICIENT evidence state returns score=None and confidence=0.0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    feat = make_dummy_feature("M1", st, volume=100.0)  # Huge volume spike
    base = make_dummy_baseline("M1", st, evidence_state="INSUFFICIENT")

    risk = scorer.calculate_score(feat, base)

    assert risk.score is None
    assert risk.confidence == 0.0
    assert risk.triggered_signals == []
    assert risk.data_quality == "GOOD"


def test_evidence_state_degraded_returns_score_with_half_confidence():
    """Verify DEGRADED evidence state returns valid EWMA score with confidence=0.5 and data_quality='DEGRADED'."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Volume = 20.0, expected = 10.0, scale = 2.0 -> M_vol = (20-10)/2 = 5.0
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


# =====================================================================
# 2. Standardized Magnitude & Triggered Signals
# =====================================================================

def test_standardized_magnitude_and_triggered_signals():
    """Verify standardized deviation calculation M_k and static_threshold signal triggering."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Volume spike: volume = 20.0 (exp = 10.0, scale = 2.0) -> M_vol = 5.0 (>= 3.5 -> triggered)
    # Velocity spike: velocity = 4.0 (exp = 2.0, scale = 0.4) -> M_vel = 5.0 (>= 3.5 -> triggered)
    feat = make_dummy_feature("M1", st, volume=20.0, velocity=4.0)
    base = make_dummy_baseline("M1", st, evidence_state="SUFFICIENT", exp_volume=10.0, scale_volume=2.0)

    risk = scorer.calculate_score(feat, base)

    assert "volume" in risk.triggered_signals
    assert "velocity" in risk.triggered_signals


# =====================================================================
# 3. EWMA Smoothing Progression
# =====================================================================

def test_ewma_smoothing_progression():
    """Verify EWMA update equation: S_ewma,t = alpha * S_raw,t + (1 - alpha) * S_ewma,t-1."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Step 1: Raw score = 10.0 -> EWMA1 = 10.0
    feat1 = make_dummy_feature("M1", st, volume=30.0)
    base1 = make_dummy_baseline("M1", st, exp_volume=10.0, scale_volume=2.0)
    risk1 = scorer.calculate_score(feat1, base1)

    assert math.isclose(risk1.score, 10.0, abs_tol=1e-4)

    # Step 2: Raw score = 0.0 -> EWMA2 = 0.3 * 0.0 + 0.7 * 10.0 = 7.0
    st2 = st + timedelta(minutes=1)
    feat2 = make_dummy_feature("M1", st2, volume=10.0)
    base2 = make_dummy_baseline("M1", st2, exp_volume=10.0, scale_volume=2.0)
    risk2 = scorer.calculate_score(feat2, base2)

    assert math.isclose(risk2.score, 7.0, abs_tol=1e-4)


# =====================================================================
# 4. Merchant EWMA State Isolation
# =====================================================================

def test_merchant_ewma_state_isolation():
    """Verify updating Merchant A EWMA state does not affect Merchant B EWMA state."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    # Merchant A score = 10.0
    feat_a = make_dummy_feature("M_A", st, volume=30.0)
    base_a = make_dummy_baseline("M_A", st, exp_volume=10.0, scale_volume=2.0)
    scorer.calculate_score(feat_a, base_a)

    # Merchant B first score (raw = 5.0) -> EWMA_B should be 5.0, NOT affected by M_A (10.0)
    feat_b = make_dummy_feature("M_B", st, volume=20.0)
    base_b = make_dummy_baseline("M_B", st, exp_volume=10.0, scale_volume=2.0)
    risk_b = scorer.calculate_score(feat_b, base_b)

    assert math.isclose(risk_b.score, 5.0, abs_tol=1e-4)


# =====================================================================
# 5. Deterministic Replay
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
# 6. GroundTruth & Holdout Isolation (AST Architectural Check)
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
# 7. RiskScore Schema Validation
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
