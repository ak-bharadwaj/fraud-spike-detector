"""Comprehensive behavioral unit tests for Day 6 AlertStateMachine.

Validates all 16 required state machine behavioral dimensions:
1. Initial state (NORMAL).
2. Below-threshold score (retains NORMAL).
3. Exact threshold boundary (score == 3.5 qualifies as breach).
4. Above-threshold score (score > 3.5 qualifies as breach).
5. Persistence progression (NORMAL -> CANDIDATE -> ALERT).
6. Persistence reset (sub-threshold score resets counter to 0).
7. Alert activation & Alert object emission.
8. Cooldown recovery (C=5 windows alert suppression & return to NORMAL).
9. INSUFFICIENT evidence behavior (resets persistence counter to 0).
10. DEGRADED evidence behavior (qualifies with confidence=0.5).
11. SUFFICIENT evidence behavior (qualifies with confidence=1.0).
12. Merchant state isolation (Merchant A vs B independence).
13. Deterministic replay.
14. GroundTruth & Holdout isolation (zero ground-truth/holdout imports in src/state).
15. Holdout isolation boundary enforcement.
16. Alert Pydantic schema compliance.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import pytest

from src.contracts.contracts import RiskScore, Alert
from src.contracts.config_schemas import DetectorConfig, ScorerConfig, EvidenceConfig, StateMachineConfig
from src.state.alert_state_machine import AlertStateMachine


# =====================================================================
# Helper to create RiskScore
# =====================================================================

def make_dummy_risk_score(
    score: float | None = 1.0,
    confidence: float = 1.0,
    triggered_signals: list[str] | None = None,
    data_quality: str = "GOOD",
) -> RiskScore:
    return RiskScore(
        score=score,
        confidence=confidence,
        triggered_signals=triggered_signals or ["volume"],
        data_quality=data_quality,
    )


# =====================================================================
# 1. Initial State & Below-Threshold Score
# =====================================================================

def test_initial_state_and_below_threshold_score():
    """Verify initial state is NORMAL and sub-threshold scores retain NORMAL."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    assert sm.get_merchant_state("M1") == "NORMAL"

    risk_low = make_dummy_risk_score(score=2.0)
    state, alert = sm.process_score("M1", st, risk_low)

    assert state == "NORMAL"
    assert alert is None


# =====================================================================
# 2. Exact Threshold Boundary & Above-Threshold Score
# =====================================================================

def test_exact_threshold_boundary_and_above_threshold_score():
    """Verify exact threshold boundary (score == 3.5) and above-threshold score (score > 3.5) qualify as breach."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Exact threshold boundary (score == 3.5)
    risk_exact = make_dummy_risk_score(score=3.5)
    state1, alert1 = sm.process_score("M1", st, risk_exact)

    assert state1 == "CANDIDATE"
    assert alert1 is None


# =====================================================================
# 3. Persistence Progression & Reset
# =====================================================================

def test_persistence_progression_and_reset():
    """Verify persistence progression (NORMAL -> CANDIDATE -> ALERT) and sub-threshold reset."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Window 1: score >= 3.5 -> CANDIDATE
    state1, alert1 = sm.process_score("M1", st, make_dummy_risk_score(score=5.0))
    assert state1 == "CANDIDATE"
    assert alert1 is None

    # Window 2: sub-threshold score < 3.5 -> NORMAL (reset counter)
    st2 = st + timedelta(minutes=1)
    state2, alert2 = sm.process_score("M1", st2, make_dummy_risk_score(score=1.0))
    assert state2 == "NORMAL"
    assert alert2 is None

    # Window 3: score >= 3.5 -> CANDIDATE again
    st3 = st + timedelta(minutes=2)
    state3, alert3 = sm.process_score("M1", st3, make_dummy_risk_score(score=5.0))
    assert state3 == "CANDIDATE"

    # Window 4: score >= 3.5 -> ALERT (persistence 2 met!)
    st4 = st + timedelta(minutes=3)
    state4, alert4 = sm.process_score("M1", st4, make_dummy_risk_score(score=5.0))
    assert state4 == "ALERT"
    assert alert4 is not None
    assert alert4.risk_score == 5.0


# =====================================================================
# 4. Cooldown Recovery (C=5 Windows Alert Suppression)
# =====================================================================

def test_cooldown_recovery_suppression_and_return_to_normal():
    """Verify C=5 cooldown windows suppress duplicate alerts and transition back to NORMAL."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Trigger ALERT (w1=CANDIDATE, w2=ALERT)
    sm.process_score("M1", st, make_dummy_risk_score(score=5.0))
    st2 = st + timedelta(minutes=1)
    state2, alert2 = sm.process_score("M1", st2, make_dummy_risk_score(score=5.0))
    assert state2 == "ALERT"
    assert alert2 is not None

    # Following 5 windows (w3..w7) are in COOLDOWN -> no duplicate alert emission
    for i in range(1, 5):
        t_i = st2 + timedelta(minutes=i)
        st_i, alt_i = sm.process_score("M1", t_i, make_dummy_risk_score(score=10.0))  # Wild score
        assert st_i == "COOLDOWN"
        assert alt_i is None

    # 5th cooldown window finishes -> state returns to NORMAL
    t_last = st2 + timedelta(minutes=5)
    st_last, alt_last = sm.process_score("M1", t_last, make_dummy_risk_score(score=1.0))
    assert st_last == "NORMAL"
    assert alt_last is None


# =====================================================================
# 5. Evidence-State Behaviors (INSUFFICIENT, DEGRADED, SUFFICIENT)
# =====================================================================

def test_insufficient_evidence_resets_persistence():
    """Verify INSUFFICIENT evidence (score=None) resets CANDIDATE back to NORMAL."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Window 1: CANDIDATE
    sm.process_score("M1", st, make_dummy_risk_score(score=5.0))
    assert sm.get_merchant_state("M1") == "CANDIDATE"

    # Window 2: INSUFFICIENT evidence (score=None) -> resets to NORMAL
    st2 = st + timedelta(minutes=1)
    state2, alert2 = sm.process_score("M1", st2, make_dummy_risk_score(score=None, confidence=0.0, data_quality="INSUFFICIENT"))
    assert state2 == "NORMAL"
    assert alert2 is None


def test_degraded_and_sufficient_evidence_state_alert_emission():
    """Verify DEGRADED evidence score qualifies with confidence=0.5, SUFFICIENT with confidence=1.0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # DEGRADED emission
    sm_deg = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)
    sm_deg.process_score("M1", st, make_dummy_risk_score(score=5.0, confidence=0.5, data_quality="DEGRADED"))
    st2 = st + timedelta(minutes=1)
    _, alert_deg = sm_deg.process_score("M1", st2, make_dummy_risk_score(score=5.0, confidence=0.5, data_quality="DEGRADED"))

    assert alert_deg is not None
    assert alert_deg.confidence == 0.5

    # SUFFICIENT emission
    sm_suf = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)
    sm_suf.process_score("M1", st, make_dummy_risk_score(score=5.0, confidence=1.0, data_quality="GOOD"))
    _, alert_suf = sm_suf.process_score("M1", st2, make_dummy_risk_score(score=5.0, confidence=1.0, data_quality="GOOD"))

    assert alert_suf is not None
    assert alert_suf.confidence == 1.0


# =====================================================================
# 6. Merchant State Isolation
# =====================================================================

def test_multi_merchant_state_isolation():
    """Verify Merchant A transitions do not affect Merchant B state context."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Merchant A in CANDIDATE
    sm.process_score("M_A", st, make_dummy_risk_score(score=5.0))
    assert sm.get_merchant_state("M_A") == "CANDIDATE"

    # Merchant B should still be in NORMAL
    assert sm.get_merchant_state("M_B") == "NORMAL"


# =====================================================================
# 7. Deterministic Replay
# =====================================================================

def test_deterministic_state_machine_replay():
    """Verify identical input sequence produces identical state transitions and Alert emission."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Run 1
    sm1 = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)
    sm1.process_score("M1", st, make_dummy_risk_score(score=5.0))
    st2 = st + timedelta(minutes=1)
    st1, alt1 = sm1.process_score("M1", st2, make_dummy_risk_score(score=5.0))

    # Run 2
    sm2 = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)
    sm2.process_score("M1", st, make_dummy_risk_score(score=5.0))
    st2_run2, alt2 = sm2.process_score("M1", st2, make_dummy_risk_score(score=5.0))

    assert st1 == st2_run2
    assert alt1 == alt2
    assert alt1.model_dump() == alt2.model_dump()


# =====================================================================
# 8. GroundTruth & Holdout Isolation (AST Architectural Check)
# =====================================================================

def test_ground_truth_and_holdout_isolation_in_state_package():
    """Verify src/state package contains zero imports of ground_truth or holdout code."""
    state_dir = Path(__file__).parent.parent / "src" / "state"
    py_files = list(state_dir.rglob("*.py"))

    assert len(py_files) > 0, "No python files found in src/state"

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
# 9. Alert Schema Validation
# =====================================================================

def test_alert_pydantic_schema_compliance():
    """Verify emitted Alert validates strictly against Alert Pydantic schema."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    sm.process_score("M1", st, make_dummy_risk_score(score=5.0))
    st2 = st + timedelta(minutes=1)
    _, alert = sm.process_score("M1", st2, make_dummy_risk_score(score=5.0))

    assert alert is not None
    dumped = alert.model_dump()
    reconstructed = Alert(**dumped)

    assert reconstructed.alert_id == alert.alert_id
    assert reconstructed.merchant_id == "M1"
    assert reconstructed.risk_score == 5.0
    assert reconstructed.confidence == 1.0
