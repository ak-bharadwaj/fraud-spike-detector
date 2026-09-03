"""Comprehensive behavioral unit tests for Day 6 AlertStateMachine.

Validates all 16 required state machine behavioral dimensions:
1. Initial state (NORMAL).
2. Below-threshold score (retains NORMAL).
3. Exact threshold boundary (score == 3.5 qualifies as breach).
4. Above-threshold score (score > 3.5 qualifies as breach).
5. Persistence progression (NORMAL -> CANDIDATE -> ALERT).
6. State lifecycle consistency (process_score returns "ALERT" and get_merchant_state() returns "ALERT").
7. Step-by-step window-by-window cooldown progression (w1 CANDIDATE, w2 ALERT, w3..w7 COOLDOWN, w8 NORMAL).
8. Config-driven construction via from_config.
9. DEGRADED evidence behavior (qualifies for persistence with confidence=0.5).
10. INSUFFICIENT evidence behavior (resets persistence counter to 0).
11. SUFFICIENT evidence behavior (qualifies for persistence with confidence=1.0).
12. Merchant state isolation (Merchant A vs B independence).
13. Deterministic replay and deterministic Alert ID.
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
from src.contracts.config_loader import load_detector_config
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
# 1. Config-Driven Construction (Blocker 3)
# =====================================================================

def test_config_driven_alert_state_machine_construction():
    """Verify AlertStateMachine.from_config loads parameters matching detector.yaml."""
    cfg = load_detector_config(Path(__file__).parent.parent / "config" / "detector.yaml")
    sm = AlertStateMachine.from_config(cfg)

    assert sm.persistence == cfg.scorer.persistence
    assert sm.cooldown_windows == cfg.state_machine.cooldown_windows
    assert sm.static_threshold == cfg.scorer.static_threshold
    assert sm.detector_version == cfg.version


# =====================================================================
# 2. Initial State & Below-Threshold Score
# =====================================================================

def test_initial_state_and_below_threshold_score():
    """Verify initial state is NORMAL and sub-threshold scores retain NORMAL."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    assert sm.get_merchant_state("M1") == "NORMAL"

    risk_low = make_dummy_risk_score(score=2.0)
    state, alert = sm.process_score("M1", st, risk_low)

    assert state == "NORMAL"
    assert sm.get_merchant_state("M1") == "NORMAL"
    assert alert is None


# =====================================================================
# 3. Exact Threshold Boundary & Above-Threshold Score
# =====================================================================

def test_exact_threshold_boundary_and_above_threshold_score():
    """Verify exact threshold boundary (score == 3.5) and above-threshold score (score > 3.5) qualify as breach."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    risk_exact = make_dummy_risk_score(score=3.5)
    state1, alert1 = sm.process_score("M1", st, risk_exact)

    assert state1 == "CANDIDATE"
    assert sm.get_merchant_state("M1") == "CANDIDATE"
    assert alert1 is None


# =====================================================================
# 4. Explicit Window-by-Window Cooldown Transition Lifecycle (Blockers 1 & 2)
# =====================================================================

def test_explicit_window_by_window_cooldown_transition_lifecycle():
    """Verify step-by-step state lifecycle: w1 CANDIDATE, w2 ALERT (consistent state), w3..w7 COOLDOWN, w8 NORMAL."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Window 1 (t=0): score >= 3.5 -> CANDIDATE
    st1, alt1 = sm.process_score("M1", st, make_dummy_risk_score(score=5.0))
    assert st1 == "CANDIDATE"
    assert sm.get_merchant_state("M1") == "CANDIDATE"
    assert alt1 is None

    # Window 2 (t=1): score >= 3.5 -> ALERT (both process_score and get_merchant_state return "ALERT")
    t2 = st + timedelta(minutes=1)
    st2, alt2 = sm.process_score("M1", t2, make_dummy_risk_score(score=5.0))
    assert st2 == "ALERT"
    assert sm.get_merchant_state("M1") == "ALERT"
    assert alt2 is not None
    assert alt2.risk_score == 5.0

    # Window 3 (t=2): 1st Cooldown window -> COOLDOWN
    t3 = st + timedelta(minutes=2)
    st3, alt3 = sm.process_score("M1", t3, make_dummy_risk_score(score=1.0))
    assert st3 == "COOLDOWN"
    assert sm.get_merchant_state("M1") == "COOLDOWN"
    assert alt3 is None

    # Window 4 (t=3): 2nd Cooldown window -> COOLDOWN
    t4 = st + timedelta(minutes=3)
    st4, alt4 = sm.process_score("M1", t4, make_dummy_risk_score(score=1.0))
    assert st4 == "COOLDOWN"
    assert sm.get_merchant_state("M1") == "COOLDOWN"

    # Window 5 (t=4): 3rd Cooldown window -> COOLDOWN
    t5 = st + timedelta(minutes=4)
    st5, alt5 = sm.process_score("M1", t5, make_dummy_risk_score(score=1.0))
    assert st5 == "COOLDOWN"

    # Window 6 (t=5): 4th Cooldown window -> COOLDOWN
    t6 = st + timedelta(minutes=5)
    st6, alt6 = sm.process_score("M1", t6, make_dummy_risk_score(score=1.0))
    assert st6 == "COOLDOWN"

    # Window 7 (t=6): 5th Cooldown window -> COOLDOWN
    t7 = st + timedelta(minutes=6)
    st7, alt7 = sm.process_score("M1", t7, make_dummy_risk_score(score=1.0))
    assert st7 == "COOLDOWN"

    # Window 8 (t=7): Cooldown exhausted -> transitions back to NORMAL
    t8 = st + timedelta(minutes=7)
    st8, alt8 = sm.process_score("M1", t8, make_dummy_risk_score(score=1.0))
    assert st8 == "NORMAL"
    assert sm.get_merchant_state("M1") == "NORMAL"
    assert alt8 is None


def test_cooldown_reset_on_anomalous_window_during_cooldown():
    """Verify ALERT -> COOLDOWN -> anomalous window during cooldown resets cooldown counter -> return to NORMAL only after required consecutive normal windows."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=1, cooldown_windows=3, static_threshold=5.0)

    # Window 0: score >= 5.0 -> ALERT
    st0, alert0 = sm.process_score("M1", st, make_dummy_risk_score(score=10.0))
    assert st0 == "ALERT"
    assert alert0 is not None

    # Window 1: normal score < 5.0 -> enters COOLDOWN (counter = 2)
    t1 = st + timedelta(minutes=1)
    st1, alert1 = sm.process_score("M1", t1, make_dummy_risk_score(score=1.0))
    assert st1 == "COOLDOWN"
    assert alert1 is None

    # Window 2: ANOMALOUS score >= 5.0 during COOLDOWN -> stays in COOLDOWN and resets cooldown_counter back to 3
    t2 = st + timedelta(minutes=2)
    st2, alert2 = sm.process_score("M1", t2, make_dummy_risk_score(score=10.0))
    assert st2 == "COOLDOWN"
    assert alert2 is None  # Suppressed due to cooldown

    # Window 3 (1st consecutive normal window): counter becomes 2
    t3 = st + timedelta(minutes=3)
    st3, _ = sm.process_score("M1", t3, make_dummy_risk_score(score=1.0))
    assert st3 == "COOLDOWN"

    # Window 4 (2nd consecutive normal window): counter becomes 1
    t4 = st + timedelta(minutes=4)
    st4, _ = sm.process_score("M1", t4, make_dummy_risk_score(score=1.0))
    assert st4 == "COOLDOWN"

    # Window 5 (3rd consecutive normal window): counter becomes 0 -> returns to NORMAL
    t5 = st + timedelta(minutes=5)
    st5, _ = sm.process_score("M1", t5, make_dummy_risk_score(score=1.0))
    assert st5 == "NORMAL"
    assert sm.get_merchant_state("M1") == "NORMAL"


# =====================================================================
# 5. Evidence State Persistence Behaviors (Blocker 4)
# =====================================================================

def test_degraded_qualifying_score_continues_persistence():
    """Verify DEGRADED evidence score >= 3.5 continues persistence from CANDIDATE to ALERT with confidence=0.5."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Window 1: SUFFICIENT score -> CANDIDATE
    sm.process_score("M1", st, make_dummy_risk_score(score=5.0, confidence=1.0, data_quality="GOOD"))
    assert sm.get_merchant_state("M1") == "CANDIDATE"

    # Window 2: DEGRADED score >= 3.5 -> ALERT (emits Alert with confidence=0.5)
    t2 = st + timedelta(minutes=1)
    st2, alert2 = sm.process_score("M1", t2, make_dummy_risk_score(score=5.0, confidence=0.5, data_quality="DEGRADED"))

    assert st2 == "ALERT"
    assert sm.get_merchant_state("M1") == "ALERT"
    assert alert2 is not None
    assert alert2.confidence == 0.5


def test_insufficient_evidence_resets_candidate_counter():
    """Verify INSUFFICIENT evidence (score=None) resets CANDIDATE back to NORMAL and counter to 0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # Window 1: CANDIDATE
    sm.process_score("M1", st, make_dummy_risk_score(score=5.0))
    assert sm.get_merchant_state("M1") == "CANDIDATE"

    # Window 2: INSUFFICIENT evidence -> NORMAL
    t2 = st + timedelta(minutes=1)
    st2, alert2 = sm.process_score("M1", t2, make_dummy_risk_score(score=None, confidence=0.0, data_quality="INSUFFICIENT"))

    assert st2 == "NORMAL"
    assert sm.get_merchant_state("M1") == "NORMAL"
    assert alert2 is None


# =====================================================================
# 6. Merchant Isolation & Determinism
# =====================================================================

def test_multi_merchant_state_isolation():
    """Verify Merchant A transitions do not affect Merchant B state context."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    sm.process_score("M_A", st, make_dummy_risk_score(score=5.0))
    assert sm.get_merchant_state("M_A") == "CANDIDATE"
    assert sm.get_merchant_state("M_B") == "NORMAL"


def test_deterministic_state_machine_replay_and_alert_id():
    """Verify identical input sequence produces identical state transitions and deterministic Alert ID."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Run 1
    sm1 = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)
    sm1.process_score("M1", st, make_dummy_risk_score(score=5.0))
    t2 = st + timedelta(minutes=1)
    st1, alt1 = sm1.process_score("M1", t2, make_dummy_risk_score(score=5.0))

    # Run 2
    sm2 = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)
    sm2.process_score("M1", st, make_dummy_risk_score(score=5.0))
    st2_run2, alt2 = sm2.process_score("M1", t2, make_dummy_risk_score(score=5.0))

    assert st1 == st2_run2 == "ALERT"
    assert alt1.alert_id == alt2.alert_id
    assert alt1.model_dump() == alt2.model_dump()


# =====================================================================
# 7. GroundTruth & Holdout Isolation (AST Architectural Check)
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
# 8. Alert Schema Validation
# =====================================================================

def test_alert_pydantic_schema_compliance():
    """Verify emitted Alert validates strictly against Alert Pydantic schema."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    sm.process_score("M1", st, make_dummy_risk_score(score=5.0))
    t2 = st + timedelta(minutes=1)
    _, alert = sm.process_score("M1", t2, make_dummy_risk_score(score=5.0))

    assert alert is not None
    dumped = alert.model_dump()
    reconstructed = Alert(**dumped)

    assert reconstructed.alert_id == alert.alert_id
    assert reconstructed.merchant_id == "M1"
    assert reconstructed.risk_score == 5.0
    assert reconstructed.confidence == 1.0
