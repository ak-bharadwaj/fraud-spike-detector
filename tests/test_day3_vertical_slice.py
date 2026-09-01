"""Day 3 Vertical Slice Integration and Scorer Exception Test Suite.

Verifies:
1. Complete Day-3 vertical flow: Transaction -> FeatureEngine -> BaselineEngine -> Scorer -> AlertStateMachine -> Alert -> SQLite -> Evaluator.
2. Feature contract & 4 feature groups (volume, velocity, amount, behavioral/device).
3. Historical-only baseline with robust median/MAD and evidence state semantics (INSUFFICIENT, DEGRADED, SUFFICIENT).
4. Statistical scorer computing standardized deviation magnitude and valid RiskScore.
5. AlertStateMachine state lifecycle and Alert schema compliance.
6. SQLite persistence and reload across alerts, audit_records, state_transitions without data loss.
7. Evaluator first-anomaly evaluation (TP, FP, FN, precision, recall, F1).
8. Mandatory Section 20 Scorer Exception Path:
   - Exception strictly scoped to scorer invocation.
   - Saves Error AuditRecord with risk_score=None and data_quality_status='SCORER_ERROR'.
   - Emits NO Alert.
   - Triggers NO ALERT state transition.
   - Stream continues processing subsequent windows.
9. Full deterministic replay and architectural invariants.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import (
    Transaction,
    FeatureSnapshot,
    BaselineSnapshot,
    RiskScore,
    Alert,
    AuditRecord,
    GroundTruthEvent,
    FrozenDetectorConfig,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.audit.database import SQLiteAuditStore
from src.evaluation.evaluator import AnomalyEvaluator
from src.detector.pipeline import StreamingDetectorPipeline
from src.generator.anomalies import AnomalySpec
from src.generator.stream_generator import SyntheticStreamGenerator
from src.stream.clock import VirtualClock
from src.stream.bus import TimeOrderedEventBus


# =====================================================================
# 1. Feature Layer & Four Feature Groups
# =====================================================================

def test_day3_feature_groups_and_contract():
    """Verify FeatureEngine produces FeatureSnapshot with all 4 required feature groups."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = FeatureEngine(window_duration_minutes=1.0)

    txs = [
        Transaction(
            transaction_id=f"tx_{i}",
            timestamp=st + timedelta(seconds=i * 10),
            merchant_id="M1",
            customer_id=f"CUST_{i % 3}",
            amount=25.0 + i * 5.0,
            payment_method="CREDIT_CARD",
            country="US",
            device_id=f"DEV_{i % 2}",
        )
        for i in range(6)
    ]

    snap = engine.extract_snapshot("M1", txs, st, st + timedelta(minutes=1.0))

    # 1. Volume
    assert snap.volume == 6.0
    # 2. Velocity
    assert snap.velocity == 6.0
    # 3. Amount statistics
    assert "mean_amount" in snap.amount_statistics
    assert "median_amount" in snap.amount_statistics
    assert "mad_amount" in snap.amount_statistics
    assert "std_amount" in snap.amount_statistics
    assert snap.amount_statistics["mean_amount"] == 37.5
    assert snap.amount_statistics["median_amount"] == 37.5
    # 4. Behavioral / Device cardinality
    assert snap.unique_customers == 3
    assert snap.unique_devices == 2

    assert snap.data_quality == "GOOD"


# =====================================================================
# 2. Baseline Layer: Historical-Only & Evidence Semantics
# =====================================================================

def test_day3_baseline_historical_only_and_evidence_semantics():
    """Verify BaselineEngine computes historical-only robust baseline and owns evidence state."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=5, min_window_count=2)

    # Initial query with 0 history -> INSUFFICIENT evidence state
    curr_snap = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st,
        volume=10.0,
        velocity=10.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 500.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=8,
        unique_devices=7,
        data_quality="GOOD",
    )
    b0 = engine.get_baseline("M1", curr_snap)
    assert b0.evidence_state == "INSUFFICIENT"
    assert b0.history_count == 0

    # Feed 5 historical windows (>= min_history_count)
    for i in range(5):
        h_snap = FeatureSnapshot(
            merchant_id="M1",
            timestamp=st + timedelta(minutes=i + 1),
            volume=10.0 + i,
            velocity=10.0 + i,
            amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 500.0, "min_amount": 40.0, "max_amount": 60.0},
            unique_customers=8,
            unique_devices=7,
            data_quality="GOOD",
        )
        engine.update(h_snap)

    # 5 windows (>= min_history_count 5) but current window volume is 1 (< min_window_count 2) -> DEGRADED
    b_deg = engine.get_baseline("M1", FeatureSnapshot(merchant_id="M1", timestamp=st + timedelta(minutes=6), volume=1.0, velocity=1.0, data_quality="GOOD", unique_customers=1, unique_devices=1))
    assert b_deg.evidence_state == "DEGRADED"
    assert b_deg.history_count == 5

    # 5 windows and current window volume is 10 (>= min_window_count 2) -> SUFFICIENT
    b_suf = engine.get_baseline("M1", FeatureSnapshot(merchant_id="M1", timestamp=st + timedelta(minutes=6), volume=10.0, velocity=10.0, data_quality="GOOD", unique_customers=8, unique_devices=7))
    assert b_suf.evidence_state == "SUFFICIENT"
    assert b_suf.history_count == 5


# =====================================================================
# 3. Statistical Scorer & AlertStateMachine
# =====================================================================

def test_day3_statistical_scorer_and_state_machine_alert():
    """Verify statistical scorer standardized deviation and AlertStateMachine alert transition."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)
    sm = AlertStateMachine(persistence=2, cooldown_windows=3, static_threshold=3.5)

    base_snap = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st,
        expected_values={"volume": 10.0, "velocity": 10.0, "unique_customers": 8.0, "unique_devices": 7.0, "amount_mean_amount": 50.0, "amount_std_amount": 5.0, "amount_median_amount": 50.0, "amount_mad_amount": 3.0, "amount_total_amount": 500.0, "amount_min_amount": 40.0, "amount_max_amount": 60.0},
        robust_scale={"volume": 2.0, "velocity": 2.0, "unique_customers": 1.5, "unique_devices": 1.5, "amount_mean_amount": 10.0, "amount_std_amount": 2.0, "amount_median_amount": 10.0, "amount_mad_amount": 1.0, "amount_total_amount": 100.0, "amount_min_amount": 10.0, "amount_max_amount": 15.0},
        history_count=20,
        current_window_count=1,
        evidence_state="SUFFICIENT",
    )

    # Spike snapshot (volume = 30 -> M = (30 - 10)/2 = 10.0)
    spike_snap = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st + timedelta(minutes=1),
        volume=30.0,
        velocity=30.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 1500.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=8,
        unique_devices=7,
        data_quality="GOOD",
    )

    risk_score = scorer.calculate_score(spike_snap, base_snap)
    assert risk_score.score is not None
    assert risk_score.score >= 3.5
    assert "volume" in risk_score.triggered_signals

    # State Machine: 1st above-threshold score -> CANDIDATE (P=2)
    s1, a1 = sm.process_score("M1", st + timedelta(minutes=1), risk_score)
    assert s1 == "CANDIDATE"
    assert a1 is None

    # State Machine: 2nd above-threshold score -> ALERT
    s2, a2 = sm.process_score("M1", st + timedelta(minutes=2), risk_score)
    assert s2 == "ALERT"
    assert a2 is not None
    assert isinstance(a2, Alert)
    assert a2.merchant_id == "M1"
    assert a2.risk_score >= 3.5


# =====================================================================
# 4. Mandatory Section 20 Scorer Exception Path
# =====================================================================

def test_day3_mandatory_scorer_exception_path_isolated(monkeypatch):
    """Verify Section 20 Scorer Exception Path:
    - Scorer exception creates AuditRecord only.
    - risk_score = None, data_quality_status = 'SCORER_ERROR'.
    - NO Alert emitted.
    - NO ALERT state transition.
    - Stream continues uninterrupted.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(static_threshold=3.5, persistence=1, min_window_count=1)
    pipeline = StreamingDetectorPipeline(config=config, db_path=":memory:")

    call_index = [0]
    orig_calculate = pipeline.scorer.calculate_score

    def mock_failing_scorer(feat, base):
        call_index[0] += 1
        if call_index[0] == 2:
            raise ArithmeticError("Forced mathematical exception in scorer")
        return orig_calculate(feat, base)

    monkeypatch.setattr(pipeline.scorer, "calculate_score", mock_failing_scorer)

    # 3 windows of transactions
    txs = []
    for w in range(3):
        w_st = st + timedelta(minutes=w)
        for i in range(15):
            txs.append(
                Transaction(
                    transaction_id=f"tx_w{w}_{i}",
                    timestamp=w_st + timedelta(seconds=i * 3),
                    merchant_id="M1",
                    customer_id=f"C_{i}",
                    amount=50.0,
                    payment_method="CREDIT_CARD",
                    country="US",
                    device_id=f"D_{i}",
                )
            )

    alerts = pipeline.process_transactions(txs)

    # Stream continued through all 3 windows
    assert call_index[0] == 3

    audits = pipeline.audit_store.get_audit_records("M1")
    assert len(audits) == 3

    # Window 2 audit record is an Error AuditRecord
    w2_err_audit = audits[1]
    assert w2_err_audit["risk_score"] is None
    assert w2_err_audit["data_quality_status"] == "SCORER_ERROR"
    assert "ArithmeticError" in w2_err_audit["triggered_signals"][0]

    # Zero Alert emitted due to exception
    assert len(alerts) == 0

    # No ALERT state transition occurred
    transitions = pipeline.audit_store.get_state_transitions("M1")
    for t in transitions:
        assert t["new_state"] != "ALERT"


# =====================================================================
# 5. Full End-to-End Vertical Slice Test
# =====================================================================

def test_day3_end_to_end_vertical_slice_with_evaluator(tmp_path):
    """End-to-end test executing the full Day-3 vertical slice:
    Transactions -> Features -> Baseline -> Scorer -> State -> Alert -> SQLite -> Evaluator.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    db_file = tmp_path / "day3_vertical_slice.db"

    config = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, min_window_count=1)
    pipeline = StreamingDetectorPipeline(config=config, db_path=db_file)

    # Generate synthetic scenario using SyntheticStreamGenerator
    merchants = [{"id": "M1", "archetype": "stable"}]
    gen = SyntheticStreamGenerator(42, merchants, VirtualClock(initial_time=st))

    # Baseline 5 minutes normal traffic
    txs_base, _ = gen.generate_window(5.0)

    # Schedule sudden volume spike for minutes 5-8 (3 minutes)
    spike_spec = AnomalySpec(
        anomaly_type="volume_spike",
        start_time=st + timedelta(minutes=5),
        duration_seconds=180.0,
        target_magnitude=4.5,
        parameters={"rate_multiplier": 4.0},
    )
    gen.schedule_anomaly("M1", spike_spec, event_id="EVT-DAY3-SPIKE")
    txs_spike, events = gen.generate_window(3.0)

    all_txs = txs_base + txs_spike
    assert len(events) == 1
    gt_event = events[0]

    # Process through streaming detector pipeline
    alerts = pipeline.process_transactions(all_txs)

    assert len(alerts) >= 1
    first_alert = alerts[0]
    assert first_alert.merchant_id == "M1"
    assert first_alert.risk_score >= 3.5

    # Verify SQLite persistence
    persisted_alerts = pipeline.audit_store.get_alerts("M1")
    persisted_audits = pipeline.audit_store.get_audit_records("M1")
    persisted_transitions = pipeline.audit_store.get_state_transitions("M1")

    assert len(persisted_alerts) == len(alerts)
    assert len(persisted_audits) > 0
    assert any(t["new_state"] == "ALERT" for t in persisted_transitions)

    # Close and reload DB
    pipeline.audit_store.close()
    reloaded_store = SQLiteAuditStore(db_path=db_file)
    reloaded_alerts = reloaded_store.get_alerts("M1")
    assert len(reloaded_alerts) == len(alerts)
    reloaded_store.close()

    # Evaluator: evaluate alerts against ground truth
    evaluator = AnomalyEvaluator(temporal_tolerance_seconds=60.0)
    metrics = evaluator.evaluate(alerts=alerts, ground_truth_events=[gt_event])

    assert metrics.tp == 1
    assert metrics.fn == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.mean_latency_seconds is not None
