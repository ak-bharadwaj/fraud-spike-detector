"""Day 3 Vertical Slice Integration and StatisticalDeviationScorer Test Suite.

Verifies:
1. StatisticalDeviationScorer:
   - Consumes FeatureSnapshot + BaselineSnapshot.
   - Calculates exact standardized deviation magnitude M_k = |f_k - exp_k| / scale_k.
   - Computes statistical score S = max_k M_k (pure statistical, NO EWMA smoothing).
   - Preserves evidence state semantics (INSUFFICIENT -> score=None, DEGRADED -> conf=0.5, SUFFICIENT -> conf=1.0).
   - Produces valid RiskScore.
   - Pure determinism and zero ground-truth / holdout dependencies.
2. Complete Day-3 vertical flow:
   Transaction -> FeatureEngine -> BaselineEngine -> StatisticalDeviationScorer -> AlertStateMachine -> Alert -> SQLite -> Evaluator.
3. Mandatory Section 20 Scorer Exception Path:
   - Scorer exception creates Error AuditRecord with risk_score=None, data_quality_status='SCORER_ERROR'.
   - Emits NO Alert.
   - Triggers NO ALERT state transition.
   - Stream continues processing subsequent windows.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
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
from src.scoring.statistical import StatisticalDeviationScorer
from src.state.alert_state_machine import AlertStateMachine
from src.audit.database import SQLiteAuditStore
from src.evaluation.evaluator import AnomalyEvaluator
from src.detector.pipeline import StreamingDetectorPipeline
from src.generator.anomalies import AnomalySpec
from src.generator.stream_generator import SyntheticStreamGenerator
from src.stream.clock import VirtualClock


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
# 3. StatisticalDeviationScorer: Mathematical Correctness & Schema
# =====================================================================

def test_day3_statistical_deviation_scorer_mathematical_correctness():
    """Verify StatisticalDeviationScorer computes exact standardized deviation M_k = |f_k - exp_k| / scale_k and S = max M_k."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = StatisticalDeviationScorer(static_threshold=3.5)

    base_snap = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st,
        expected_values={
            "volume": 10.0,
            "velocity": 10.0,
            "unique_customers": 8.0,
            "unique_devices": 7.0,
            "amount_total_amount": 500.0,
            "amount_mean_amount": 50.0,
            "amount_std_amount": 5.0,
            "amount_median_amount": 50.0,
            "amount_mad_amount": 3.0,
            "amount_min_amount": 40.0,
            "amount_max_amount": 60.0,
        },
        robust_scale={
            "volume": 2.0,
            "velocity": 2.0,
            "unique_customers": 1.0,
            "unique_devices": 1.0,
            "amount_total_amount": 100.0,
            "amount_mean_amount": 10.0,
            "amount_std_amount": 2.0,
            "amount_median_amount": 10.0,
            "amount_mad_amount": 1.0,
            "amount_min_amount": 10.0,
            "amount_max_amount": 15.0,
        },
        history_count=20,
        current_window_count=10,
        evidence_state="SUFFICIENT",
    )

    # Feature with volume = 22.0 -> M_vol = |22 - 10| / 2 = 6.0
    feat_snap = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st + timedelta(minutes=1),
        volume=22.0,
        velocity=22.0,
        amount_statistics={
            "total_amount": 1100.0,  # M_total = |1100 - 500| / 100 = 6.0
            "mean_amount": 50.0,     # M_mean = 0.0
            "std_amount": 5.0,       # M_std = 0.0
            "median_amount": 50.0,   # M_med = 0.0
            "mad_amount": 3.0,       # M_mad = 0.0
            "min_amount": 40.0,      # M_min = 0.0
            "max_amount": 60.0,      # M_max = 0.0
        },
        unique_customers=8,          # M_cust = 0.0
        unique_devices=7,            # M_dev = 0.0
        data_quality="GOOD",
    )

    risk_score = scorer.calculate_score(feat_snap, base_snap)

    assert isinstance(risk_score, RiskScore)
    assert risk_score.score == 6.0
    assert risk_score.confidence == 1.0
    assert "volume" in risk_score.triggered_signals
    assert "velocity" in risk_score.triggered_signals
    assert "total_amount" in risk_score.triggered_signals
    assert risk_score.data_quality == "GOOD"

    # Test DEGRADED evidence state
    base_deg = base_snap.model_copy(update={"evidence_state": "DEGRADED"})
    risk_deg = scorer.calculate_score(feat_snap, base_deg)
    assert risk_deg.score == 6.0
    assert risk_deg.confidence == 0.5
    assert risk_deg.data_quality == "DEGRADED"

    # Test INSUFFICIENT evidence state
    base_ins = base_snap.model_copy(update={"evidence_state": "INSUFFICIENT"})
    risk_ins = scorer.calculate_score(feat_snap, base_ins)
    assert risk_ins.score is None
    assert risk_ins.confidence == 0.0
    assert risk_ins.triggered_signals == []


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
    scorer = StatisticalDeviationScorer(static_threshold=config.static_threshold)
    pipeline = StreamingDetectorPipeline(config=config, scorer=scorer, db_path=":memory:")

    call_index = [0]
    orig_calculate = scorer.calculate_score

    def mock_failing_scorer(feat, base):
        call_index[0] += 1
        if call_index[0] == 2:
            raise ArithmeticError("Forced mathematical exception in statistical scorer")
        return orig_calculate(feat, base)

    monkeypatch.setattr(scorer, "calculate_score", mock_failing_scorer)

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
# 5. Full End-to-End Vertical Slice Test with Evaluator
# =====================================================================

def test_day3_end_to_end_vertical_slice_with_evaluator(tmp_path):
    """End-to-end test executing the full Day-3 vertical slice:
    Transactions -> Features -> Baseline -> StatisticalDeviationScorer -> State -> Alert -> SQLite -> Evaluator.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    db_file = tmp_path / "day3_vertical_slice.db"

    config = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, min_window_count=1)
    scorer = StatisticalDeviationScorer(static_threshold=config.static_threshold)
    pipeline = StreamingDetectorPipeline(config=config, scorer=scorer, db_path=db_file)

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

    # Process through streaming detector pipeline with StatisticalDeviationScorer
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
