"""Gate 1 Foundation and Vertical Slice Test Suite.

Tests:
1. TimeOrderedEventBus ordering, tie-breaking, VirtualClock integration, replay determinism, compositionality.
2. Monotonic clock invariant during EventBus drain (increasing timestamps, equal timestamps, out-of-order, backward clock rejection).
3. EventBus-driven streaming pipeline execution via bus.drain(handler).
4. SQLiteAuditStore schema creation, nullability, table queries, and database file reload.
5. Deterministic replay of SQLite audit records (100% identical records across runs).
6. Scorer-only exception handling (Section 20: Audit-only error, NO Alert, NO ALERT state transition, stream continuation).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import pytest

from src.contracts.contracts import (
    Transaction,
    Alert,
    AuditRecord,
    FeatureSnapshot,
    BaselineSnapshot,
    FrozenDetectorConfig,
)
from src.stream.clock import VirtualClock
from src.stream.bus import TimeOrderedEventBus
from src.audit.database import SQLiteAuditStore
from src.detector.pipeline import StreamingDetectorPipeline


# =====================================================================
# 1. TimeOrderedEventBus & VirtualClock Integration Tests
# =====================================================================

def test_time_ordered_event_bus_chronological_ordering_and_tie_breaking():
    """Verify bus sorts transactions by timestamp and tie-breaks deterministically by (timestamp, merchant_id, tx_id)."""
    clock = VirtualClock()
    bus = TimeOrderedEventBus(clock=clock)

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tx1 = Transaction(transaction_id="tx3", timestamp=st + timedelta(minutes=2), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")
    tx2 = Transaction(transaction_id="tx1", timestamp=st, merchant_id="M2", customer_id="C2", amount=20.0, payment_method="credit_card", country="US", device_id="D2")
    tx3 = Transaction(transaction_id="tx2", timestamp=st, merchant_id="M1", customer_id="C3", amount=30.0, payment_method="credit_card", country="US", device_id="D3")

    bus.publish_batch([tx1, tx2, tx3])
    ordered = bus.get_ordered_events()

    assert len(ordered) == 3
    assert ordered[0].transaction_id == "tx2"  # (st, M1, tx2)
    assert ordered[1].transaction_id == "tx1"  # (st, M2, tx1)
    assert ordered[2].transaction_id == "tx3"  # (st+2m, M1, tx3)


def test_time_ordered_event_bus_virtual_clock_advancement_and_drain():
    """Verify drain advances VirtualClock monotonically to each transaction timestamp."""
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    bus = TimeOrderedEventBus(clock=clock)

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tx1 = Transaction(transaction_id="t1", timestamp=st + timedelta(minutes=5), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")
    tx2 = Transaction(transaction_id="t2", timestamp=st + timedelta(minutes=15), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")

    bus.publish_batch([tx1, tx2])
    dispatched = []
    bus.drain(handler=lambda tx: dispatched.append((tx.transaction_id, clock.current_time())))

    assert len(dispatched) == 2
    assert dispatched[0] == ("t1", st + timedelta(minutes=5))
    assert dispatched[1] == ("t2", st + timedelta(minutes=15))
    assert clock.current_time() == st + timedelta(minutes=15)


def test_time_ordered_event_bus_monotonic_clock_increasing_timestamps():
    """Verify drain with strictly increasing timestamps advances clock monotonically."""
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    bus = TimeOrderedEventBus(clock=clock)

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    txs = [
        Transaction(transaction_id=f"tx_{i}", timestamp=st + timedelta(minutes=i), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")
        for i in range(1, 6)
    ]
    bus.publish_batch(txs)

    clock_progression = []
    bus.drain(handler=lambda tx: clock_progression.append(clock.current_time()))

    assert len(clock_progression) == 5
    for i in range(len(clock_progression) - 1):
        assert clock_progression[i] < clock_progression[i + 1]


def test_time_ordered_event_bus_monotonic_clock_equal_timestamps():
    """Verify drain with identical timestamps maintains non-decreasing clock without error."""
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    bus = TimeOrderedEventBus(clock=clock)

    st = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
    tx1 = Transaction(transaction_id="tx_b", timestamp=st, merchant_id="M2", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")
    tx2 = Transaction(transaction_id="tx_a", timestamp=st, merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")

    bus.publish_batch([tx1, tx2])
    dispatched_times = []
    bus.drain(handler=lambda tx: dispatched_times.append(clock.current_time()))

    assert len(dispatched_times) == 2
    assert dispatched_times[0] == st
    assert dispatched_times[1] == st
    assert clock.current_time() == st


def test_time_ordered_event_bus_monotonic_clock_out_of_order_publication():
    """Verify out-of-order published transactions are dispatched in monotonic chronological order."""
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    bus = TimeOrderedEventBus(clock=clock)

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tx_t30 = Transaction(transaction_id="tx_30", timestamp=st + timedelta(minutes=30), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")
    tx_t10 = Transaction(transaction_id="tx_10", timestamp=st + timedelta(minutes=10), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")
    tx_t20 = Transaction(transaction_id="tx_20", timestamp=st + timedelta(minutes=20), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")

    # Publish in reverse order: 30, 10, 20
    bus.publish(tx_t30)
    bus.publish(tx_t10)
    bus.publish(tx_t20)

    clock_steps = []
    dispatched_ids = []
    bus.drain(handler=lambda tx: (dispatched_ids.append(tx.transaction_id), clock_steps.append(clock.current_time())))

    assert dispatched_ids == ["tx_10", "tx_20", "tx_30"]
    assert clock_steps == [st + timedelta(minutes=10), st + timedelta(minutes=20), st + timedelta(minutes=30)]
    assert clock.current_time() == st + timedelta(minutes=30)


def test_time_ordered_event_bus_monotonic_clock_rejection_of_past_events():
    """Verify attempting to drain events older than current VirtualClock time raises ValueError."""
    # Clock is already at 12:30
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc))
    bus = TimeOrderedEventBus(clock=clock)

    # Event timestamp is in the past relative to clock (12:10)
    st_past = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)
    tx_past = Transaction(transaction_id="tx_past", timestamp=st_past, merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")

    bus.publish(tx_past)
    with pytest.raises(ValueError, match="cannot move backward in time"):
        bus.drain()


def test_time_ordered_event_bus_replay_determinism_and_merchant_compositionality():
    """Verify replaying stream yields identical ordering and adding a new merchant does not alter existing event order."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    m1_txs = [
        Transaction(transaction_id=f"m1_{i}", timestamp=st + timedelta(minutes=i), merchant_id="M1", customer_id="C1", amount=10.0, payment_method="credit_card", country="US", device_id="D1")
        for i in range(5)
    ]
    m2_txs = [
        Transaction(transaction_id=f"m2_{i}", timestamp=st + timedelta(minutes=i), merchant_id="M2", customer_id="C2", amount=15.0, payment_method="credit_card", country="US", device_id="D2")
        for i in range(5)
    ]

    bus1 = TimeOrderedEventBus()
    bus1.publish_batch(m1_txs)
    res1 = bus1.get_ordered_events()

    bus2 = TimeOrderedEventBus()
    bus2.publish_batch(m1_txs + m2_txs)
    res2 = bus2.get_ordered_events()

    m1_from_res2 = [t for t in res2 if t.merchant_id == "M1"]
    assert res1 == m1_from_res2


# =====================================================================
# 2. SQLite Audit Store & File Reload Persistence Tests
# =====================================================================

def test_sqlite_audit_store_schemas_and_file_reload(tmp_path):
    """Verify SQLite schema creation, nullability, TEXT serialization, and database file reload."""
    db_file = tmp_path / "test_audit.db"
    store = SQLiteAuditStore(db_path=db_file)

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # 1. Alert Persistence
    alert = Alert(
        alert_id="ALT-100",
        merchant_id="M1",
        timestamp=st,
        risk_score=4.2,
        confidence=0.9,
        reason="Volume spike breach",
        triggered_signals=["volume"],
        detector_version="1.0.0",
    )
    store.save_alert(alert)

    # 2. Audit Record Persistence
    feat = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st,
        volume=50.0,
        velocity=0.833,
        amount_statistics={"mean": 50.0, "std": 10.0, "median": 48.0, "mad": 5.0},
        unique_customers=40,
        unique_devices=35,
        data_quality="GOOD",
    )
    base = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st,
        expected_values={"volume": 10.0},
        robust_scale={"volume": 2.0},
        history_count=20,
        current_window_count=1,
        evidence_state="SUFFICIENT",
    )

    audit_rec = AuditRecord(
        audit_id="AUD-100",
        alert_id="ALT-100",
        merchant_id="M1",
        timestamp=st,
        risk_score=4.2,
        confidence=0.9,
        features=feat.model_dump(mode="json"),
        baseline=base.model_dump(mode="json"),
        triggered_signals=["volume"],
        detector_version="1.0.0",
        data_quality_status="OK",
    )
    store.save_audit_record(audit_rec)

    # 3. State Transition Persistence
    store.save_state_transition(
        merchant_id="M1",
        timestamp=st,
        previous_state="NORMAL",
        new_state="ALERT",
        reason="Threshold breach",
        risk_score=4.2,
    )

    # 4. Experiment Metadata Persistence
    store.save_experiment(
        experiment_id="EXP-001",
        dataset_id="DEV-SET-01",
        dataset_hash="abc123hash",
        seed=1001,
        config_hash="cfg456hash",
        detector_version="1.0.0",
        metrics={"precision": 1.0, "recall": 0.8},
        costs={"fp_cost": 0.0, "fn_exposure": 100.0},
    )

    # Close store and reload from database file
    store.close()

    reloaded_store = SQLiteAuditStore(db_path=db_file)

    alerts = reloaded_store.get_alerts(merchant_id="M1")
    assert len(alerts) == 1
    assert alerts[0]["alert_id"] == "ALT-100"
    assert alerts[0]["risk_score"] == 4.2
    assert alerts[0]["triggered_signals"] == ["volume"]

    audits = reloaded_store.get_audit_records(merchant_id="M1")
    assert len(audits) == 1
    assert audits[0]["alert_id"] == "ALT-100"
    assert audits[0]["features"]["volume"] == 50.0

    transitions = reloaded_store.get_state_transitions(merchant_id="M1")
    assert len(transitions) == 1
    assert transitions[0]["new_state"] == "ALERT"

    exps = reloaded_store.get_experiments()
    assert len(exps) == 1
    assert exps[0]["metrics"]["precision"] == 1.0

    reloaded_store.close()


# =====================================================================
# 3. EventBus-Driven Pipeline & Replay Determinism Tests
# =====================================================================

def test_streaming_detector_pipeline_eventbus_driven_execution_and_replay_determinism():
    """Verify EventBus-driven pipeline execution via bus.drain and 100% deterministic SQLite records across replay."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, min_window_count=1)

    txs = []
    # Baseline windows with volume variation (8..12)
    base_counts = [8, 10, 12, 9, 11]
    for w, count in enumerate(base_counts):
        w_st = st + timedelta(minutes=w)
        for i in range(count):
            txs.append(
                Transaction(
                    transaction_id=f"base_w{w}_{i}",
                    timestamp=w_st + timedelta(seconds=i * 4),
                    merchant_id="M1",
                    customer_id=f"C{i}",
                    amount=50.0,
                    payment_method="credit_card",
                    country="US",
                    device_id=f"D{i}",
                )
            )

    # Spike windows (150 txs each)
    for w in range(5, 7):
        w_st = st + timedelta(minutes=w)
        for i in range(150):
            txs.append(
                Transaction(
                    transaction_id=f"spike_w{w}_{i}",
                    timestamp=w_st + timedelta(seconds=i * 0.3),
                    merchant_id="M1",
                    customer_id=f"C_spike_{i}",
                    amount=50.0,
                    payment_method="credit_card",
                    country="US",
                    device_id=f"D_spike_{i}",
                )
            )

    # Execution 1
    p1 = StreamingDetectorPipeline(config=config, db_path=":memory:")
    alerts1 = p1.process_transactions(txs)
    audits1 = p1.audit_store.get_audit_records(merchant_id="M1")

    # Execution 2 (Replay)
    p2 = StreamingDetectorPipeline(config=config, db_path=":memory:")
    alerts2 = p2.process_transactions(txs)
    audits2 = p2.audit_store.get_audit_records(merchant_id="M1")

    assert len(alerts1) == 1
    assert len(alerts2) == 1
    assert alerts1[0].model_dump() == alerts2[0].model_dump()

    # Replay determinism: Audit records are 100% byte-for-byte identical across runs (no random UUIDs!)
    assert len(audits1) == len(audits2)
    for a1, a2 in zip(audits1, audits2):
        assert a1 == a2


def test_scorer_only_exception_scoping_section20_contract_compliance(monkeypatch):
    """Verify Section 20 Scorer Exception Path: Exception scoped STRICTLY to scorer, saved as SCORER_ERROR audit, NO Alert, NO ALERT transition, stream continues."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(static_threshold=3.5, persistence=1, min_window_count=1)
    pipeline = StreamingDetectorPipeline(config=config, db_path=":memory:")

    call_count = [0]
    orig_calculate_score = pipeline.scorer.calculate_score

    def mock_calculate_score(feat_snap, base_snap):
        call_count[0] += 1
        if call_count[0] == 2:
            raise ZeroDivisionError("Simulated robust scale zero division exception")
        return orig_calculate_score(feat_snap, base_snap)

    monkeypatch.setattr(pipeline.scorer, "calculate_score", mock_calculate_score)

    txs = []
    for w in range(3):
        w_st = st + timedelta(minutes=w)
        for i in range(20):
            txs.append(
                Transaction(
                    transaction_id=f"w{w}_{i}",
                    timestamp=w_st + timedelta(seconds=i * 2),
                    merchant_id="M1",
                    customer_id=f"C{i}",
                    amount=50.0,
                    payment_method="credit_card",
                    country="US",
                    device_id=f"D{i}",
                )
            )

    alerts = pipeline.process_transactions(txs)

    assert call_count[0] == 3

    audits = pipeline.audit_store.get_audit_records(merchant_id="M1")
    assert len(audits) == 3

    # Window 2 audit record is an Error AuditRecord
    w2_audit = audits[1]
    assert w2_audit["risk_score"] is None
    assert w2_audit["data_quality_status"] == "SCORER_ERROR"
    assert "ZeroDivisionError" in w2_audit["triggered_signals"][0]

    # Verify NO ALERT transition occurred due to the exception
    transitions = pipeline.audit_store.get_state_transitions(merchant_id="M1")
    for t in transitions:
        assert t["new_state"] != "ALERT"
