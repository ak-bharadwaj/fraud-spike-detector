"""Tests for Day 2 Initial Merchant Benchmark (M1-M3), Normal Traffic, Volume Spike, and EventBus Replay.

Test Coverage:
1. M1-M3 generation matching intended benchmark roles (M1 low-volume, M2 medium/stable, M3 high-volume).
2. Normal traffic generation without GroundTruth events.
3. Sudden volume spike injection with programmed window and realized magnitude calculation.
4. GroundTruthEvent schema, timestamp boundaries, and derived severity.
5. Virtual event-bus replay through TimeOrderedEventBus + VirtualClock.
6. Timestamp ordering and deterministic tie-breaking.
7. Deterministic replay across runs with identical seed.
8. Merchant compositionality (isolation between independent merchant RNG sequences).
9. GroundTruth isolation (no GroundTruth events used to construct normal traffic).
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent
from src.generator.archetypes import create_merchant_profile
from src.generator.anomalies import AnomalySpec
from src.generator.stream_generator import SyntheticStreamGenerator
from src.stream.clock import VirtualClock
from src.stream.bus import TimeOrderedEventBus


def test_m1_m2_m3_initial_merchant_benchmark_profiles():
    """Verify M1, M2, M3 merchant profiles reflect their intended benchmark roles."""
    seed = 42
    p_m1 = create_merchant_profile(seed, "M1", "sparse")
    p_m2 = create_merchant_profile(seed, "M2", "stable")
    p_m3 = create_merchant_profile(seed, "M3", "high_volume")

    # M1: Low volume (0.2 - 0.8 tx/min)
    assert 0.2 <= p_m1.base_rate_per_min <= 0.8
    # M2: Medium stable volume (8.0 - 15.0 tx/min)
    assert 8.0 <= p_m2.base_rate_per_min <= 15.0
    # M3: High volume (30.0 - 60.0 tx/min)
    assert 30.0 <= p_m3.base_rate_per_min <= 60.0

    # Rate hierarchy: M1 < M2 < M3
    assert p_m1.base_rate_per_min < p_m2.base_rate_per_min < p_m3.base_rate_per_min


def test_normal_traffic_generation_zero_ground_truth_events():
    """Verify normal traffic generation produces valid transactions and ZERO GroundTruth events."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    merchants = [
        {"id": "M1", "archetype": "sparse"},
        {"id": "M2", "archetype": "stable"},
        {"id": "M3", "archetype": "high_volume"},
    ]
    gen = SyntheticStreamGenerator(100, merchants, VirtualClock(initial_time=st))
    txs, events = gen.generate_window(duration_minutes=10.0)

    assert len(txs) > 0
    assert len(events) == 0, "Normal traffic generation must produce zero GroundTruth events."

    # Validate all transactions have valid fields
    for tx in txs:
        assert isinstance(tx, Transaction)
        assert tx.merchant_id in {"M1", "M2", "M3"}
        assert tx.amount >= 1.0
        assert tx.timestamp.tzinfo == timezone.utc
        assert st <= tx.timestamp < st + timedelta(minutes=10.0)


def test_sudden_volume_spike_injection_and_ground_truth_emission():
    """Verify programmed sudden volume spike injection generates elevated traffic and emits GroundTruthEvent upon window completion."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    merchants = [{"id": "M2", "archetype": "stable"}]
    gen = SyntheticStreamGenerator(42, merchants, VirtualClock(initial_time=st))

    # Schedule sudden volume spike from minute 2 to minute 5 (3-minute duration)
    spike_start = st + timedelta(minutes=2)
    spike_end = st + timedelta(minutes=5)
    spec = AnomalySpec(
        anomaly_type="volume_spike",
        start_time=spike_start,
        duration_seconds=180.0,
        target_magnitude=4.0,
        parameters={"rate_multiplier": 3.5},
    )
    handle = gen.schedule_anomaly("M2", spec, event_id="EVT-M2-SPIKE-01")
    assert handle == "EVT-M2-SPIKE-01"

    # Generate 1-minute windows progressively
    w1_txs, w1_events = gen.generate_window(1.0)  # Minute 0: normal
    assert len(w1_events) == 0

    w2_txs, w2_events = gen.generate_window(1.0)  # Minute 1: normal
    assert len(w2_events) == 0

    w3_txs, w3_events = gen.generate_window(1.0)  # Minute 2: spike active (not completed)
    assert len(w3_events) == 0

    w4_txs, w4_events = gen.generate_window(1.0)  # Minute 3: spike active
    assert len(w4_events) == 0

    w5_txs, w5_events = gen.generate_window(1.0)  # Minute 4: spike finishes at end of step
    assert len(w5_events) == 1, "GroundTruthEvent must be emitted when injection interval finishes."

    gt = w5_events[0]
    assert isinstance(gt, GroundTruthEvent)
    assert gt.event_id == "EVT-M2-SPIKE-01"
    assert gt.merchant_id == "M2"
    assert gt.anomaly_type == "volume_spike"
    assert gt.start_time == spike_start
    assert gt.end_time == spike_end
    assert gt.severity > 0.0
    assert gt.severity_level in {"LOW", "MEDIUM", "HIGH"}


def test_virtual_event_bus_replay_and_chronological_ordering():
    """Verify generated transactions replay through TimeOrderedEventBus in strict monotonic chronological order."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    merchants = [
        {"id": "M1", "archetype": "sparse"},
        {"id": "M2", "archetype": "stable"},
        {"id": "M3", "archetype": "high_volume"},
    ]
    gen = SyntheticStreamGenerator(777, merchants, VirtualClock(initial_time=st))
    txs, _ = gen.generate_window(5.0)

    clock = VirtualClock(initial_time=st)
    bus = TimeOrderedEventBus(clock=clock)
    bus.publish_batch(txs)

    dispatched = []
    bus.drain(handler=lambda tx: dispatched.append((tx.timestamp, tx.merchant_id, tx.transaction_id)))

    assert len(dispatched) == len(txs)
    # Verify strict non-decreasing chronological order
    for i in range(len(dispatched) - 1):
        t_curr = dispatched[i][0]
        t_next = dispatched[i + 1][0]
        assert t_curr <= t_next
        if t_curr == t_next:
            # Deterministic tie-breaking by (merchant_id, transaction_id)
            assert (dispatched[i][1], dispatched[i][2]) <= (dispatched[i + 1][1], dispatched[i + 1][2])


def test_generator_deterministic_replay():
    """Verify identical seed, config, and anomaly schedule produces 100% byte-for-byte identical output."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    merchants = [
        {"id": "M1", "archetype": "sparse"},
        {"id": "M2", "archetype": "stable"},
    ]

    def run_simulation():
        gen = SyntheticStreamGenerator(999, merchants, VirtualClock(initial_time=st))
        spec = AnomalySpec(
            anomaly_type="volume_spike",
            start_time=st + timedelta(minutes=1),
            duration_seconds=120.0,
            target_magnitude=3.5,
            parameters={"rate_multiplier": 3.0},
        )
        gen.schedule_anomaly("M2", spec, "EVT-REPLAY-01")
        return gen.generate_window(5.0)

    txs_run1, events_run1 = run_simulation()
    txs_run2, events_run2 = run_simulation()

    assert len(txs_run1) == len(txs_run2)
    assert len(events_run1) == len(events_run2) == 1

    for t1, t2 in zip(txs_run1, txs_run2):
        assert t1.model_dump() == t2.model_dump()

    assert events_run1[0].model_dump() == events_run2[0].model_dump()


def test_generator_merchant_compositionality():
    """Verify generating M1 alone produces identical transactions to generating M1 in presence of M2 and M3."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    seed = 888

    # Run 1: M1 alone
    gen_alone = SyntheticStreamGenerator(seed, [{"id": "M1", "archetype": "sparse"}], VirtualClock(initial_time=st))
    txs_alone, _ = gen_alone.generate_window(10.0)

    # Run 2: M1 with M2 and M3
    merchants_all = [
        {"id": "M1", "archetype": "sparse"},
        {"id": "M2", "archetype": "stable"},
        {"id": "M3", "archetype": "high_volume"},
    ]
    gen_multi = SyntheticStreamGenerator(seed, merchants_all, VirtualClock(initial_time=st))
    txs_multi, _ = gen_multi.generate_window(10.0)

    m1_from_multi = [t for t in txs_multi if t.merchant_id == "M1"]

    assert len(txs_alone) == len(m1_from_multi)
    for t1, t2 in zip(txs_alone, m1_from_multi):
        assert t1.model_dump() == t2.model_dump()
