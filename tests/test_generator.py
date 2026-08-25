"""Tests for Day 2 Synthetic Benchmark Generator.

Tests:
1. Deterministic generation
2. Merchant compositional reproducibility
3. All six required archetypes (stable, seasonal, growing, volatile, sparse, mixed)
4. All seven required anomaly classes (velocity, volume, amount, behavioral, attribute, sustained, compound)
5. Ground-truth creation and severity derivation
6. Temporal validity (start_time < end_time)
7. No Overlapping Active Events per merchant invariant
8. Legitimate surge generation vs anomaly separation
9. Seasonality and sparse merchant behavior
10. Benchmark smoke test fixture
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent
from src.generator.archetypes import (
    create_merchant_profile,
    compute_legitimate_rate,
    sample_legitimate_amount,
)
from src.generator.anomalies import (
    AnomalySpec,
    create_ground_truth_event,
    compute_standardized_magnitude,
    compute_compound_severity,
)
from src.generator.stream_generator import (
    SyntheticStreamGenerator,
    OverlapAnomalyError,
)
from src.stream.clock import VirtualClock


def test_deterministic_generation():
    """Verify same global seed + merchant config produces identical streams."""
    seed = 42
    merchant_configs = [
        {"id": "M1", "archetype": "stable"},
        {"id": "M2", "archetype": "seasonal"},
    ]

    clock1 = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=merchant_configs, clock=clock1)
    txs1, _ = gen1.generate_window(duration_minutes=5.0)

    clock2 = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen2 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=merchant_configs, clock=clock2)
    txs2, _ = gen2.generate_window(duration_minutes=5.0)

    assert len(txs1) == len(txs2)
    for t1, t2 in zip(txs1, txs2):
        assert t1.transaction_id == t2.transaction_id
        assert t1.amount == t2.amount
        assert t1.merchant_id == t2.merchant_id


def test_merchant_compositional_reproducibility():
    """Verify adding merchant B does not alter merchant A's stream."""
    seed = 100
    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Stream 1: M1 alone
    clock1 = VirtualClock(initial_time=start_time)
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock1)
    txs_m1_alone, _ = gen1.generate_window(duration_minutes=10.0)

    # Stream 2: M1 and M2 together
    clock2 = VirtualClock(initial_time=start_time)
    gen2 = SyntheticStreamGenerator(
        global_seed=seed,
        merchant_configs=[{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "volatile"}],
        clock=clock2,
    )
    txs_combined, _ = gen2.generate_window(duration_minutes=10.0)
    txs_m1_together = [t for t in txs_combined if t.merchant_id == "M1"]

    assert len(txs_m1_alone) == len(txs_m1_together)
    for t1, t2 in zip(txs_m1_alone, txs_m1_together):
        assert t1.transaction_id == t2.transaction_id
        assert t1.amount == t2.amount


def test_all_six_archetypes():
    """Verify creation and generation for all 6 merchant archetypes."""
    seed = 42
    archetypes = ["stable", "seasonal", "growing", "volatile", "sparse", "mixed"]
    configs = [{"id": f"M_{a}", "archetype": a} for a in archetypes]

    clock = VirtualClock()
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=configs, clock=clock)

    assert len(gen.profiles) == 6
    txs, _ = gen.generate_window(duration_minutes=10.0)
    merchant_ids_present = {t.merchant_id for t in txs}
    assert "M_stable" in merchant_ids_present
    assert "M_sparse" in merchant_ids_present


def test_sparse_merchant_behavior():
    """Verify sparse merchant produces lower transaction counts compared to stable merchant."""
    seed = 42
    configs = [
        {"id": "M_sparse", "archetype": "sparse"},
        {"id": "M_stable", "archetype": "stable"},
    ]
    clock = VirtualClock()
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=configs, clock=clock)

    txs, _ = gen.generate_window(duration_minutes=30.0)
    sparse_count = len([t for t in txs if t.merchant_id == "M_sparse"])
    stable_count = len([t for t in txs if t.merchant_id == "M_stable"])

    assert sparse_count < stable_count, f"Sparse count ({sparse_count}) should be < stable count ({stable_count})"


def test_seasonality_effects():
    """Verify diurnal seasonality alters transaction rate by time of day."""
    prof = create_merchant_profile(42, "M_seasonal", "seasonal")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Compare rate at 03:00 vs 15:00 UTC
    t_night = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
    t_day = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)

    rate_night = compute_legitimate_rate(prof, t_night, sim_start)
    rate_day = compute_legitimate_rate(prof, t_day, sim_start)

    assert rate_day > rate_night, f"Day rate ({rate_day}) should be > night rate ({rate_night})"


def test_legitimate_surges_vs_anomalies():
    """Verify promotional surge increases rate without generating GroundTruthEvents."""
    seed = 42
    configs = [{"id": "M1", "archetype": "stable"}]
    clock = VirtualClock()
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=configs, clock=clock)

    # Window without surge
    txs_normal, events_normal = gen.generate_window(duration_minutes=5.0, is_surge_active={"M1": False})
    # Window with promotional surge
    txs_surge, events_surge = gen.generate_window(duration_minutes=5.0, is_surge_active={"M1": True})

    assert len(txs_surge) > len(txs_normal)
    assert len(events_normal) == 0
    assert len(events_surge) == 0, "Legitimate promotional surge must NOT emit GroundTruthEvent!"


def test_all_seven_anomaly_types():
    """Verify scheduling and generating all 7 required anomaly classes."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    anomaly_classes = [
        "velocity_spike",
        "volume_spike",
        "amount_spike",
        "behavioral_shift",
        "attribute_anomaly",
        "sustained_anomaly",
        "compound_anomaly",
    ]

    base_time = clock.current_time()
    for i, atype in enumerate(anomaly_classes):
        st = base_time + timedelta(minutes=i * 10)
        spec = AnomalySpec(
            anomaly_type=atype,
            start_time=st,
            duration_seconds=300.0,
            target_magnitude=3.5,
            parameters={"rate_multiplier": 3.0, "country": "HIGH_RISK_GEO"},
        )
        gt = gen.schedule_anomaly("M1", spec, event_id=f"EVT-{atype}")
        assert gt.anomaly_type == atype
        assert gt.start_time < gt.end_time


def test_no_overlapping_anomalies_per_merchant():
    """Verify No Overlapping Active Events per merchant invariant."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    spec1 = AnomalySpec(anomaly_type="volume_spike", start_time=st1, duration_seconds=300.0, target_magnitude=3.0, parameters={})
    gen.schedule_anomaly("M1", spec1, event_id="EVT-1")

    # Schedule overlapping event (starts at 10:02, while EVT-1 ends at 10:05)
    st2 = datetime(2026, 1, 1, 10, 2, tzinfo=timezone.utc)
    spec2 = AnomalySpec(anomaly_type="velocity_spike", start_time=st2, duration_seconds=300.0, target_magnitude=4.0, parameters={})

    with pytest.raises(OverlapAnomalyError):
        gen.schedule_anomaly("M1", spec2, event_id="EVT-2")


def test_ground_truth_severity_derivation():
    """Verify severity calculation and derived severity_level in GroundTruthEvent."""
    st = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    spec_low = AnomalySpec(anomaly_type="amount_spike", start_time=st, duration_seconds=60.0, target_magnitude=1.5, parameters={})
    gt_low = create_ground_truth_event("E-LOW", "M1", spec_low)
    assert gt_low.severity == 1.5
    assert gt_low.severity_level == "LOW"

    spec_high = AnomalySpec(anomaly_type="compound_anomaly", start_time=st, duration_seconds=60.0, target_magnitude=4.5, parameters={})
    gt_high = create_ground_truth_event("E-HIGH", "M1", spec_high)
    assert gt_high.severity == 4.5
    assert gt_high.severity_level == "HIGH"


def test_smoke_benchmark_fixture():
    """Benchmark smoke test fixture verifying multi-merchant stream and ground-truth generation."""
    seed = 777
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    merchant_configs = [
        {"id": "M1", "archetype": "stable"},
        {"id": "M2", "archetype": "seasonal"},
        {"id": "M3", "archetype": "sparse"},
    ]
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=merchant_configs, clock=clock)

    # Schedule non-overlapping anomalies
    st = clock.current_time()
    spec1 = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M1", spec1, "EVT-SMOKE-1")

    spec2 = AnomalySpec("amount_spike", st + timedelta(minutes=5), 120.0, 4.2, {"amount_multiplier": 5.0})
    gen.schedule_anomaly("M2", spec2, "EVT-SMOKE-2")

    txs, events = gen.generate_window(duration_minutes=10.0)

    assert len(txs) > 0
    assert len(events) == 2
    assert isinstance(txs[0], Transaction)
    assert isinstance(events[0], GroundTruthEvent)
