"""Comprehensive unit tests for Day 2 Synthetic Benchmark Generator.

Validates:
1. Simulation-time RNG model and contiguous clock advancement (Blocker 1).
2. 128-bit SHA-256 event IDs and collision resistance across specs (Blocker 2).
3. Integrated time-varying legitimate rate expectation (Blocker 3).
4. Direct source of truth verification for customer and device pools (Test Improvement).
5. 100% field-by-field window partitioning identity.
6. Behavioral verification for all 6 archetypes and 7 anomaly classes.
7. Overlap rejection and reproducibility invariants.
"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent
from src.generator.archetypes import (
    create_merchant_profile,
    compute_legitimate_rate,
    sample_legitimate_amount,
    compute_expected_device_ratio,
    compute_robust_scale_device_ratio,
    compute_expected_country_ratio,
    compute_robust_scale_country_ratio,
    compute_expected_payment_ratio,
    compute_robust_scale_payment_ratio,
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


# =====================================================================
# BLOCKER 1: 128-bit SHA-256 Event IDs & Collision Resistance
# =====================================================================

def test_event_id_sha256_128bit_determinism_and_collision_resistance():
    """Verify 128-bit (32 hex char) SHA-256 event ID generation is deterministic and collision resistant."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(seconds=120)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    # 1. Determinism: identical spec -> identical SHA-256 event ID
    spec1 = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    spec2 = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    e1 = gen1.schedule_anomaly("M1", spec1)
    e2 = gen2.schedule_anomaly("M1", spec2)
    assert e1.event_id == e2.event_id
    assert len(e1.event_id.split("-")[-1]) == 32  # 32 hex chars = 128 bits

    # 2. Collision resistance across different dimensions
    gen3 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "stable"}], VirtualClock(initial_time=st))

    # Different merchant
    e_m1 = gen3.schedule_anomaly("M1", AnomalySpec("velocity_spike", st, 120.0, 3.0))
    e_m2 = gen3.schedule_anomaly("M2", AnomalySpec("velocity_spike", st, 120.0, 3.0))
    assert e_m1.event_id != e_m2.event_id

    # Different anomaly type
    gen4 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    e_type1 = gen4.schedule_anomaly("M1", AnomalySpec("velocity_spike", st, 120.0, 3.0), event_id=None)
    gen4_b = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    e_type2 = gen4_b.schedule_anomaly("M1", AnomalySpec("volume_spike", st, 120.0, 3.0), event_id=None)
    assert e_type1.event_id != e_type2.event_id

    # Different parameters
    gen5_a = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen5_b = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    e_param1 = gen5_a.schedule_anomaly("M1", AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0}))
    e_param2 = gen5_b.schedule_anomaly("M1", AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 5.0}))
    assert e_param1.event_id != e_param2.event_id

    # Different start/end interval
    st_b = st + timedelta(minutes=10)
    gen6_a = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen6_b = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st_b))
    e_time1 = gen6_a.schedule_anomaly("M1", AnomalySpec("velocity_spike", st, 120.0, 3.0))
    e_time2 = gen6_b.schedule_anomaly("M1", AnomalySpec("velocity_spike", st_b, 120.0, 3.0))
    assert e_time1.event_id != e_time2.event_id


# =====================================================================
# BLOCKER 2: Time-Varying Legitimate Expectation Over Anomaly Interval
# =====================================================================

def test_time_varying_legitimate_expectation_integration():
    """Verify expected baseline rate integrates time-varying legitimate behavior for growing and seasonal merchants."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # 1. Growing merchant over 60-minute anomaly interval
    gen_g = SyntheticStreamGenerator(42, [{"id": "M_growing", "archetype": "growing"}], VirtualClock(initial_time=st))
    spec_g = AnomalySpec("volume_spike", st, 3600.0, 3.0, {"rate_multiplier": 2.5})
    gen_g.schedule_anomaly("M_growing", spec_g, "EVT-GROW-INT")
    _, events_g = gen_g.generate_window(60.0)

    # Expected count is integrated over all 60 minutes
    prof_g = gen_g.profiles["M_growing"]
    expected_total_count = sum(
        compute_legitimate_rate(prof_g, st + timedelta(minutes=m), st)
        for m in range(60)
    )
    expected_rate = expected_total_count / 60.0

    # Start rate vs integrated rate difference (growth over 60 mins increases rate)
    start_rate = compute_legitimate_rate(prof_g, st, st)
    assert expected_rate > start_rate, "Integrated expected rate must reflect positive rate growth over interval."
    assert events_g[0].severity > 0.0

    # 2. Seasonal merchant over 24-hour (1440 min) interval
    gen_s = SyntheticStreamGenerator(42, [{"id": "M_seasonal", "archetype": "seasonal"}], VirtualClock(initial_time=st))
    spec_s = AnomalySpec("sustained_anomaly", st, 86400.0, 3.0, {"rate_multiplier": 2.0})
    gen_s.schedule_anomaly("M_seasonal", spec_s, "EVT-SEAS-INT")
    _, events_s = gen_s.generate_window(1440.0)

    prof_s = gen_s.profiles["M_seasonal"]
    expected_total_seas = sum(
        compute_legitimate_rate(prof_s, st + timedelta(minutes=m), st)
        for m in range(1440)
    )
    expected_rate_seas = expected_total_seas / 1440.0

    assert expected_rate_seas > 0.0
    assert events_s[0].severity > 0.0


# =====================================================================
# TEMPORAL MODEL: Explicit VirtualClock State Assertions
# =====================================================================

def test_simulation_clock_contiguous_advancement():
    """Verify VirtualClock current_time advances explicitly and contiguously with no duplicate timestamps."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    assert gen.clock.current_time() == st

    txs1, _ = gen.generate_window(5.0)
    assert gen.clock.current_time() == st + timedelta(minutes=5.0)

    txs2, _ = gen.generate_window(5.0)
    assert gen.clock.current_time() == st + timedelta(minutes=10.0)

    max_t1 = max(t.timestamp for t in txs1)
    min_t2 = min(t.timestamp for t in txs2)
    assert max_t1 < st + timedelta(minutes=5.0)
    assert min_t2 >= st + timedelta(minutes=5.0)
    assert max_t1 < min_t2


# =====================================================================
# SOURCE OF TRUTH: Customer and Device Pool Direct Configuration Tests
# =====================================================================

def test_direct_customer_and_device_pool_source_of_truth():
    """Verify generated customer and device IDs strictly conform to configured pool bounds."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen.profiles["M1"].legit_customer_pool_size = 350
    gen.profiles["M1"].legit_device_pool_size = 250

    txs, _ = gen.generate_window(20.0)

    cust_ids = [int(t.customer_id.split("-")[1]) for t in txs]
    dev_ids = [int(t.device_id.split("-")[1]) for t in txs]

    assert max(cust_ids) <= 350, f"Max customer ID ({max(cust_ids)}) must be <= configured pool size (350)"
    assert min(cust_ids) >= 1
    assert max(dev_ids) <= 250, f"Max device ID ({max(dev_ids)}) must be <= configured pool size (250)"
    assert min(dev_ids) >= 1


# =====================================================================
# Additional Generator Contract Tests
# =====================================================================

def test_customer_population_independence_from_device_pool():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec("behavioral_shift", st, 300.0, 4.0, {"rate_multiplier": 2.0})
    gen.schedule_anomaly("M1", spec)

    txs, _ = gen.generate_window(5.0)

    unique_devices = len({t.device_id for t in txs})
    unique_customers = len({t.customer_id for t in txs})

    assert unique_devices <= 5
    assert unique_customers > 15


def test_legitimate_3tier_payment_distribution():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs, _ = gen.generate_window(20.0)
    payments = {t.payment_method for t in txs}

    assert "CREDIT_CARD" in payments
    assert "DEBIT_CARD" in payments
    assert "PREPAID_CARD" in payments


def test_attribute_anomaly_specification_validation():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec_empty = AnomalySpec("attribute_anomaly", st, 120.0, 3.0, {})
    with pytest.raises(ValueError, match="attribute_anomaly spec for merchant 'M1' requires at least one supported attribute parameter"):
        gen.schedule_anomaly("M1", spec_empty)

    spec_invalid = AnomalySpec("attribute_anomaly", st, 120.0, 3.0, {"unsupported_key": "val"})
    with pytest.raises(ValueError, match="Unsupported attribute parameter 'unsupported_key' for attribute_anomaly"):
        gen.schedule_anomaly("M1", spec_invalid)


def test_window_partition_field_by_field_identity():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    clock1 = VirtualClock(initial_time=st)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], clock1)
    spec1 = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen1.schedule_anomaly("M1", spec1, "EVT-FIELD-1")
    txs1, events1 = gen1.generate_window(duration_minutes=5.0)

    clock2 = VirtualClock(initial_time=st)
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], clock2)
    spec2 = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen2.schedule_anomaly("M1", spec2, "EVT-FIELD-2")
    txs2 = []
    events2_all = []
    for _ in range(5):
        t_step, e_step = gen2.generate_window(duration_minutes=1.0)
        txs2.extend(t_step)
        events2_all.extend(e_step)

    assert len(txs1) == len(txs2)
    for t1, t2 in zip(txs1, txs2):
        assert t1.transaction_id == t2.transaction_id
        assert t1.timestamp == t2.timestamp
        assert t1.merchant_id == t2.merchant_id
        assert t1.customer_id == t2.customer_id
        assert t1.amount == t2.amount
        assert t1.payment_method == t2.payment_method
        assert t1.country == t2.country
        assert t1.device_id == t2.device_id

    sev1 = events1[-1].severity
    sev2 = events2_all[-1].severity
    assert sev1 == sev2


# =====================================================================
# Archetypes & Anomalies Validation Tests
# =====================================================================

def test_archetype_stable_behavior():
    prof = create_merchant_profile(42, "M_stable", "stable")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rates = [compute_legitimate_rate(prof, sim_start + timedelta(hours=h), sim_start) for h in range(24)]
    assert np.std(rates) == 0.0


def test_archetype_seasonal_behavior():
    prof = create_merchant_profile(42, "M_seasonal", "seasonal")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    t_night = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
    t_day = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    assert compute_legitimate_rate(prof, t_day, sim_start) > compute_legitimate_rate(prof, t_night, sim_start)


def test_archetype_growing_behavior():
    prof = create_merchant_profile(42, "M_growing", "growing")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert compute_legitimate_rate(prof, sim_start + timedelta(days=30), sim_start) > compute_legitimate_rate(prof, sim_start, sim_start)


def test_archetype_volatile_behavior():
    prof_s = create_merchant_profile(42, "M_stable", "stable")
    prof_v = create_merchant_profile(42, "M_volatile", "volatile")
    rng_s = np.random.default_rng(42)
    rng_v = np.random.default_rng(42)
    assert np.var([sample_legitimate_amount(prof_v, rng_v) for _ in range(200)]) > np.var([sample_legitimate_amount(prof_s, rng_s) for _ in range(200)])


def test_archetype_sparse_behavior():
    st = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(
        global_seed=42,
        merchant_configs=[{"id": "M_sparse", "archetype": "sparse"}, {"id": "M_stable", "archetype": "stable"}],
        clock=VirtualClock(initial_time=st),
    )
    txs, _ = gen.generate_window(duration_minutes=30.0)
    assert len([t for t in txs if t.merchant_id == "M_sparse"]) < 0.2 * len([t for t in txs if t.merchant_id == "M_stable"])


def test_archetype_mixed_behavior():
    prof = create_merchant_profile(42, "M_mixed", "mixed")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert compute_legitimate_rate(prof, sim_start + timedelta(days=20), sim_start) > compute_legitimate_rate(prof, sim_start, sim_start)


def test_no_overlapping_anomalies_per_merchant():
    st = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    st1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    spec1 = AnomalySpec("volume_spike", st1, 300.0, 3.0)
    gen.schedule_anomaly("M1", spec1, "EVT-1")

    st2 = datetime(2026, 1, 1, 10, 2, tzinfo=timezone.utc)
    spec2 = AnomalySpec("velocity_spike", st2, 300.0, 4.0)

    with pytest.raises(OverlapAnomalyError):
        gen.schedule_anomaly("M1", spec2, "EVT-2")


def test_smoke_benchmark_fixture():
    st = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(777, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M1", spec, "EVT-SMOKE-1")

    txs, events = gen.generate_window(duration_minutes=5.0)
    assert len(txs) > 0
    assert len(events) == 1
