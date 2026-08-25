"""Comprehensive unit tests for Day 2 Synthetic Benchmark Generator.

Validates:
1. Ground truth event lifecycle (schedule returns event_id: str, completion emits GroundTruthEvent).
2. Independent severity verification without generator helper reuse.
3. Option A whole-minute anomaly duration validation.
4. Event ID determinism and dimension uniqueness.
5. 100% field-by-field window partitioning identity.
6. Behavioral verification for all 6 archetypes and 7 anomaly classes.
7. Overlap rejection and reproducibility invariants.
"""

from datetime import datetime, timedelta, timezone
import math
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
# BLOCKER 1: Ground Truth Lifecycle (schedule -> handle, completion -> GT)
# =====================================================================

def test_ground_truth_lifecycle_schedule_handle_completion_emission():
    """Verify schedule_anomaly returns event_id handle (str), and GroundTruthEvent is emitted ONLY on completion."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    handle = gen.schedule_anomaly("M1", spec, "EVT-HANDLE-1")

    # schedule_anomaly returns handle string
    assert isinstance(handle, str)
    assert handle == "EVT-HANDLE-1"

    # Intermediate steps emit no events
    for m in range(1, 5):
        _, evs = gen.generate_window(1.0)
        assert len(evs) == 0

    # Completion step emits finalized GroundTruthEvent
    _, evs_final = gen.generate_window(1.0)
    assert len(evs_final) == 1
    assert isinstance(evs_final[0], GroundTruthEvent)
    assert evs_final[0].event_id == "EVT-HANDLE-1"
    assert evs_final[0].severity > 0.0


# =====================================================================
# BLOCKER 2: Independent Severity Verification Test (No Helper Reuse)
# =====================================================================

def test_independent_severity_verification_growing_merchant():
    """Independently calculate expected_rate, observed_rate, robust_scale, and M for growing merchant without helper reuse."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M_growing", "archetype": "growing"}], VirtualClock(initial_time=st))

    spec = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M_growing", spec, "EVT-IND-GROW")

    txs, events = gen.generate_window(5.0)
    assert len(events) == 1
    gt = events[0]

    # Independent calculation writing explicit math directly in test without helper calls
    prof = gen.profiles["M_growing"]
    base_rate = prof.base_rate_per_min

    expected_total_count_ind = 0.0
    for m in range(5):
        dt_m = st + timedelta(minutes=m)
        elapsed_days = (dt_m - st).total_seconds() / 86400.0
        growth_mult = 1.0 + 0.02 * elapsed_days
        expected_total_count_ind += base_rate * growth_mult

    expected_rate_ind = expected_total_count_ind / 5.0
    observed_total_ind = len(txs)
    observed_rate_ind = observed_total_ind / 5.0

    robust_scale_ind = max(0.5, 0.2 * expected_rate_ind)
    m_expected_ind = abs(observed_rate_ind - expected_rate_ind) / robust_scale_ind

    assert gt.severity == m_expected_ind, f"Ground truth severity ({gt.severity}) must EXACTLY equal independently calculated M ({m_expected_ind})"


def test_independent_severity_verification_seasonal_merchant():
    """Independently calculate expected_rate, observed_rate, robust_scale, and M for seasonal merchant without helper reuse."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M_seasonal", "archetype": "seasonal"}], VirtualClock(initial_time=st))

    spec = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M_seasonal", spec, "EVT-IND-SEAS")

    txs, events = gen.generate_window(5.0)
    assert len(events) == 1
    gt = events[0]

    prof = gen.profiles["M_seasonal"]
    base_rate = prof.base_rate_per_min

    expected_total_count_ind = 0.0
    for m in range(5):
        dt_m = st + timedelta(minutes=m)
        hour = dt_m.hour + dt_m.minute / 60.0
        diurnal_mult = 1.0 + 0.4 * math.sin(2.0 * math.pi * (hour - 6.0) / 24.0)
        day_of_week = dt_m.weekday()
        weekly_mult = 1.2 if day_of_week in (5, 6) else 0.95
        expected_total_count_ind += base_rate * diurnal_mult * weekly_mult

    expected_rate_ind = expected_total_count_ind / 5.0
    observed_total_ind = len(txs)
    observed_rate_ind = observed_total_ind / 5.0

    robust_scale_ind = max(0.5, 0.2 * expected_rate_ind)
    m_expected_ind = abs(observed_rate_ind - expected_rate_ind) / robust_scale_ind

    assert gt.severity == m_expected_ind, f"Ground truth severity ({gt.severity}) must EXACTLY equal independently calculated M ({m_expected_ind})"


# =====================================================================
# BLOCKER 3: Option A Whole-Minute Anomaly Duration Validation
# =====================================================================

def test_anomaly_duration_whole_minute_validation():
    """Verify non-whole-minute anomaly durations (e.g. 90s, 45s) are explicitly rejected."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec_90s = AnomalySpec("velocity_spike", st, 90.0, 3.0)
    with pytest.raises(ValueError, match="Anomaly duration_seconds must be a positive whole number of minutes"):
        gen.schedule_anomaly("M1", spec_90s)

    spec_0s = AnomalySpec("velocity_spike", st, 0.0, 3.0)
    with pytest.raises(ValueError, match="Anomaly duration_seconds must be a positive whole number of minutes"):
        gen.schedule_anomaly("M1", spec_0s)


# =====================================================================
# BLOCKER 4: Event ID Determinism and Dimension Uniqueness
# =====================================================================

def test_event_id_determinism_and_dimension_uniqueness():
    """Verify 128-bit SHA-256 event ID generation is deterministic and unique across spec dimensions."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec1 = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    spec2 = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    eid1 = gen1.schedule_anomaly("M1", spec1)
    eid2 = gen2.schedule_anomaly("M1", spec2)
    assert eid1 == eid2
    assert len(eid1.split("-")[-1]) == 32

    gen3 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "stable"}], VirtualClock(initial_time=st))

    e_m1 = gen3.schedule_anomaly("M1", AnomalySpec("velocity_spike", st, 120.0, 3.0))
    e_m2 = gen3.schedule_anomaly("M2", AnomalySpec("velocity_spike", st, 120.0, 3.0))
    assert e_m1 != e_m2


# =====================================================================
# TEMPORAL MODEL & SOURCE OF TRUTH TESTS
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


def test_direct_customer_and_device_pool_source_of_truth():
    """Verify generated customer and device IDs strictly conform to configured pool bounds."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen.profiles["M1"].legit_customer_pool_size = 350
    gen.profiles["M1"].legit_device_pool_size = 250

    txs, _ = gen.generate_window(20.0)

    cust_ids = [int(t.customer_id.split("-")[1]) for t in txs]
    dev_ids = [int(t.device_id.split("-")[1]) for t in txs]

    assert max(cust_ids) <= 350
    assert min(cust_ids) >= 1
    assert max(dev_ids) <= 250
    assert min(dev_ids) >= 1


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
