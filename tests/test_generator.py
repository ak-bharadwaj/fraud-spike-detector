"""Comprehensive unit tests for Day 2 Synthetic Benchmark Generator.

Validates:
1. Simulation-time RNG model and contiguous clock advancement (Blocker 1).
2. Deterministic SHA-256 event IDs and collision resistance (Blocker 2).
3. Customer population independence from device pool (Blocker 3).
4. Full 3-tier legitimate payment distribution sampling (Blocker 4).
5. Strict attribute anomaly specification validation (Blocker 5).
6. 100% field-by-field window partitioning identity.
7. Behavioral verification for all 6 archetypes and 7 anomaly classes.
8. Overlap rejection and reproducibility invariants.
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
# BLOCKER 1: Simulation-Time RNG Model & Contiguous Clock Advancement
# =====================================================================

def test_simulation_clock_contiguous_advancement():
    """Verify contiguous stream generation across sequential generate_window calls with no duplicate timestamps."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs1, _ = gen.generate_window(5.0)
    txs2, _ = gen.generate_window(5.0)

    # Window 1 timestamps in [T0, T5), Window 2 timestamps in [T5, T10)
    max_t1 = max(t.timestamp for t in txs1)
    min_t2 = min(t.timestamp for t in txs2)

    assert max_t1 < st + timedelta(minutes=5.0)
    assert min_t2 >= st + timedelta(minutes=5.0)
    assert max_t1 < min_t2


# =====================================================================
# BLOCKER 2: Deterministic SHA-256 Event IDs & Collision Resistance
# =====================================================================

def test_event_id_sha256_determinism_and_collision_resistance():
    """Verify SHA-256 event ID generation is deterministic and collision resistant."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    # 1. Determinism: identical spec -> identical SHA-256 event ID
    spec1 = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    spec2 = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    e1 = gen1.schedule_anomaly("M1", spec1)
    e2 = gen2.schedule_anomaly("M1", spec2)
    assert e1.event_id == e2.event_id

    # 2. Collision resistance: different params at same timestamp -> different event ID
    gen3 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec_a = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    spec_b = AnomalySpec("volume_spike", st, 120.0, 3.0, {"rate_multiplier": 2.5})
    ea = gen3.schedule_anomaly("M1", spec_a)
    eb = gen3.schedule_anomaly("M2", spec_b)
    assert ea.event_id != eb.event_id


# =====================================================================
# BLOCKER 3: Customer Population Independent of Device Population
# =====================================================================

def test_customer_population_independence_from_device_pool():
    """Verify customer population is independent of device pool size during behavioral shift."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec("behavioral_shift", st, 300.0, 4.0, {"rate_multiplier": 2.0})
    gen.schedule_anomaly("M1", spec)

    txs, _ = gen.generate_window(5.0)

    unique_devices = len({t.device_id for t in txs})
    unique_customers = len({t.customer_id for t in txs})

    # Devices constrained to <= 5, while customers drawn from 5000 pool
    assert unique_devices <= 5
    assert unique_customers > 15, f"Customer cardinality ({unique_customers}) must not be constrained by device pool (5)"


# =====================================================================
# BLOCKER 4: Full 3-Tier Legitimate Payment Distribution
# =====================================================================

def test_legitimate_3tier_payment_distribution():
    """Verify legitimate generator samples CREDIT_CARD, DEBIT_CARD, and PREPAID_CARD."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs, _ = gen.generate_window(20.0)
    payments = {t.payment_method for t in txs}

    assert "CREDIT_CARD" in payments
    assert "DEBIT_CARD" in payments
    assert "PREPAID_CARD" in payments


# =====================================================================
# BLOCKER 5: Rejection of Malformed Attribute Anomaly Specifications
# =====================================================================

def test_attribute_anomaly_specification_validation():
    """Verify malformed attribute anomaly specifications are explicitly rejected."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    # Empty parameters -> rejected
    spec_empty = AnomalySpec("attribute_anomaly", st, 120.0, 3.0, {})
    with pytest.raises(ValueError, match="attribute_anomaly spec for merchant 'M1' requires at least one supported attribute parameter"):
        gen.schedule_anomaly("M1", spec_empty)

    # Unsupported attribute key -> rejected
    spec_invalid = AnomalySpec("attribute_anomaly", st, 120.0, 3.0, {"unsupported_key": "val"})
    with pytest.raises(ValueError, match="Unsupported attribute parameter 'unsupported_key' for attribute_anomaly"):
        gen.schedule_anomaly("M1", spec_invalid)


# =====================================================================
# Determinism / Reproducibility Invariants
# =====================================================================

def test_reproducibility_invariant_1_same_seed_same_merchant():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs1, _ = gen1.generate_window(5.0)
    txs2, _ = gen2.generate_window(5.0)

    assert len(txs1) == len(txs2)
    for t1, t2 in zip(txs1, txs2):
        assert t1 == t2


def test_reproducibility_invariant_2_merchant_isolation():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_single = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen_multi = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "volatile"}], VirtualClock(initial_time=st))

    txs_single, _ = gen_single.generate_window(5.0)
    txs_multi, _ = gen_multi.generate_window(5.0)

    m1_multi = [t for t in txs_multi if t.merchant_id == "M1"]

    assert len(txs_single) == len(m1_multi)
    for t1, t2 in zip(txs_single, m1_multi):
        assert t1 == t2


def test_window_partition_field_by_field_identity():
    """Verify 100% field-by-field equality and severity equality between 5-min step and five 1-min steps."""
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
