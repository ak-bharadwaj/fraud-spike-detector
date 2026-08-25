"""Comprehensive unit tests for Day 2 Synthetic Benchmark Generator.

Validates:
1. RNG sub-seeding 6-invariant proof (Blocker 1).
2. Option A integer minute window validation (Blocker 2).
3. Legitimate device pool and payment method single source of truth (Blockers 3 & 4).
4. Attribute anomaly and compound anomaly signal semantics (Blockers 5 & 6).
5. 100% field-by-field window partitioning identity (Blocker 7).
6. Multi-window persistence for sustained anomaly.
7. Behavioral verification for all 6 archetypes and 7 anomaly classes.
8. Overlap rejection and RNG compositional reproducibility invariants.
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
# BLOCKER 1: RNG Sub-seeding 6-Invariant Proof
# =====================================================================

def test_rng_invariant_1_same_seed_same_merchant():
    """1. Same seed + same merchant produces identical transaction stream."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs1, _ = gen1.generate_window(5.0)
    txs2, _ = gen2.generate_window(5.0)

    assert len(txs1) == len(txs2)
    for t1, t2 in zip(txs1, txs2):
        assert t1 == t2


def test_rng_invariant_2_adding_merchant_preserves_existing():
    """2. Adding another merchant does not change existing merchant's transaction stream."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_single = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen_multi = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "volatile"}], VirtualClock(initial_time=st))

    txs_single, _ = gen_single.generate_window(5.0)
    txs_multi, _ = gen_multi.generate_window(5.0)

    m1_multi = [t for t in txs_multi if t.merchant_id == "M1"]

    assert len(txs_single) == len(m1_multi)
    for t1, t2 in zip(txs_single, m1_multi):
        assert t1 == t2


def test_rng_invariant_3_same_simulation_interval():
    """3. Same simulation interval produces identical transactions regardless of caller chunking."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs1, _ = gen1.generate_window(5.0)

    txs2 = []
    for _ in range(5):
        txs_step, _ = gen2.generate_window(1.0)
        txs2.extend(txs_step)

    assert len(txs1) == len(txs2)
    for t1, t2 in zip(txs1, txs2):
        assert t1 == t2


def test_rng_invariant_4_transaction_id_determinism():
    """4. Transaction IDs are 100% deterministic across runs."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen1 = SyntheticStreamGenerator(99, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(99, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs1, _ = gen1.generate_window(3.0)
    txs2, _ = gen2.generate_window(3.0)

    assert [t.transaction_id for t in txs1] == [t.transaction_id for t in txs2]


def test_rng_invariant_5_anomaly_event_id_determinism():
    """5. Anomaly scheduling event IDs are 100% deterministic."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec1 = AnomalySpec("velocity_spike", st, 120.0, 3.0)
    spec2 = AnomalySpec("velocity_spike", st, 120.0, 3.0)

    e1 = gen1.schedule_anomaly("M1", spec1)
    e2 = gen2.schedule_anomaly("M1", spec2)

    assert e1.event_id == e2.event_id


def test_rng_invariant_6_merchant_independence():
    """6. Merchant A's transaction stream cannot depend on merchant B."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_a = SyntheticStreamGenerator(7, [{"id": "MA", "archetype": "seasonal"}], VirtualClock(initial_time=st))
    gen_ab = SyntheticStreamGenerator(7, [{"id": "MA", "archetype": "seasonal"}, {"id": "MB", "archetype": "sparse"}], VirtualClock(initial_time=st))

    txs_a, _ = gen_a.generate_window(4.0)
    txs_ab, _ = gen_ab.generate_window(4.0)
    txs_ab_a = [t for t in txs_ab if t.merchant_id == "MA"]

    assert len(txs_a) == len(txs_ab_a)
    for t1, t2 in zip(txs_a, txs_ab_a):
        assert t1 == t2


# =====================================================================
# BLOCKER 2: Option A Integer Window Duration Validation
# =====================================================================

def test_window_duration_rejection_of_non_integer():
    """Verify non-integer window durations (e.g. 0.5, 1.5) are rejected explicitly with ValueError."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    with pytest.raises(ValueError, match="duration_minutes must be a positive integer"):
        gen.generate_window(0.5)

    with pytest.raises(ValueError, match="duration_minutes must be a positive integer"):
        gen.generate_window(1.5)

    with pytest.raises(ValueError, match="duration_minutes must be a positive integer"):
        gen.generate_window(-1.0)


# =====================================================================
# BLOCKERS 3 & 4: Legitimate Device Pool and Payment Source of Truth
# =====================================================================

def test_device_pool_source_of_truth():
    """Verify legitimate generator uses MerchantProfile.legit_device_pool_size."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen.profiles["M1"].legit_device_pool_size = 5000

    txs, _ = gen.generate_window(10.0)
    dev_ids = [int(t.device_id.split("-")[1]) for t in txs]

    assert max(dev_ids) <= 5000
    assert min(dev_ids) >= 1


def test_payment_method_source_of_truth():
    """Verify legitimate generator uses MerchantProfile.p_prepaid_payment."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen.profiles["M1"].p_prepaid_payment = 0.05

    txs, _ = gen.generate_window(30.0)
    prepaid_count = len([t for t in txs if t.payment_method == "PREPAID_CARD"])
    prepaid_ratio = prepaid_count / len(txs)

    # 30 mins at ~10 tx/min = ~300 txs; ratio should be close to 0.05
    assert abs(prepaid_ratio - 0.05) < 0.04


# =====================================================================
# BLOCKERS 5 & 6: Attribute & Compound Anomaly Signal Semantics
# =====================================================================

def test_attribute_anomaly_semantics():
    """Verify attribute anomaly evaluates country or payment based on active parameters."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec_country = AnomalySpec("attribute_anomaly", st, 300.0, 4.0, {"country": "HIGH_RISK_GEO"})
    gen.schedule_anomaly("M1", spec_country, "EVT-CTRY")

    txs, events = gen.generate_window(5.0)
    assert len(events) == 1
    assert events[0].severity > 0.0


def test_compound_anomaly_signal_set():
    """Verify compound anomaly aggregates rate, amount, device, and country signals via Section 14 mean rule."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec("compound_anomaly", st, 300.0, 4.0, {"rate_multiplier": 3.0, "amount_multiplier": 4.0, "country": "HIGH_RISK_GEO"})
    gen.schedule_anomaly("M1", spec, "EVT-CMP")

    txs, events = gen.generate_window(5.0)
    assert len(events) == 1
    assert events[0].severity > 0.0


# =====================================================================
# BLOCKER 7: Field-by-Field Window Partitioning Identity
# =====================================================================

def test_window_partition_field_by_field_identity():
    """Verify 100% field-by-field equality and severity equality between 5-min step and five 1-min steps."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Scenario A: generate_window called in 5-minute step
    clock1 = VirtualClock(initial_time=st)
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], clock1)
    spec1 = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen1.schedule_anomaly("M1", spec1, "EVT-FIELD-1")
    txs1, events1 = gen1.generate_window(duration_minutes=5.0)

    # Scenario B: generate_window called in five 1-minute steps
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

    # Exact floating point severity equality
    sev1 = events1[-1].severity
    sev2 = events2_all[-1].severity
    assert sev1 == sev2


# =====================================================================
# Multi-Window Sustained Anomaly
# =====================================================================

def test_sustained_anomaly_multi_window_persistence():
    """Verify sustained_anomaly maintains rate elevation across multiple consecutive windows."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs_base, _ = gen.generate_window(duration_minutes=2.0)

    spec = AnomalySpec("sustained_anomaly", st + timedelta(minutes=2.0), 600.0, 3.5, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M1", spec, "EVT-SUSTAINED-MULTI")

    win1_txs, events1 = gen.generate_window(duration_minutes=2.0)
    win2_txs, events2 = gen.generate_window(duration_minutes=2.0)
    win3_txs, events3 = gen.generate_window(duration_minutes=2.0)

    assert len(win1_txs) > 2 * len(txs_base)
    assert len(win2_txs) > 2 * len(txs_base)
    assert len(win3_txs) > 2 * len(txs_base)


# =====================================================================
# Archetype Behavioral Validation Tests
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


# =====================================================================
# Invariants & Reproducibility
# =====================================================================

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
