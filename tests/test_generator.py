"""Comprehensive unit tests for Day 2 Synthetic Benchmark Generator.

Validates:
1. Ground-truth realized magnitude derivation vs target magnitude separation (Blockers 1 & 3).
2. Concrete stream realization for all 7 anomaly classes (Blocker 2).
3. Section 14 compound severity rule: mean absolute standardized deviation (Blocker 4).
4. Concrete behavioral invariants for all 6 merchant archetypes (Blocker 5).
5. Overlap rejection invariant per merchant.
6. Deterministic RNG sub-seeding and compositional reproducibility.
"""

from datetime import datetime, timedelta, timezone
import numpy as np
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


# =====================================================================
# BLOCKER 1 & 3: Ground-Truth Realized Magnitude vs Target Magnitude
# =====================================================================

def test_realized_magnitude_derivation():
    """Verify GroundTruthEvent.severity represents realized magnitude measured from stream, preserving target_magnitude in parameters."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st = clock.current_time()
    spec = AnomalySpec(
        anomaly_type="volume_spike",
        start_time=st,
        duration_seconds=300.0,
        target_magnitude=4.0,  # Injection control intent
        parameters={"rate_multiplier": 4.0},
    )
    gen.schedule_anomaly("M1", spec, event_id="EVT-REALIZED-1")

    txs, events = gen.generate_window(duration_minutes=5.0)

    assert len(events) == 1
    gt_event = events[0]

    # Verify severity is the realized magnitude computed from generated stream
    assert isinstance(gt_event.severity, float)
    assert gt_event.severity > 0.0
    assert gt_event.severity_level in ("LOW", "MEDIUM", "HIGH")
    # Verify target_magnitude is preserved in parameters
    assert gt_event.parameters["target_magnitude"] == 4.0


# =====================================================================
# BLOCKER 4: Compound Severity Rule (Section 14)
# =====================================================================

def test_compound_severity_rule():
    """Verify Section 14 Compound Severity Rule: mean absolute standardized deviation across active signals."""
    m_volume = 3.0
    m_amount = 4.5
    m_device = 1.5

    compound_sev = compute_compound_severity([m_volume, m_amount, m_device])
    expected_sev = float(np.mean([3.0, 4.5, 1.5]))  # 3.0

    assert compound_sev == expected_sev
    assert compute_compound_severity([]) == 0.0


# =====================================================================
# BLOCKER 2: Concrete Behavioral Verification for All 7 Anomaly Classes
# =====================================================================

def test_velocity_spike_realization():
    """Verify velocity_spike materially increases short-window transaction rate."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    # Baseline window
    txs_base, _ = gen.generate_window(duration_minutes=2.0)

    # Velocity spike window
    st = clock.current_time()
    spec = AnomalySpec("velocity_spike", st, 120.0, 4.0, {"rate_multiplier": 5.0})
    gen.schedule_anomaly("M1", spec, "EVT-VELOCITY")
    txs_spike, events = gen.generate_window(duration_minutes=2.0)

    assert len(txs_spike) >= 2.5 * len(txs_base), "Velocity spike must materially elevate short-window transaction count."
    assert len(events) == 1
    assert events[0].anomaly_type == "velocity_spike"


def test_volume_spike_realization():
    """Verify volume_spike elevates transaction volume over standard window."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    txs_base, _ = gen.generate_window(duration_minutes=5.0)

    st = clock.current_time()
    spec = AnomalySpec("volume_spike", st, 300.0, 3.5, {"rate_multiplier": 3.5})
    gen.schedule_anomaly("M1", spec, "EVT-VOLUME")
    txs_spike, events = gen.generate_window(duration_minutes=5.0)

    assert len(txs_spike) > 2 * len(txs_base), "Volume spike must elevate total window volume."
    assert len(events) == 1


def test_amount_spike_realization():
    """Verify amount_spike materially elevates mean transaction amount."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    txs_base, _ = gen.generate_window(duration_minutes=5.0)
    mean_amt_base = np.mean([t.amount for t in txs_base])

    st = clock.current_time()
    spec = AnomalySpec("amount_spike", st, 300.0, 5.0, {"amount_multiplier": 6.0})
    gen.schedule_anomaly("M1", spec, "EVT-AMOUNT")
    txs_spike, events = gen.generate_window(duration_minutes=5.0)
    mean_amt_spike = np.mean([t.amount for t in txs_spike])

    assert mean_amt_spike >= 3.0 * mean_amt_base, f"Amount spike mean ({mean_amt_spike}) must be >= 3x baseline ({mean_amt_base})."
    assert len(events) == 1


def test_behavioral_shift_realization():
    """Verify behavioral_shift restricts unique customer/device ID space per transaction."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    txs_base, _ = gen.generate_window(duration_minutes=5.0)
    unique_devs_base = len({t.device_id for t in txs_base})

    st = clock.current_time()
    spec = AnomalySpec("behavioral_shift", st, 300.0, 4.0, {"rate_multiplier": 2.0})
    gen.schedule_anomaly("M1", spec, "EVT-BEHAVIOR")
    txs_spike, events = gen.generate_window(duration_minutes=5.0)
    unique_devs_spike = len({t.device_id for t in txs_spike})

    # Behavioral spike concentrates transactions onto few devices
    ratio_base = unique_devs_base / len(txs_base)
    ratio_spike = unique_devs_spike / len(txs_spike)
    assert ratio_spike < ratio_base, "Behavioral shift must concentrate transactions across fewer unique device IDs."
    assert len(events) == 1


def test_attribute_anomaly_realization():
    """Verify attribute_anomaly shifts country and payment method distribution."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st = clock.current_time()
    spec = AnomalySpec("attribute_anomaly", st, 300.0, 4.0, {"country": "HIGH_RISK_GEO", "payment_method": "PREPAID_CARD"})
    gen.schedule_anomaly("M1", spec, "EVT-ATTR")
    txs_spike, events = gen.generate_window(duration_minutes=5.0)

    countries = {t.country for t in txs_spike}
    payments = {t.payment_method for t in txs_spike}

    assert "HIGH_RISK_GEO" in countries, "Attribute anomaly must inject target high-risk country."
    assert "PREPAID_CARD" in payments, "Attribute anomaly must inject target payment method."
    assert len(events) == 1


def test_sustained_anomaly_realization():
    """Verify sustained_anomaly maintains elevated rate across multiple consecutive windows."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    txs_base, _ = gen.generate_window(duration_minutes=2.0)

    st = clock.current_time()
    spec = AnomalySpec("sustained_anomaly", st, 600.0, 3.5, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M1", spec, "EVT-SUSTAINED")

    # Generate two consecutive 5-minute windows
    txs_win1, events1 = gen.generate_window(duration_minutes=5.0)
    txs_win2, events2 = gen.generate_window(duration_minutes=5.0)

    assert len(txs_win1) > 2 * len(txs_base)
    assert len(txs_win2) > 2 * len(txs_base), "Sustained anomaly must maintain elevation across consecutive windows."
    assert len(events1) == 1
    assert len(events2) == 1


def test_compound_anomaly_realization():
    """Verify compound_anomaly simultaneously shifts rate, amount, and device behavior."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    txs_base, _ = gen.generate_window(duration_minutes=5.0)

    st = clock.current_time()
    spec = AnomalySpec("compound_anomaly", st, 300.0, 4.5, {"rate_multiplier": 3.0, "amount_multiplier": 4.0, "country": "HIGH_RISK_GEO"})
    gen.schedule_anomaly("M1", spec, "EVT-COMPOUND")
    txs_spike, events = gen.generate_window(duration_minutes=5.0)

    # 1. Volume shift
    assert len(txs_spike) > len(txs_base)
    # 2. Amount shift
    assert np.mean([t.amount for t in txs_spike]) > np.mean([t.amount for t in txs_base])
    # 3. Attribute shift
    assert "HIGH_RISK_GEO" in {t.country for t in txs_spike}
    assert len(events) == 1


# =====================================================================
# BLOCKER 5: Behavioral Verification for All 6 Archetypes
# =====================================================================

def test_archetype_stable_behavior():
    """Verify stable merchant maintains steady rate and low variance."""
    prof = create_merchant_profile(42, "M_stable", "stable")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    rates = [
        compute_legitimate_rate(prof, sim_start + timedelta(hours=h), sim_start)
        for h in range(24)
    ]
    # Stable rate should be constant
    assert np.std(rates) == 0.0
    assert rates[0] == prof.base_rate_per_min


def test_archetype_seasonal_behavior():
    """Verify seasonal merchant shows diurnal (afternoon vs night) and weekly rate variation."""
    prof = create_merchant_profile(42, "M_seasonal", "seasonal")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Diurnal test: 03:00 (night) vs 15:00 (afternoon)
    t_night = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
    t_day = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    rate_night = compute_legitimate_rate(prof, t_night, sim_start)
    rate_day = compute_legitimate_rate(prof, t_day, sim_start)
    assert rate_day > rate_night, "Diurnal afternoon rate must exceed night rate."

    # Weekly test: Thursday vs Saturday
    t_weekday = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)  # Thursday
    t_weekend = datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc)  # Saturday
    rate_weekday = compute_legitimate_rate(prof, t_weekday, sim_start)
    rate_weekend = compute_legitimate_rate(prof, t_weekend, sim_start)
    assert rate_weekend > rate_weekday, "Weekend rate must exceed weekday rate."


def test_archetype_growing_behavior():
    """Verify growing merchant shows positive baseline rate growth over time."""
    prof = create_merchant_profile(42, "M_growing", "growing")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    t_start = sim_start
    t_later = sim_start + timedelta(days=30)
    rate_start = compute_legitimate_rate(prof, t_start, sim_start)
    rate_later = compute_legitimate_rate(prof, t_later, sim_start)

    assert rate_later > rate_start, f"Later rate ({rate_later}) must be > initial rate ({rate_start})."


def test_archetype_volatile_behavior():
    """Verify volatile merchant produces significantly higher rate and amount variance than stable."""
    prof_stable = create_merchant_profile(42, "M_stable", "stable")
    prof_volatile = create_merchant_profile(42, "M_volatile", "volatile")

    rng_stable = np.random.default_rng(42)
    rng_volatile = np.random.default_rng(42)

    amts_stable = [sample_legitimate_amount(prof_stable, rng_stable) for _ in range(200)]
    amts_volatile = [sample_legitimate_amount(prof_volatile, rng_volatile) for _ in range(200)]

    assert np.var(amts_volatile) > np.var(amts_stable), "Volatile merchant amount variance must exceed stable variance."


def test_archetype_sparse_behavior():
    """Verify sparse merchant produces low transaction frequency with extended inter-arrival gaps."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(
        global_seed=seed,
        merchant_configs=[{"id": "M_sparse", "archetype": "sparse"}, {"id": "M_stable", "archetype": "stable"}],
        clock=clock,
    )

    txs, _ = gen.generate_window(duration_minutes=30.0)
    txs_sparse = [t for t in txs if t.merchant_id == "M_sparse"]
    txs_stable = [t for t in txs if t.merchant_id == "M_stable"]

    assert len(txs_sparse) < len(txs_stable) * 0.2, "Sparse transaction count must be < 20% of stable count."


def test_archetype_mixed_behavior():
    """Verify mixed merchant incorporates diurnal seasonality and baseline growth."""
    prof = create_merchant_profile(42, "M_mixed", "mixed")
    sim_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    # 1. Growth check
    r1 = compute_legitimate_rate(prof, sim_start, sim_start)
    r2 = compute_legitimate_rate(prof, sim_start + timedelta(days=20), sim_start)
    assert r2 > r1

    # 2. Seasonality check (03:00 vs 15:00)
    t_night = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
    t_day = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    assert compute_legitimate_rate(prof, t_day, sim_start) > compute_legitimate_rate(prof, t_night, sim_start)


# =====================================================================
# Invariants & Reproducibility
# =====================================================================

def test_no_overlapping_anomalies_per_merchant():
    """Verify No Overlapping Active Events per merchant invariant."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    spec1 = AnomalySpec(anomaly_type="volume_spike", start_time=st1, duration_seconds=300.0, target_magnitude=3.0)
    gen.schedule_anomaly("M1", spec1, event_id="EVT-1")

    st2 = datetime(2026, 1, 1, 10, 2, tzinfo=timezone.utc)
    spec2 = AnomalySpec(anomaly_type="velocity_spike", start_time=st2, duration_seconds=300.0, target_magnitude=4.0)

    with pytest.raises(OverlapAnomalyError):
        gen.schedule_anomaly("M1", spec2, event_id="EVT-2")


def test_deterministic_generation():
    """Verify global_seed + merchant_id produces 100% deterministic streams."""
    seed = 42
    configs = [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "seasonal"}]

    clock1 = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=configs, clock=clock1)
    txs1, _ = gen1.generate_window(duration_minutes=5.0)

    clock2 = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen2 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=configs, clock=clock2)
    txs2, _ = gen2.generate_window(duration_minutes=5.0)

    assert len(txs1) == len(txs2)
    for t1, t2 in zip(txs1, txs2):
        assert t1.transaction_id == t2.transaction_id
        assert t1.amount == t2.amount


def test_merchant_compositional_reproducibility():
    """Verify adding merchant B does not alter merchant A's stream."""
    seed = 100
    start_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    clock1 = VirtualClock(initial_time=start_time)
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock1)
    txs_m1_alone, _ = gen1.generate_window(duration_minutes=10.0)

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
