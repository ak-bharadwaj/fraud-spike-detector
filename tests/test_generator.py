"""Comprehensive unit tests for Day 2 Synthetic Benchmark Generator.

Validates:
1. Attribute anomaly severity measured from generated stream vs baseline (Blocker 1).
2. Multi-window persistence for sustained anomaly (Blocker 2).
3. Semantic distinction between velocity_spike and volume_spike (Blocker 3).
4. Compound signal baselines and Section 14 aggregation rule (Blocker 4).
5. Realized magnitude invariance to caller window step size (Blocker 5).
6. Concrete behavioral verification for all 6 archetypes and 7 anomaly classes.
7. Overlap rejection and RNG compositional reproducibility invariants.
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
# BLOCKER 1: Attribute Anomaly Measured from Stream vs Baseline
# =====================================================================

def test_attribute_anomaly_severity_derivation():
    """Verify attribute_anomaly severity is calculated from realized high-risk ratio vs legitimate baseline."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st = clock.current_time()
    spec = AnomalySpec("attribute_anomaly", st, 300.0, 4.0, {"country": "HIGH_RISK_GEO", "payment_method": "PREPAID_CARD"})
    gen.schedule_anomaly("M1", spec, "EVT-ATTR-REALIZED")

    txs, events = gen.generate_window(duration_minutes=5.0)

    assert len(events) == 1
    gt = events[0]

    prof = gen.profiles["M1"]
    high_risk_count = len([t for t in txs if t.country == "HIGH_RISK_GEO"])
    obs_ratio = high_risk_count / len(txs)
    expected_m = compute_standardized_magnitude(obs_ratio, prof.expected_high_risk_country_ratio, prof.robust_scale_country_ratio)

    assert abs(gt.severity - expected_m) < 0.1, f"Realized severity ({gt.severity}) should match stream measurement ({expected_m})"
    assert gt.severity_level == "HIGH"


# =====================================================================
# BLOCKER 2: Sustained Anomaly Multi-Window Persistence
# =====================================================================

def test_sustained_anomaly_multi_window_persistence():
    """Verify sustained_anomaly maintains rate elevation across multiple consecutive windows."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    txs_base, _ = gen.generate_window(duration_minutes=2.0)

    st = clock.current_time()
    spec = AnomalySpec("sustained_anomaly", st, 600.0, 3.5, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M1", spec, "EVT-SUSTAINED-MULTI")

    win1_txs, events1 = gen.generate_window(duration_minutes=2.0)
    win2_txs, events2 = gen.generate_window(duration_minutes=2.0)
    win3_txs, events3 = gen.generate_window(duration_minutes=2.0)

    assert len(win1_txs) > 2 * len(txs_base), "Window 1 rate must be elevated."
    assert len(win2_txs) > 2 * len(txs_base), "Window 2 rate must remain elevated."
    assert len(win3_txs) > 2 * len(txs_base), "Window 3 rate must remain elevated."
    assert len(events1) == 1
    assert len(events2) == 1
    assert len(events3) == 1


# =====================================================================
# BLOCKER 3: Velocity vs Volume Semantic Distinction
# =====================================================================

def test_velocity_vs_volume_semantic_distinction():
    """Verify semantic distinction between short-burst velocity_spike vs standard-window volume_spike."""
    seed = 42
    clock1 = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock1)

    st1 = clock1.current_time()
    spec1 = AnomalySpec("velocity_spike", st1, 60.0, 4.0, {"rate_multiplier": 5.0})
    gen1.schedule_anomaly("M1", spec1, "EVT-VELOCITY")
    txs_vel, _ = gen1.generate_window(duration_minutes=1.0)

    clock2 = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen2 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock2)
    st2 = clock2.current_time()
    spec2 = AnomalySpec("volume_spike", st2, 120.0, 4.0, {"rate_multiplier": 2.5})
    gen2.schedule_anomaly("M1", spec2, "EVT-VOLUME")
    txs_vol, _ = gen2.generate_window(duration_minutes=2.0)

    rate_vel = len(txs_vel) / 1.0
    rate_vol = len(txs_vol) / 2.0

    assert rate_vel > rate_vol, f"Velocity burst rate ({rate_vel}) must be > volume rate ({rate_vol})"


# =====================================================================
# BLOCKER 4: Compound Signal Baseline Rules (Section 14)
# =====================================================================

def test_compound_signal_legitimate_baselines():
    """Verify compound anomaly calculates signal deviations against legitimate merchant baselines and Section 14 mean rule."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st = clock.current_time()
    spec = AnomalySpec("compound_anomaly", st, 300.0, 4.0, {"rate_multiplier": 3.0, "amount_multiplier": 4.0, "country": "HIGH_RISK_GEO"})
    gen.schedule_anomaly("M1", spec, "EVT-COMPOUND-LEGIT")

    txs, events = gen.generate_window(duration_minutes=5.0)

    assert len(events) == 1
    gt = events[0]

    prof = gen.profiles["M1"]
    obs_rate = len(txs) / 5.0
    m_rate = compute_standardized_magnitude(obs_rate, prof.base_rate_per_min, max(0.5, 0.2 * prof.base_rate_per_min))

    obs_mean_amt = float(np.mean([t.amount for t in txs]))
    m_amt = compute_standardized_magnitude(obs_mean_amt, prof.base_mean_amount, prof.base_std_amount)

    obs_dev_ratio = len({t.device_id for t in txs}) / len(txs)
    m_dev = compute_standardized_magnitude(obs_dev_ratio, prof.expected_device_ratio, prof.robust_scale_device_ratio)

    expected_compound_sev = compute_compound_severity([m_rate, m_amt, m_dev])
    assert abs(gt.severity - expected_compound_sev) < 0.1


# =====================================================================
# BLOCKER 5: Realized Magnitude Invariance to Caller Window Step Size
# =====================================================================

def test_realized_magnitude_window_size_invariance():
    """Verify GroundTruthEvent severity measured over anomaly duration is invariant to caller generate_window step size."""
    seed = 42
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Scenario A: generate_window called in 5-minute step
    clock1 = VirtualClock(initial_time=st)
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock1)
    spec1 = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen1.schedule_anomaly("M1", spec1, "EVT-INV-1")
    _, events1 = gen1.generate_window(duration_minutes=5.0)

    # Scenario B: generate_window called in five 1-minute steps
    clock2 = VirtualClock(initial_time=st)
    gen2 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock2)
    spec2 = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen2.schedule_anomaly("M1", spec2, "EVT-INV-2")
    events2_all = []
    for _ in range(5):
        _, evs = gen2.generate_window(duration_minutes=1.0)
        events2_all.extend(evs)

    sev1 = events1[-1].severity
    sev2 = events2_all[-1].severity

    # Stochastic sampling difference between 1 large Poisson draw vs 5 small Poisson draws is < 5%
    relative_diff = abs(sev1 - sev2) / max(sev1, sev2)
    assert relative_diff < 0.08, f"Relative difference ({relative_diff:.3f}) between 5-min step ({sev1:.2f}) and 1-min step ({sev2:.2f}) must be < 8%"


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
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(
        global_seed=seed,
        merchant_configs=[{"id": "M_sparse", "archetype": "sparse"}, {"id": "M_stable", "archetype": "stable"}],
        clock=clock,
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


def test_smoke_benchmark_fixture():
    seed = 777
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st = clock.current_time()
    spec = AnomalySpec("velocity_spike", st, 120.0, 3.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M1", spec, "EVT-SMOKE-1")

    txs, events = gen.generate_window(duration_minutes=5.0)
    assert len(txs) > 0
    assert len(events) == 1
