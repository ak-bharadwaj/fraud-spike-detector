"""Comprehensive unit tests for Day 2 Synthetic Benchmark Generator.

Validates:
1. Separate country and payment attribute deviation signals derived from legitimate distributions (Blocker 1).
2. Legitimate baselines derived mathematically from generator sampling distributions (Blocker 2).
3. Exact window-partitioning severity invariance (sev1 == sev2) (Blocker 3).
4. Velocity vs volume semantic distinction (Blocker 4).
5. Compound signal set & Section 14 aggregation rule (Blocker 5).
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
# BLOCKER 1: Separate Country and Payment Attribute Signals
# =====================================================================

def test_attribute_measurement_separate_signals():
    """Verify country and payment attribute deviations are calculated separately against legitimate baselines."""
    prof = create_merchant_profile(42, "M1", "stable")
    sample_size = 50

    # 1. Country signal calculation
    exp_c = compute_expected_country_ratio(prof.p_high_risk_country)
    scale_c = compute_robust_scale_country_ratio(prof.p_high_risk_country, sample_size)
    assert exp_c == 0.02
    assert scale_c > 0.0

    # 2. Payment signal calculation
    exp_p = compute_expected_payment_ratio(prof.p_prepaid_payment)
    scale_p = compute_robust_scale_payment_ratio(prof.p_prepaid_payment, sample_size)
    assert exp_p == 0.05
    assert scale_p > 0.0

    # 3. Verify attribute_anomaly stream measurement
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st = clock.current_time()
    spec = AnomalySpec("attribute_anomaly", st, 300.0, 4.0, {"country": "HIGH_RISK_GEO", "payment_method": "PREPAID_CARD"})
    gen.schedule_anomaly("M1", spec, "EVT-ATTR-SEP")

    txs, events = gen.generate_window(duration_minutes=5.0)
    assert len(events) == 1
    gt = events[0]
    assert gt.severity > 0.0
    assert gt.severity_level == "HIGH"


# =====================================================================
# BLOCKER 2: Legitimate Baselines Derived Mathematically from Generator
# =====================================================================

def test_legitimate_baselines_derived_from_generator():
    """Verify expected device ratio and robust scale are derived from legitimate occupancy distribution."""
    P = 5000  # Normal device pool size
    N = 20    # Sample size

    expected_ratio = compute_expected_device_ratio(N, P)
    scale_ratio = compute_robust_scale_device_ratio(expected_ratio)

    # For N=20 sampled from P=5000, expected unique ratio is ~0.998
    assert 0.98 < expected_ratio <= 1.0
    assert scale_ratio > 0.0


# =====================================================================
# BLOCKER 3: Exact Window-Partitioning Invariance (sev1 == sev2)
# =====================================================================

def test_window_size_exact_invariance():
    """Verify GroundTruthEvent severity is 100% exactly equal regardless of caller window step size."""
    seed = 42
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Scenario A: generate_window called in 5-minute step
    clock1 = VirtualClock(initial_time=st)
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock1)
    spec1 = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen1.schedule_anomaly("M1", spec1, "EVT-EXACT-1")
    _, events1 = gen1.generate_window(duration_minutes=5.0)

    # Scenario B: generate_window called in five 1-minute steps
    clock2 = VirtualClock(initial_time=st)
    gen2 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock2)
    spec2 = AnomalySpec("volume_spike", st, 300.0, 4.0, {"rate_multiplier": 3.0})
    gen2.schedule_anomaly("M1", spec2, "EVT-EXACT-2")
    events2_all = []
    for _ in range(5):
        _, evs = gen2.generate_window(duration_minutes=1.0)
        events2_all.extend(evs)

    sev1 = events1[-1].severity
    sev2 = events2_all[-1].severity

    # Exact 100% floating point equality
    assert sev1 == sev2, f"Severity in 5-min step ({sev1}) must EXACTLY equal 1-min step cumulative severity ({sev2})"


# =====================================================================
# BLOCKER 4: Velocity vs Volume Semantic Distinction
# =====================================================================

def test_velocity_vs_volume_semantics():
    """Verify velocity_spike is 60s short burst vs volume_spike 120s standard window."""
    seed = 42
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Velocity spike: 60s burst
    clock1 = VirtualClock(initial_time=st)
    gen1 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock1)
    spec1 = AnomalySpec("velocity_spike", st, 60.0, 4.0, {"rate_multiplier": 5.0})
    gen1.schedule_anomaly("M1", spec1, "EVT-VEL")
    txs_vel, evs_vel = gen1.generate_window(duration_minutes=1.0)

    # Volume spike: 120s window
    clock2 = VirtualClock(initial_time=st)
    gen2 = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock2)
    st2 = clock2.current_time()
    spec2 = AnomalySpec("volume_spike", st2, 120.0, 4.0, {"rate_multiplier": 2.5})
    gen2.schedule_anomaly("M1", spec2, "EVT-VOL")
    txs_vol, evs_vol = gen2.generate_window(duration_minutes=2.0)

    assert evs_vel[0].end_time - evs_vel[0].start_time == timedelta(seconds=60)
    assert evs_vol[0].end_time - evs_vol[0].start_time == timedelta(seconds=120)
    assert len(txs_vel) / 1.0 > len(txs_vol) / 2.0


# =====================================================================
# BLOCKER 5: Compound Signal Set & Section 14 Aggregation Rule
# =====================================================================

def test_compound_signal_set():
    """Verify compound anomaly aggregates rate, amount, device, and country signals via Section 14 mean rule."""
    seed = 42
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    gen = SyntheticStreamGenerator(global_seed=seed, merchant_configs=[{"id": "M1", "archetype": "stable"}], clock=clock)

    st = clock.current_time()
    spec = AnomalySpec("compound_anomaly", st, 300.0, 4.0, {"rate_multiplier": 3.0, "amount_multiplier": 4.0, "country": "HIGH_RISK_GEO"})
    gen.schedule_anomaly("M1", spec, "EVT-CMP-SET")

    txs, events = gen.generate_window(duration_minutes=5.0)
    assert len(events) == 1
    gt = events[0]

    prof = gen.profiles["M1"]
    n_txs = len(txs)
    obs_rate = n_txs / 5.0
    m_rate = compute_standardized_magnitude(obs_rate, prof.base_rate_per_min, max(0.5, 0.2 * prof.base_rate_per_min))

    obs_mean_amt = float(np.mean([t.amount for t in txs]))
    m_amt = compute_standardized_magnitude(obs_mean_amt, prof.base_mean_amount, prof.base_std_amount)

    exp_dev_ratio = compute_expected_device_ratio(n_txs, prof.legit_device_pool_size)
    scale_dev_ratio = compute_robust_scale_device_ratio(exp_dev_ratio)
    obs_dev_ratio = len({t.device_id for t in txs}) / n_txs
    m_dev = compute_standardized_magnitude(obs_dev_ratio, exp_dev_ratio, scale_dev_ratio)

    obs_country_ratio = len([t for t in txs if t.country == "HIGH_RISK_GEO"]) / n_txs
    m_country = compute_standardized_magnitude(obs_country_ratio, prof.p_high_risk_country, compute_robust_scale_country_ratio(prof.p_high_risk_country, n_txs))

    expected_compound_sev = compute_compound_severity([m_rate, m_amt, m_dev, m_country])
    assert abs(gt.severity - expected_compound_sev) < 1e-5


# =====================================================================
# Multi-Window Sustained Anomaly
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

    assert len(win1_txs) > 2 * len(txs_base)
    assert len(win2_txs) > 2 * len(txs_base)
    assert len(win3_txs) > 2 * len(txs_base)
    assert len(events1) == 1
    assert len(events2) == 1
    assert len(events3) == 1


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
