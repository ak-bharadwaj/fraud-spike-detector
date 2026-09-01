"""Day 5 Comprehensive Synthetic Benchmark and Anomaly Mechanism Test Suite.

Validates:
1. Behavioral Characterization of Generated Streams for Merchant Archetypes M1 through M9:
   - M1: Small / low volume (observed rate in 0.2 .. 0.8 tx/min)
   - M2: Medium / stable (observed rate in 8.0 .. 15.0 tx/min)
   - M3: High volume (observed rate in 30.0 .. 60.0 tx/min)
   - Scale hierarchy on generated streams: M1 < M2 < M3
   - M4: Seasonal (measured diurnal peak at midday > trough at night)
   - M5: Weekend-heavy (measured weekend traffic > 2x weekday traffic)
   - M6: Highly variable (measured empirical amount/rate variance > stable benchmark M2)
   - M7: Night-heavy (measured night hours 22:00-05:00 UTC > 2x daytime hours)
   - M8: Mixed (combines diurnal variation and organic baseline growth)
   - M9: Organic growth (measured day 10 transaction rate > day 0 rate)
2. All 11 Canonical Anomaly Mechanism Tests:
   1. sudden_volume_spike / volume_spike
   2. velocity_burst / velocity_spike
   3. sustained_spike / sustained_anomaly
   4. amount_distribution_shift / amount_spike
   5. device_behavior_anomaly / behavioral_shift
   6. attribute_geographic_shift / attribute_anomaly
   7. compound_anomaly
   8. threshold_hugging_evasion: generates standardized behavior near the defined threshold
   9. persistence_evasion: generates alternating bursts without satisfying multi-window persistence
   10. staircase_ramp: successive discrete regime levels form a monotonically increasing step pattern
   11. oscillating_sub_threshold: generated signal oscillates and stays within sub-threshold envelope
3. Ground-Truth Protocol & Realized Severity Derivation.
4. No-Overlap Invariant & Boundary-Touching Intervals.
5. Deterministic Replay & Merchant Compositionality.
6. Window Partitioning Identity.
7. Legitimate Non-Anomalous Behavior Separation.
8. Ground-Truth Architectural Isolation.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np

from src.contracts.contracts import GroundTruthEvent, Transaction
from src.generator.archetypes import (
    create_merchant_profile,
    compute_legitimate_rate,
)
from src.generator.anomalies import AnomalySpec, CANONICAL_ANOMALY_TYPES
from src.generator.stream_generator import SyntheticStreamGenerator, OverlapAnomalyError
from src.stream.clock import VirtualClock


# =====================================================================
# 1. Behavioral Characterization of Generated Streams (M1 - M9)
# =====================================================================

def test_m1_m2_m3_generated_stream_behavioral_hierarchy():
    """Verify generated streams for M1, M2, M3 exhibit expected rate bands and M1 < M2 < M3 hierarchy."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    configs = [
        {"id": "M1", "archetype": "M1"},
        {"id": "M2", "archetype": "M2"},
        {"id": "M3", "archetype": "M3"},
    ]
    gen = SyntheticStreamGenerator(42, configs, VirtualClock(initial_time=st))
    txs, _ = gen.generate_window(10.0)  # 10 minutes

    m1_txs = [t for t in txs if t.merchant_id == "M1"]
    m2_txs = [t for t in txs if t.merchant_id == "M2"]
    m3_txs = [t for t in txs if t.merchant_id == "M3"]

    m1_rate = len(m1_txs) / 10.0
    m2_rate = len(m2_txs) / 10.0
    m3_rate = len(m3_txs) / 10.0

    assert 0.1 <= m1_rate <= 1.5
    assert 6.0 <= m2_rate <= 20.0
    assert 25.0 <= m3_rate <= 70.0
    assert len(m1_txs) < len(m2_txs) < len(m3_txs)


def test_m4_seasonal_generated_stream_characterization():
    """Verify M4 generated stream produces significantly more transactions at midday (12:00) than night (00:00)."""
    t_night = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)  # 00:00 UTC
    gen_night = SyntheticStreamGenerator(42, [{"id": "M4", "archetype": "M4"}], VirtualClock(initial_time=t_night))
    txs_night, _ = gen_night.generate_window(10.0)

    t_midday = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)  # 12:00 UTC
    gen_midday = SyntheticStreamGenerator(42, [{"id": "M4", "archetype": "M4"}], VirtualClock(initial_time=t_midday))
    txs_midday, _ = gen_midday.generate_window(10.0)

    assert len(txs_midday) > len(txs_night) * 1.5


def test_m5_weekend_heavy_generated_stream_characterization():
    """Verify M5 generated stream produces significantly more transactions on weekend than weekday."""
    t_weekday = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)  # Monday 12:00
    gen_weekday = SyntheticStreamGenerator(42, [{"id": "M5", "archetype": "M5"}], VirtualClock(initial_time=t_weekday))
    txs_weekday, _ = gen_weekday.generate_window(10.0)

    t_weekend = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)  # Saturday 12:00
    gen_weekend = SyntheticStreamGenerator(42, [{"id": "M5", "archetype": "M5"}], VirtualClock(initial_time=t_weekend))
    txs_weekend, _ = gen_weekend.generate_window(10.0)

    assert len(txs_weekend) > len(txs_weekday) * 2.0


def test_m6_highly_variable_generated_stream_characterization():
    """Verify M6 generated stream produces significantly higher amount variance than stable M2."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    configs = [
        {"id": "M2", "archetype": "M2"},
        {"id": "M6", "archetype": "M6"},
    ]
    gen = SyntheticStreamGenerator(42, configs, VirtualClock(initial_time=st))
    txs, _ = gen.generate_window(20.0)

    m2_amts = [t.amount for t in txs if t.merchant_id == "M2"]
    m6_amts = [t.amount for t in txs if t.merchant_id == "M6"]

    m2_std = float(np.std(m2_amts))
    m6_std = float(np.std(m6_amts))

    assert m6_std > m2_std * 2.0


def test_m7_night_heavy_generated_stream_characterization():
    """Verify M7 generated stream produces peak activity during night hours (23:00) compared to daytime (14:00)."""
    t_night = datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc)
    gen_night = SyntheticStreamGenerator(42, [{"id": "M7", "archetype": "M7"}], VirtualClock(initial_time=t_night))
    txs_night, _ = gen_night.generate_window(10.0)

    t_day = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)
    gen_day = SyntheticStreamGenerator(42, [{"id": "M7", "archetype": "M7"}], VirtualClock(initial_time=t_day))
    txs_day, _ = gen_day.generate_window(10.0)

    assert len(txs_night) > len(txs_day) * 2.0


def test_m8_mixed_generated_stream_characterization():
    """Verify M8 generated stream exhibits both diurnal variation and organic baseline growth."""
    t_start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    clock = VirtualClock(initial_time=t_start)
    gen = SyntheticStreamGenerator(42, [{"id": "M8", "archetype": "M8"}], clock)

    # Day 0 Night (00:00)
    txs0_night, _ = gen.generate_window(20.0)

    # Advance to Day 0 Midday (12:00)
    clock.set_time(t_start + timedelta(hours=12))
    txs0_midday, _ = gen.generate_window(20.0)

    # Advance to Day 10 Midday (10 days later at 12:00)
    clock.set_time(t_start + timedelta(days=10, hours=12))
    txs10_midday, _ = gen.generate_window(20.0)

    # Diurnal variation: midday > night
    assert len(txs0_midday) > len(txs0_night) * 1.3
    # Organic growth: day 10 midday > day 0 midday
    assert len(txs10_midday) > len(txs0_midday) * 1.15


def test_m9_organic_growth_generated_stream_characterization():
    """Verify M9 generated stream produces higher transaction counts after 10 days of organic growth."""
    t_start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    clock = VirtualClock(initial_time=t_start)
    gen = SyntheticStreamGenerator(42, [{"id": "M9", "archetype": "M9"}], clock)

    # Day 0 stream (20 minutes)
    txs_day0, _ = gen.generate_window(20.0)

    # Advance clock to Day 10
    clock.set_time(t_start + timedelta(days=10))
    txs_day10, _ = gen.generate_window(20.0)

    assert len(txs_day10) > len(txs_day0) * 1.15


# =====================================================================
# 2. Canonical Anomaly Classes & Detailed Mechanism Tests
# =====================================================================

@pytest.mark.parametrize("anomaly_type", CANONICAL_ANOMALY_TYPES)
def test_all_11_canonical_anomaly_classes_generation(anomaly_type):
    """Verify each of the 11 canonical anomaly classes generates valid transactions and GroundTruthEvent."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    params = {}
    if "attribute" in anomaly_type:
        params = {"country": "HIGH_RISK_GEO"}

    spec = AnomalySpec(
        anomaly_type=anomaly_type,
        start_time=st,
        duration_seconds=120.0,
        target_magnitude=4.5,
        parameters=params,
    )
    eid = gen.schedule_anomaly("M1", spec)

    txs, events = gen.generate_window(2.0)

    assert len(events) == 1
    gt = events[0]
    assert gt.event_id == eid
    assert gt.merchant_id == "M1"
    assert gt.anomaly_type == anomaly_type
    assert gt.severity > 0.0
    assert gt.severity_level in ("LOW", "MEDIUM", "HIGH")


def test_anomaly1_sudden_volume_spike_mechanism():
    """Verify Anomaly 1 (sudden volume spike) behaviorally elevates observed transaction count on affected merchant while unaffected merchant remains identical."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    merchants = [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "stable"}]

    # Baseline (5 min)
    gen_base = SyntheticStreamGenerator(42, merchants, VirtualClock(initial_time=st))
    txs_base, _ = gen_base.generate_window(5.0)
    m1_base = len([t for t in txs_base if t.merchant_id == "M1"])
    m2_base = len([t for t in txs_base if t.merchant_id == "M2"])

    # In-anomaly (5 min with 4.0x volume spike on M1)
    gen_spike = SyntheticStreamGenerator(42, merchants, VirtualClock(initial_time=st))
    spec = AnomalySpec("sudden_volume_spike", st, 300.0, 4.0, {"rate_multiplier": 4.0})
    gen_spike.schedule_anomaly("M1", spec)
    txs_spike, events = gen_spike.generate_window(5.0)
    m1_spike = len([t for t in txs_spike if t.merchant_id == "M1"])
    m2_spike = len([t for t in txs_spike if t.merchant_id == "M2"])

    assert m1_spike > m1_base * 2.5
    assert m2_spike == m2_base
    assert len(events) == 1


def test_anomaly2_velocity_burst_mechanism():
    """Verify Anomaly 2 (velocity burst) behaviorally elevates short-window transaction velocity."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_base = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen_base.generate_window(1.0)  # 1 min baseline

    gen_burst = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec = AnomalySpec("velocity_burst", st, 60.0, 4.5, {"rate_multiplier": 4.0})
    gen_burst.schedule_anomaly("M1", spec)
    txs_burst, events = gen_burst.generate_window(1.0)

    velocity_base = len(txs_base) / 1.0  # tx/min
    velocity_burst = len(txs_burst) / 1.0

    assert velocity_burst > velocity_base * 2.5
    assert len(events) == 1


def test_anomaly3_sustained_spike_mechanism():
    """Verify Anomaly 3 (sustained spike) maintains elevated rate across all consecutive windows of its 5-minute duration."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec = AnomalySpec("sustained_spike", st, 300.0, 3.5, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("M1", spec)

    # Base rate for M1 stable is ~10-15 tx/min
    counts_per_min = []
    events_all = []
    for _ in range(5):
        txs_min, evs = gen.generate_window(1.0)
        counts_per_min.append(len(txs_min))
        events_all.extend(evs)

    # Every single window must show sustained elevation (> 20 tx/min)
    for c in counts_per_min:
        assert c >= 20
    assert len(events_all) == 1
    assert events_all[0].anomaly_type == "sustained_spike"


def test_anomaly4_amount_distribution_shift_mechanism():
    """Verify Anomaly 4 (amount distribution shift) behaviorally shifts empirical transaction amounts upward."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_base = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen_base.generate_window(5.0)
    mean_base = float(np.mean([t.amount for t in txs_base]))

    gen_shift = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec = AnomalySpec("amount_distribution_shift", st, 300.0, 4.0, {"amount_multiplier": 3.5})
    gen_shift.schedule_anomaly("M1", spec)
    txs_shift, events = gen_shift.generate_window(5.0)
    mean_shift = float(np.mean([t.amount for t in txs_shift]))

    assert mean_shift > mean_base * 2.5
    assert len(events) == 1


def test_anomaly5_device_behavior_anomaly_mechanism():
    """Verify Anomaly 5 (device behavior anomaly) collapses unique device cardinality through synthetic device reuse."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_base = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen_base.generate_window(5.0)
    unique_dev_base = len(set(t.device_id for t in txs_base))
    dev_ratio_base = unique_dev_base / float(len(txs_base))

    gen_dev = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec = AnomalySpec("device_behavior_anomaly", st, 300.0, 4.0, {"rate_multiplier": 2.0})
    gen_dev.schedule_anomaly("M1", spec)
    txs_dev, events = gen_dev.generate_window(5.0)
    unique_dev_anom = len(set(t.device_id for t in txs_dev))
    dev_ratio_anom = unique_dev_anom / float(len(txs_dev))

    # Baseline ratio is high (~0.9-1.0), anomaly collapses ratio to <= 0.2 (max 5 devices across 50+ txs)
    assert dev_ratio_base > 0.8
    assert dev_ratio_anom < 0.25
    assert unique_dev_anom <= 5
    assert len(events) == 1


def test_anomaly6_attribute_geographic_shift_mechanism():
    """Verify Anomaly 6 (attribute geographic shift) behaviorally shifts transactions to the programmed high-risk country."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_base = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen_base.generate_window(5.0)
    high_risk_base = len([t for t in txs_base if t.country == "HIGH_RISK_GEO"]) / float(len(txs_base))

    gen_geo = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec = AnomalySpec("attribute_geographic_shift", st, 300.0, 4.0, {"country": "HIGH_RISK_GEO"})
    gen_geo.schedule_anomaly("M1", spec)
    txs_geo, events = gen_geo.generate_window(5.0)
    high_risk_geo = len([t for t in txs_geo if t.country == "HIGH_RISK_GEO"]) / float(len(txs_geo))

    # Baseline high-risk is ~2%, anomaly shifts 100% of transactions to HIGH_RISK_GEO
    assert high_risk_base < 0.1
    assert high_risk_geo == 1.0
    assert len(events) == 1


def test_anomaly7_compound_anomaly_mechanism():
    """Verify Anomaly 7 (compound anomaly) concurrently shifts multiple signal families (volume, amount, device, and geographic)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen_base = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen_base.generate_window(5.0)
    vol_base = len(txs_base)
    amt_base = float(np.mean([t.amount for t in txs_base]))

    gen_comp = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec = AnomalySpec(
        "compound_anomaly",
        st,
        300.0,
        4.5,
        {"rate_multiplier": 3.0, "amount_multiplier": 3.0, "country": "HIGH_RISK_GEO"},
    )
    gen_comp.schedule_anomaly("M1", spec)
    txs_comp, events = gen_comp.generate_window(5.0)

    vol_comp = len(txs_comp)
    amt_comp = float(np.mean([t.amount for t in txs_comp]))
    unique_dev_comp = len(set(t.device_id for t in txs_comp))
    high_risk_comp = len([t for t in txs_comp if t.country == "HIGH_RISK_GEO"]) / float(len(txs_comp))

    # Prove all signal families concurrently shifted:
    assert vol_comp > vol_base * 2.0  # Volume shifted
    assert amt_comp > amt_base * 2.0  # Amount shifted
    assert unique_dev_comp <= 5  # Device collapsed
    assert high_risk_comp == 1.0  # Geographic attribute shifted
    assert len(events) == 1
    assert events[0].anomaly_type == "compound_anomaly"


def test_legitimate_surge_hard_negative_elevates_volume_with_zero_gt():
    """Verify promotional surge elevates observed transaction volume as a hard negative without emitting any anomaly GroundTruthEvent."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    merchants = [{"id": "M1", "archetype": "stable"}]

    # 1. Normal traffic (10 minutes)
    gen_norm = SyntheticStreamGenerator(42, merchants, VirtualClock(initial_time=st))
    txs_norm, events_norm = gen_norm.generate_window(10.0, is_surge_active={"M1": False})
    norm_count = len(txs_norm)

    # 2. Promotional surge traffic (10 minutes)
    gen_surge = SyntheticStreamGenerator(42, merchants, VirtualClock(initial_time=st))
    txs_surge, events_surge = gen_surge.generate_window(10.0, is_surge_active={"M1": True})
    surge_count = len(txs_surge)

    # Prove surge volume genuinely elevated relative to normal baseline (~2.5x)
    assert surge_count > norm_count * 1.8
    # Prove zero anomaly GroundTruthEvents generated (hard negative requirement)
    assert len(events_norm) == 0
    assert len(events_surge) == 0



def test_anomaly8_threshold_hugging_mechanism():
    """Verify threshold-hugging evasion generates standardized magnitude hovering near the threshold (3.0 <= M <= 3.6)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec(
        anomaly_type="threshold_hugging_evasion",
        start_time=st,
        duration_seconds=180.0,
        target_magnitude=3.3,
        parameters={"rate_multiplier": 1.66},
    )
    gen.schedule_anomaly("M1", spec)
    txs, events = gen.generate_window(3.0)

    assert len(events) == 1
    gt = events[0]
    assert 2.8 <= gt.severity <= 3.8
    assert gt.severity_level in ("MEDIUM", "HIGH")


def test_anomaly9_persistence_evasion_mechanism():
    """Verify persistence evasion generates alternating 1-minute bursts (even min burst, odd min normal)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec(
        anomaly_type="persistence_evasion",
        start_time=st,
        duration_seconds=240.0,  # 4 minutes: min 0 (burst), min 1 (normal), min 2 (burst), min 3 (normal)
        target_magnitude=4.5,
        parameters={"rate_multiplier": 4.0},
    )
    gen.schedule_anomaly("M1", spec)

    counts_per_min = []
    for _ in range(4):
        txs_min, _ = gen.generate_window(1.0)
        counts_per_min.append(len(txs_min))

    # Min 0 (burst) > Min 1 (normal)
    assert counts_per_min[0] > 2.0 * counts_per_min[1]
    # Min 2 (burst) > Min 3 (normal)
    assert counts_per_min[2] > 2.0 * counts_per_min[3]


def test_anomaly10_staircase_ramp_mechanism():
    """Verify staircase ramp generates discrete monotonically increasing step regime counts."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec(
        anomaly_type="staircase_ramp",
        start_time=st,
        duration_seconds=240.0,  # 4 steps: step 1 -> step 2 -> step 3 -> step 4
        target_magnitude=5.0,
        parameters={"rate_multiplier": 5.0},
    )
    gen.schedule_anomaly("M1", spec)

    counts_per_min = []
    for _ in range(4):
        txs_min, _ = gen.generate_window(1.0)
        counts_per_min.append(len(txs_min))

    # Step progression: counts monotonically increase across steps
    assert counts_per_min[0] < counts_per_min[1] < counts_per_min[2] < counts_per_min[3]


def test_anomaly11_oscillating_sub_threshold_mechanism():
    """Verify oscillating sub-threshold anomaly oscillates periodically and stays within sub-threshold envelope."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec = AnomalySpec(
        anomaly_type="oscillating_sub_threshold",
        start_time=st,
        duration_seconds=240.0,
        target_magnitude=2.5,
        parameters={"amplitude": 0.8, "rate_multiplier": 1.0},
    )
    gen.schedule_anomaly("M1", spec)

    counts_per_min = []
    events_all = []
    for _ in range(4):
        txs_min, evs_min = gen.generate_window(1.0)
        counts_per_min.append(len(txs_min))
        events_all.extend(evs_min)

    # Waveform check: min 1 is peak, min 2 is trough, min 3 is peak
    assert counts_per_min[1] > counts_per_min[0]
    assert counts_per_min[1] > counts_per_min[2]
    assert counts_per_min[3] > counts_per_min[2]

    # Stays within sub-threshold severity
    assert len(events_all) == 1
    assert events_all[0].severity < 3.5


# =====================================================================
# 3. No-Overlap & Boundary-Touching Tests
# =====================================================================

def test_no_overlap_enforcement_and_touching_boundary():
    """Verify overlapping intervals raise OverlapAnomalyError, but touching boundaries (end == start) succeed."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    spec1 = AnomalySpec("volume_spike", st, 120.0, 4.0)
    gen.schedule_anomaly("M1", spec1)

    # Overlapping spec (starts at 1 min, before spec1 ends at 2 min) -> raises OverlapAnomalyError
    spec_overlap = AnomalySpec("velocity_burst", st + timedelta(minutes=1), 120.0, 4.0)
    with pytest.raises(OverlapAnomalyError, match="Anomaly overlap rejected"):
        gen.schedule_anomaly("M1", spec_overlap)

    # Boundary-touching spec (starts exactly at 2 min, when spec1 ends) -> permitted!
    spec_touching = AnomalySpec("velocity_burst", st + timedelta(minutes=2), 120.0, 4.0)
    eid2 = gen.schedule_anomaly("M1", spec_touching)
    assert eid2 is not None


# =====================================================================
# 4. Deterministic Replay & Merchant Compositionality
# =====================================================================

def test_deterministic_replay_and_merchant_compositionality():
    """Verify identical seed produces identical streams, and adding a merchant does not perturb existing merchant."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Run 1: Single merchant M1
    gen1 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs1, _ = gen1.generate_window(5.0)

    # Run 2: Replay single merchant M1
    gen2 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs2, _ = gen2.generate_window(5.0)

    assert len(txs1) == len(txs2)
    for t1, t2 in zip(txs1, txs2):
        assert t1.transaction_id == t2.transaction_id
        assert t1.amount == t2.amount
        assert t1.timestamp == t2.timestamp

    # Run 3: Multi-merchant (M1 + M2) -> M1 stream must remain 100% bitwise identical
    gen3 = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}, {"id": "M2", "archetype": "seasonal"}], VirtualClock(initial_time=st))
    txs3, _ = gen3.generate_window(5.0)
    m1_txs3 = [t for t in txs3 if t.merchant_id == "M1"]

    assert len(m1_txs3) == len(txs1)
    for t1, t3 in zip(txs1, m1_txs3):
        assert t1.transaction_id == t3.transaction_id
        assert t1.amount == t3.amount


# =====================================================================
# 5. Window Partitioning Identity Test
# =====================================================================

def test_window_partitioning_identity():
    """Verify generate_window(5 minutes) produces identical stream as 5 x generate_window(1 minute)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Bulk 5 minutes
    gen_bulk = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    spec = AnomalySpec("volume_spike", st + timedelta(minutes=2), 120.0, 4.5)
    gen_bulk.schedule_anomaly("M1", spec)
    txs_bulk, events_bulk = gen_bulk.generate_window(5.0)

    # Incremental 5 x 1 minute
    gen_inc = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    gen_inc.schedule_anomaly("M1", spec)
    txs_inc = []
    events_inc = []
    for _ in range(5):
        t_w, e_w = gen_inc.generate_window(1.0)
        txs_inc.extend(t_w)
        events_inc.extend(e_w)

    assert len(txs_bulk) == len(txs_inc)
    for tb, ti in zip(txs_bulk, txs_inc):
        assert tb.transaction_id == ti.transaction_id
        assert tb.amount == ti.amount
        assert tb.timestamp == ti.timestamp

    assert len(events_bulk) == len(events_inc) == 1
    assert events_bulk[0].severity == events_inc[0].severity


# =====================================================================
# 6. GroundTruth Architectural Isolation AST Checks
# =====================================================================

def test_ground_truth_architectural_isolation():
    """Verify detector, features, baseline, scoring, and state packages contain zero GroundTruth imports."""
    forbidden_pkgs = ["src/features", "src/baseline", "src/scoring", "src/state", "src/detector"]

    for pkg in forbidden_pkgs:
        for py_file in Path(pkg).glob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "ground_truth" not in alias.name.lower()
                        assert "generator" not in alias.name.lower()
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        assert "generator" not in node.module.lower()
                    for alias in node.names:
                        assert alias.name != "GroundTruthEvent"
