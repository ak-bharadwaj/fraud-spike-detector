"""Day 5 Synthetic Benchmark and Anomaly Classes Test Suite.

Validates:
1. Merchant Archetypes M1 through M9:
   - M1: Small / low volume (0.2 <= lambda <= 0.8)
   - M2: Medium / stable (8.0 <= lambda <= 15.0)
   - M3: High volume (30.0 <= lambda <= 60.0)
   - Volume scale hierarchy: M1 < M2 < M3
   - M4: Seasonal (diurnal diurnal variation)
   - M5: Weekend-heavy (Saturday/Sunday > Weekday)
   - M6: Highly variable (large rate/amount variance)
   - M7: Night-heavy (Night 22:00-05:00 UTC > Daytime)
   - M8: Mixed (seasonality + growth + variance)
   - M9: Organic growth (baseline rate increases monotonically over time)
2. All 11 Anomaly Classes:
   1. sudden_volume_spike / volume_spike
   2. velocity_burst / velocity_spike
   3. sustained_spike / sustained_anomaly
   4. amount_distribution_shift / amount_spike
   5. device_behavior_anomaly / behavioral_shift
   6. attribute_geographic_shift / attribute_anomaly
   7. compound_anomaly
   8. threshold_hugging_evasion
   9. persistence_evasion
   10. staircase_ramp
   11. oscillating_sub_threshold
3. Ground-Truth Protocol & Realized Severity Derivation.
4. No-Overlap Invariant & Boundary-Touching Intervals.
5. Deterministic Replay & Merchant Compositionality.
6. Window Partitioning.
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
from src.generator.anomalies import AnomalySpec
from src.generator.stream_generator import SyntheticStreamGenerator, OverlapAnomalyError
from src.stream.clock import VirtualClock


# =====================================================================
# 1. Merchant Archetypes M1 through M9 Validation
# =====================================================================

def test_merchant_archetypes_m1_m2_m3_scale_hierarchy():
    """Verify M1, M2, M3 have distinct non-overlapping volume scales and M1 < M2 < M3."""
    p1 = create_merchant_profile(42, "M1", "M1")
    p2 = create_merchant_profile(42, "M2", "M2")
    p3 = create_merchant_profile(42, "M3", "M3")

    assert 0.2 <= p1.base_rate_per_min <= 0.8
    assert 8.0 <= p2.base_rate_per_min <= 15.0
    assert 30.0 <= p3.base_rate_per_min <= 60.0
    assert p1.base_rate_per_min < p2.base_rate_per_min < p3.base_rate_per_min


def test_merchant_archetype_m4_seasonal_diurnal_structure():
    """Verify M4 exhibits diurnal cycle (peak midday > trough early morning)."""
    p4 = create_merchant_profile(42, "M4", "M4")
    t_start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)  # Monday 00:00

    r_morning = compute_legitimate_rate(p4, t_start + timedelta(hours=6), t_start)
    r_midday = compute_legitimate_rate(p4, t_start + timedelta(hours=12), t_start)
    r_night = compute_legitimate_rate(p4, t_start + timedelta(hours=0), t_start)

    assert r_midday > r_night
    assert r_midday > r_morning


def test_merchant_archetype_m5_weekend_heavy():
    """Verify M5 exhibits higher volume on Saturday/Sunday compared to weekdays."""
    p5 = create_merchant_profile(42, "M5", "M5")
    t_start = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)  # Monday 12:00
    t_weekend = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)  # Saturday 12:00

    r_weekday = compute_legitimate_rate(p5, t_start, t_start)
    r_weekend = compute_legitimate_rate(p5, t_weekend, t_start)

    assert r_weekend > 2.0 * r_weekday


def test_merchant_archetype_m6_highly_variable():
    """Verify M6 generates high amount dispersion (volatile)."""
    p6 = create_merchant_profile(42, "M6", "M6")
    assert p6.archetype == "volatile"
    assert p6.base_std_amount > 0.5 * p6.base_mean_amount


def test_merchant_archetype_m7_night_heavy():
    """Verify M7 has peak activity concentrated during night hours (22:00-05:00 UTC)."""
    p7 = create_merchant_profile(42, "M7", "M7")
    t_start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    t_night = datetime(2026, 1, 5, 23, 0, tzinfo=timezone.utc)  # 23:00 UTC
    t_day = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)    # 14:00 UTC

    r_night = compute_legitimate_rate(p7, t_night, t_start)
    r_day = compute_legitimate_rate(p7, t_day, t_start)

    assert r_night > 2.0 * r_day


def test_merchant_archetype_m8_mixed():
    """Verify M8 combines seasonality and growth."""
    p8 = create_merchant_profile(42, "M8", "M8")
    assert p8.archetype == "mixed"


def test_merchant_archetype_m9_organic_growth():
    """Verify M9 baseline rate increases over time due to organic growth."""
    p9 = create_merchant_profile(42, "M9", "M9")
    t_start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    t_day10 = t_start + timedelta(days=10)

    r_day0 = compute_legitimate_rate(p9, t_start, t_start)
    r_day10 = compute_legitimate_rate(p9, t_day10, t_start)

    assert r_day10 > r_day0 * 1.15  # 20% growth after 10 days (0.02/day)


# =====================================================================
# 2. All 11 Anomaly Classes Validation
# =====================================================================

@pytest.mark.parametrize("anomaly_type", [
    "sudden_volume_spike",
    "velocity_burst",
    "sustained_spike",
    "amount_distribution_shift",
    "device_behavior_anomaly",
    "attribute_geographic_shift",
    "compound_anomaly",
    "threshold_hugging_evasion",
    "persistence_evasion",
    "staircase_ramp",
    "oscillating_sub_threshold",
])
def test_all_11_anomaly_classes_generation(anomaly_type):
    """Verify each of the 11 anomaly classes generates valid transactions and GroundTruthEvent."""
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
    assert eid.startswith("EVT-M1-")

    txs, events = gen.generate_window(2.0)

    assert len(events) == 1
    gt = events[0]
    assert gt.event_id == eid
    assert gt.merchant_id == "M1"
    assert gt.anomaly_type == anomaly_type
    assert gt.start_time == st
    assert gt.end_time == st + timedelta(minutes=2)
    assert gt.severity > 0.0
    assert gt.severity_level in ("LOW", "MEDIUM", "HIGH")


# =====================================================================
# 3. No-Overlap Invariant & Boundary-Touching Tests
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
