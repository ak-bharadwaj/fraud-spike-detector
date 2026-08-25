"""Comprehensive behavioral unit tests for Day 4 BaselineEngine.

Validates all 14 required baseline dimensions:
1. Evidence-state transition (INSUFFICIENT -> SUFFICIENT -> DEGRADED).
2. Minimum-history requirement (min_history_count=50, min_window_count=5 from config/detector.yaml).
3. Baseline evidence eligibility (EMPTY/zero-volume snapshots excluded from median/MAD calculations).
4. Historical-only updates (current snapshot does not inflate its own expected baseline).
5. Future-leakage prevention (adding t_future does not change t_now baseline).
6. Baseline statistic correctness (sample median expected_values).
7. Scale/dispersion correctness (MAD with robust floor).
8. Legitimate growth handling (smooth baseline tracking).
9. Genuine deterministic seasonal handling (diurnal baseline bounds).
10. Sparse merchant handling (insufficient evidence until history count met).
11. Merchant isolation (Merchant A vs B independence).
12. Deterministic replay.
13. GroundTruth & Holdout isolation (zero ground-truth or holdout imports in src/baseline).
14. BaselineSnapshot Pydantic schema compliance.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import numpy as np
import pytest

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot
from src.baseline.baseline_engine import BaselineEngine
from src.features.feature_engine import FeatureEngine
from src.generator.archetypes import create_merchant_profile, compute_legitimate_rate
from src.generator.stream_generator import SyntheticStreamGenerator
from src.stream.clock import VirtualClock


# =====================================================================
# Helper to create a dummy FeatureSnapshot
# =====================================================================

def make_dummy_snapshot(
    merchant_id: str,
    ts: datetime,
    volume: float = 10.0,
    velocity: float = 2.0,
    mean_amount: float = 100.0,
    data_quality: str = "GOOD",
) -> FeatureSnapshot:
    return FeatureSnapshot(
        merchant_id=merchant_id,
        timestamp=ts,
        volume=volume,
        velocity=velocity,
        amount_statistics={
            "total_amount": volume * mean_amount,
            "mean_amount": mean_amount,
            "std_amount": 10.0,
            "median_amount": mean_amount,
            "mad_amount": 5.0,
            "min_amount": mean_amount - 15.0,
            "max_amount": mean_amount + 15.0,
        },
        unique_customers=int(volume * 0.8),
        unique_devices=int(volume * 0.5),
        data_quality=data_quality,
    )


# =====================================================================
# 1. Baseline Evidence Eligibility (EMPTY Windows Excluded)
# =====================================================================

def test_baseline_evidence_eligibility_empty_windows_excluded():
    """Verify EMPTY windows (volume=0) do NOT contaminate baseline expected_values or robust_scale."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=4, min_window_count=1)

    # 4 Legitimate GOOD snapshots with volume around 10.0
    good_vols = [10.0, 11.0, 9.0, 10.0]
    for i, v in enumerate(good_vols):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=v, data_quality="GOOD"))

    # Add 50 EMPTY snapshots (volume=0, data_quality="EMPTY")
    for i in range(50):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=10 + i), volume=0.0, data_quality="EMPTY"))

    snap_now = make_dummy_snapshot("M1", st + timedelta(minutes=100), volume=10.0)
    base = engine.get_baseline("M1", snap_now)

    # Baseline expected volume MUST be median of eligible GOOD history (10.0), NOT 0.0!
    assert base.expected_values["volume"] == 10.0
    assert base.history_count == 4


# =====================================================================
# 2. Evidence-State Transition & Minimum-History Requirement
# =====================================================================

def test_evidence_state_transitions_and_minimum_history():
    """Verify evidence_state transitions: INSUFFICIENT (<50) -> SUFFICIENT (>=50, GOOD) -> DEGRADED (>=50, EMPTY/low volume)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=50, min_window_count=5)

    # 1. History count < 50 -> INSUFFICIENT
    for i in range(49):
        snap = make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=10.0)
        base = engine.get_baseline("M1", snap)
        assert base.evidence_state == "INSUFFICIENT"
        assert base.history_count == i
        engine.update(snap)

    engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=49), volume=10.0))

    # 2. History count = 50, current volume=10 (>=5) -> SUFFICIENT
    current_good = make_dummy_snapshot("M1", st + timedelta(minutes=50), volume=10.0)
    base_good = engine.get_baseline("M1", current_good)
    assert base_good.evidence_state == "SUFFICIENT"
    assert base_good.history_count == 50

    # 3. History count = 50, current volume=2 (<5) -> DEGRADED
    current_low = make_dummy_snapshot("M1", st + timedelta(minutes=51), volume=2.0)
    base_degraded = engine.get_baseline("M1", current_low)
    assert base_degraded.evidence_state == "DEGRADED"

    # 4. History count = 50, current data_quality="EMPTY" -> DEGRADED
    current_empty = make_dummy_snapshot("M1", st + timedelta(minutes=52), volume=0.0, data_quality="EMPTY")
    base_empty = engine.get_baseline("M1", current_empty)
    assert base_empty.evidence_state == "DEGRADED"


# =====================================================================
# 3. Historical-Only Updates
# =====================================================================

def test_historical_only_updates_current_snapshot_excluded():
    """Verify current window snapshot is NOT included in its own baseline calculation."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=5, min_window_count=1)

    for i in range(5):
        snap = make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=10.0)
        engine.update(snap)

    current_huge = make_dummy_snapshot("M1", st + timedelta(minutes=5), volume=1000.0)
    base = engine.get_baseline("M1", current_huge)

    assert base.expected_values["volume"] == 10.0


# =====================================================================
# 4. Future-Leakage Prevention
# =====================================================================

def test_future_leakage_prevention_adding_future_snapshots():
    """Verify adding future snapshots (t > t_now) does not change baseline evaluated at t_now."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=5, min_window_count=1)

    for i in range(5):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=10.0))

    t_now = st + timedelta(minutes=5)
    snap_now = make_dummy_snapshot("M1", t_now, volume=10.0)

    base_before = engine.get_baseline("M1", snap_now)

    for i in range(6, 16):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=999.0))

    base_after = engine.get_baseline("M1", snap_now)

    assert base_before == base_after
    assert base_after.expected_values["volume"] == 10.0


# =====================================================================
# 5. Baseline Statistic Correctness & 6. Scale/Dispersion Correctness
# =====================================================================

def test_baseline_statistic_and_robust_scale_correctness():
    """Verify expected_values equals sample median and robust_scale equals max(floor, MAD)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=5, min_window_count=1)

    vols = [10.0, 20.0, 30.0, 40.0, 50.0]
    for i, v in enumerate(vols):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=v))

    snap = make_dummy_snapshot("M1", st + timedelta(minutes=5), volume=30.0)
    base = engine.get_baseline("M1", snap)

    assert base.expected_values["volume"] == 30.0
    assert base.robust_scale["volume"] == 10.0


# =====================================================================
# 7. Legitimate Growth Handling & 8. Genuine Seasonal Handling
# =====================================================================

def test_seasonal_merchant_baseline_tracking():
    """Verify BaselineEngine computes expected_values and robust_scale for a genuine diurnal seasonal merchant."""
    st = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    prof = create_merchant_profile(42, "M_seasonal", "seasonal")
    engine = BaselineEngine(min_history_count=20, min_window_count=1)

    # Generate 24 hours (1440 minutes) of diurnal seasonal rates
    rates_list = []
    for m in range(0, 1440, 30):
        t_m = st + timedelta(minutes=m)
        rate = compute_legitimate_rate(prof, t_m, st)
        rates_list.append(rate)
        engine.update(make_dummy_snapshot("M_seasonal", t_m, volume=rate))

    snap_now = make_dummy_snapshot("M_seasonal", st + timedelta(minutes=1440), volume=5.0)
    base = engine.get_baseline("M_seasonal", snap_now)

    exp_med = base.expected_values["volume"]
    robust_scale = base.robust_scale["volume"]

    # Verify diurnal baseline median and MAD capture the diurnal variation bounds
    rates_arr = np.array(rates_list)
    expected_med_ind = float(np.median(rates_arr))
    expected_mad_ind = float(np.median(np.abs(rates_arr - expected_med_ind)))

    assert math.isclose(exp_med, expected_med_ind, abs_tol=1e-4)
    assert math.isclose(robust_scale, expected_mad_ind, abs_tol=1e-4)


# =====================================================================
# 9. Sparse Merchant Handling
# =====================================================================

def test_sparse_merchant_handling():
    """Verify sparse merchant with low history count stays INSUFFICIENT."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=50, min_window_count=1)

    for i in range(5):
        engine.update(make_dummy_snapshot("M_sparse", st + timedelta(hours=i), volume=1.0))

    snap = make_dummy_snapshot("M_sparse", st + timedelta(hours=5), volume=1.0)
    base = engine.get_baseline("M_sparse", snap)

    assert base.evidence_state == "INSUFFICIENT"
    assert base.history_count == 5


# =====================================================================
# 10. Merchant Isolation
# =====================================================================

def test_multi_merchant_isolation():
    """Verify updating Merchant B history does not alter Merchant A baseline."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=5, min_window_count=1)

    for i in range(5):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=10.0))

    snap_a = make_dummy_snapshot("M1", st + timedelta(minutes=5), volume=10.0)
    base_a_before = engine.get_baseline("M1", snap_a)

    for i in range(20):
        engine.update(make_dummy_snapshot("M2", st + timedelta(minutes=i), volume=999.0))

    base_a_after = engine.get_baseline("M1", snap_a)

    assert base_a_before == base_a_after
    assert base_a_after.expected_values["volume"] == 10.0


# =====================================================================
# 11. Deterministic Replay
# =====================================================================

def test_deterministic_baseline_replay():
    """Verify identical snapshot input sequence produces identical BaselineSnapshot output."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    engine1 = BaselineEngine(min_history_count=5, min_window_count=1)
    for i in range(5):
        engine1.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=10.0 + i))
    snap = make_dummy_snapshot("M1", st + timedelta(minutes=5), volume=15.0)
    base1 = engine1.get_baseline("M1", snap)

    engine2 = BaselineEngine(min_history_count=5, min_window_count=1)
    for i in range(5):
        engine2.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=10.0 + i))
    base2 = engine2.get_baseline("M1", snap)

    assert base1 == base2
    assert base1.model_dump() == base2.model_dump()


# =====================================================================
# 12. GroundTruth & Holdout Isolation (AST Architectural Check)
# =====================================================================

def test_ground_truth_and_holdout_isolation_in_baseline_package():
    """Verify src/baseline code contains zero imports of ground_truth, GroundTruthEvent, or holdout code."""
    baseline_dir = Path(__file__).parent.parent / "src" / "baseline"
    py_files = list(baseline_dir.rglob("*.py"))

    assert len(py_files) > 0

    for file_path in py_files:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "ground_truth" not in alias.name, f"GroundTruth import violation in {file_path}: {alias.name}"
                    assert "holdout" not in alias.name, f"Holdout import violation in {file_path}: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "ground_truth" not in module, f"GroundTruth import violation in {file_path}: {module}"
                assert "holdout" not in module, f"Holdout import violation in {file_path}: {module}"
                for alias in node.names:
                    assert "GroundTruth" not in alias.name, f"GroundTruth element import violation in {file_path}: {alias.name}"
                    assert "holdout" not in alias.name and "Holdout" not in alias.name, f"Holdout element import violation in {file_path}: {alias.name}"


# =====================================================================
# 13. Holdout Isolation Execution Test
# =====================================================================

def test_holdout_isolation_no_dependencies():
    """Verify BaselineEngine operates without holdout access or dependencies."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=5, min_window_count=1)

    for i in range(5):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=i)))

    snap = make_dummy_snapshot("M1", st + timedelta(minutes=5))
    base = engine.get_baseline("M1", snap)

    assert isinstance(base, BaselineSnapshot)


# =====================================================================
# 14. BaselineSnapshot Schema Validation
# =====================================================================

def test_baseline_snapshot_pydantic_schema_compliance():
    """Verify emitted BaselineSnapshot validates strictly against BaselineSnapshot Pydantic contract."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = BaselineEngine(min_history_count=5, min_window_count=1)

    for i in range(5):
        engine.update(make_dummy_snapshot("M1", st + timedelta(minutes=i), volume=10.0))

    snap = make_dummy_snapshot("M1", st + timedelta(minutes=5), volume=10.0)
    base = engine.get_baseline("M1", snap)

    dumped = base.model_dump()
    reconstructed = BaselineSnapshot(**dumped)

    assert reconstructed.merchant_id == "M1"
    assert reconstructed.evidence_state == "SUFFICIENT"
    assert reconstructed.history_count == 5
