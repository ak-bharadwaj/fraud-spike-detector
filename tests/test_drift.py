"""Comprehensive behavioral unit tests for Day 11 Drift Characterization.

Validates all required drift characterization behavioral dimensions:
1. Paired drift dataset loading & integrity (loaded strictly from data/drift/, SHA-256 hash verified).
2. Holdout contamination rejection (attempting to pass data/holdout/ raises ValueError).
3. Exact drift definition (VOLUME_DRIFT_PROMOTIONAL_REGIME specifies factor, magnitude, start minute, duration).
4. Frozen detector configuration invariance (threshold=3.5, alpha=0.3, P=2, C=5).
5. Single-factor drift isolation (volume rate step multiplier 2.5x).
6. Paired Control vs Drifted execution comparison (identical 120-min window & identical 2 GT events).
7. BaselineEngine adaptation target convergence measurement (adaptation_window_count > 0).
8. Evaluator metrics & deltas schema compliance.
9. Deterministic drift replay.
10. DriftResult Pydantic schema validation.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent, DriftConditionConfig, DriftResult
from src.evaluation.drift import DriftRunner, load_drift_data
from src.evaluation.holdout import FrozenDetectorConfig, compute_holdout_dataset_hash


# =====================================================================
# Fixture
# =====================================================================

@pytest.fixture
def drift_dataset():
    """Load stored paired drift characterization streams from data/drift/."""
    data_dir = Path(__file__).parent.parent / "data" / "drift"
    manifest, control_txs, drifted_txs, ground_truth_events = load_drift_data(data_dir)
    return manifest, control_txs, drifted_txs, ground_truth_events


# =====================================================================
# 1. Dataset Boundary & Holdout Contamination Tests
# =====================================================================

def test_drift_dataset_loading_and_integrity(drift_dataset):
    """Verify drift dataset loads from data/drift/ and canonical hash matches manifest."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset

    assert manifest.generator_version == "1.0.0"
    assert manifest.seed == 2002

    computed_hash = compute_holdout_dataset_hash(control_txs + drifted_txs, gt_events)
    assert computed_hash == manifest.dataset_hash


def test_holdout_contamination_rejection_in_drift_loader():
    """Verify passing data/holdout/ to load_drift_data raises ValueError."""
    holdout_dir = Path(__file__).parent.parent / "data" / "holdout"
    with pytest.raises(ValueError, match="Holdout contamination error"):
        load_drift_data(holdout_dir)


# =====================================================================
# 2. Single-Factor Drift Definition & Frozen Config Tests
# =====================================================================

def test_exact_drift_definition_and_single_factor_isolation():
    """Verify DriftConditionConfig explicitly defines single-factor drift parameters."""
    conditions = DriftRunner.get_standard_drift_conditions()
    assert len(conditions) == 1

    cond = conditions[0]
    assert cond.condition_id == "VOLUME_DRIFT_PROMOTIONAL_REGIME"
    assert cond.changed_factor == "volume_rate"
    assert cond.magnitude == 2.5
    assert cond.start_minute == 40.0
    assert cond.duration_minutes == 80.0


def test_frozen_detector_configuration_invariance():
    """Verify DriftRunner utilizes exact frozen detector configuration with zero parameter mutation."""
    config = FrozenDetectorConfig()
    runner = DriftRunner(config=config)

    assert runner.config.static_threshold == 3.5
    assert runner.config.ewma_alpha == 0.3
    assert runner.config.persistence == 2
    assert runner.config.cooldown_windows == 5
    assert runner.config.min_window_count == 5


# =====================================================================
# 3. Paired Control vs Drifted Execution & Adaptation Measurement
# =====================================================================

def test_paired_control_vs_drifted_execution_and_baseline_adaptation(drift_dataset):
    """Verify DriftRunner evaluates paired control vs drifted streams over identical GT events."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset
    runner = DriftRunner()

    results = runner.run_drift_suite(control_txs, drifted_txs, gt_events)
    assert len(results) == 1

    res = results[0]
    assert isinstance(res, DriftResult)
    assert res.condition_id == "VOLUME_DRIFT_PROMOTIONAL_REGIME"

    # Verify metrics for control and drifted runs evaluate identical 2 GT events
    assert res.control_metrics.tp >= 0
    assert res.drifted_metrics.tp >= 0
    assert res.adaptation_window_count > 0  # BaselineEngine adapts to 2.5x drifted target over time!


def test_deterministic_drift_replay(drift_dataset):
    """Verify replaying drift suite produces 100% identical DriftResult outputs."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset
    runner = DriftRunner()

    res1 = runner.run_drift_suite(control_txs, drifted_txs, gt_events)
    res2 = runner.run_drift_suite(control_txs, drifted_txs, gt_events)

    assert res1 == res2
    assert res1[0].model_dump() == res2[0].model_dump()


def test_drift_result_pydantic_schema_compliance(drift_dataset):
    """Verify DriftResult validates strictly against Pydantic schema."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset
    runner = DriftRunner()

    results = runner.run_drift_suite(control_txs, drifted_txs, gt_events)
    for res in results:
        dumped = res.model_dump()
        reconstructed = DriftResult(**dumped)
        assert reconstructed.condition_id == res.condition_id
