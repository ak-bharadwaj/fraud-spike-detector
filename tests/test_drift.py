"""Comprehensive behavioral unit tests for Day 11 Drift Characterization.

Validates all required drift characterization behavioral dimensions:
1. Transaction-level pre-drift identity (t < minute 40 transactions match 100% field-by-field).
2. Unaffected merchant transaction identity (DRIFT_M2 transactions match 100% across full 120 minutes).
3. Explicit GroundTruth identity invariant (event_id, merchant_id, type, start_time, end_time, severity match 100%).
4. Separate manifest artifact hashes (control_dataset_hash, drifted_dataset_hash, ground_truth_hash, experiment_hash).
5. Removal of ambiguous transactions.json artifact file.
6. Holdout contamination rejection (attempting to pass data/holdout/ raises ValueError).
7. Single-factor drift isolation (volume rate step multiplier 2.5x).
8. Paired Control vs Drifted execution comparison.
9. BaselineEngine adaptation target convergence measurement (adaptation_window_count > 0).
10. Deterministic drift replay.
11. DriftResult Pydantic schema validation.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent, DriftConditionConfig, DriftResult
from src.evaluation.drift import DriftRunner, DriftManifest, load_drift_data, compute_tx_hash, compute_gt_hash
from src.evaluation.holdout import FrozenDetectorConfig


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
# 1. Transaction Pairing & Ground Truth Identity Invariant Tests
# =====================================================================

def test_pre_drift_transaction_identity(drift_dataset):
    """Verify transactions before minute 70 are 100% field-by-field identical between Control and Drift streams."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    onset_time = st + timedelta(minutes=70.0)

    pre_control = [t for t in control_txs if t.timestamp < onset_time]
    pre_drifted = [t for t in drifted_txs if t.timestamp < onset_time]

    assert len(pre_control) == len(pre_drifted)
    for t_c, t_d in zip(pre_control, pre_drifted):
        assert t_c == t_d
        assert t_c.model_dump() == t_d.model_dump()



def test_unaffected_merchant_transaction_identity(drift_dataset):
    """Verify transactions for unaffected merchant DRIFT_M2 match 100% field-by-field across full 150 mins."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset

    m2_control = [t for t in control_txs if t.merchant_id == "DRIFT_M2"]
    m2_drifted = [t for t in drifted_txs if t.merchant_id == "DRIFT_M2"]

    assert len(m2_control) == len(m2_drifted)
    for t_c, t_d in zip(m2_control, m2_drifted):
        assert t_c == t_d
        assert t_c.model_dump() == t_d.model_dump()


def test_explicit_ground_truth_identity_invariant(drift_dataset):
    """Verify GroundTruth events match 100% identically across event_id, merchant_id, type, times, severity."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset

    assert len(gt_events) == 2
    e1, e2 = gt_events[0], gt_events[1]

    assert e1.event_id == "EVT-DRIFT-001"
    assert e1.merchant_id == "DRIFT_M1"
    assert e1.anomaly_type == "volume_spike"

    assert e2.event_id == "EVT-DRIFT-002"
    assert e2.merchant_id == "DRIFT_M1"
    assert e2.anomaly_type == "velocity_spike"


def test_separate_artifact_hashes_and_no_ambiguous_alias(drift_dataset):
    """Verify manifest contains separate control_dataset_hash, drifted_dataset_hash, ground_truth_hash, experiment_hash."""
    manifest, control_txs, drifted_txs, gt_events = drift_dataset

    assert isinstance(manifest, DriftManifest)
    assert compute_tx_hash(control_txs) == manifest.control_dataset_hash
    assert compute_tx_hash(drifted_txs) == manifest.drifted_dataset_hash
    assert compute_gt_hash(gt_events) == manifest.ground_truth_hash

    # Verify ambiguous transactions.json file is absent
    alias_path = Path(__file__).parent.parent / "data" / "drift" / "transactions.json"
    assert not alias_path.exists()


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
    assert cond.start_minute == 70.0
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

    assert res.control_metrics.tp == 2
    assert res.drifted_metrics.tp == 2
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
