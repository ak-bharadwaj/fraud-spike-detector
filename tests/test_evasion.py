"""Comprehensive test suite for Day 12 Adversarial Evasion Characterization protocol."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent, EvasionResult
from src.evaluation.holdout import FrozenDetectorConfig
from src.evaluation.evasion import (
    EvasionRunner,
    EvasionManifest,
    load_evasion_data,
    compute_tx_hash,
    compute_gt_hash,
)


@pytest.fixture
def evasion_dataset():
    data_dir = Path(__file__).parent.parent / "data" / "evasion"
    return load_evasion_data(data_dir)


# =====================================================================
# 1. Artifact Integrity & Holdout Firewall Tests
# =====================================================================

def test_evasion_artifact_integrity_and_hash_verification(evasion_dataset):
    """Verify manifest integrity, transaction checksums, and combined experiment hash."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset

    assert isinstance(manifest, EvasionManifest)
    assert compute_tx_hash(control_txs) == manifest.control_dataset_hash
    assert compute_tx_hash(evasion_txs) == manifest.evasion_dataset_hash
    assert compute_gt_hash(gt_events) == manifest.ground_truth_hash


def test_holdout_contamination_rejection_in_evasion_loader():
    """Verify passing data/holdout/ to load_evasion_data raises ValueError."""
    holdout_dir = Path(__file__).parent.parent / "data" / "holdout"
    with pytest.raises(ValueError, match="Holdout contamination error"):
        load_evasion_data(holdout_dir)


# =====================================================================
# 2. Single-Factor Evasion Definition & Pairing Tests
# =====================================================================

def test_exact_evasion_definition_and_single_factor_isolation():
    """Verify EvasionConditionConfig explicitly defines single-factor evasion parameters."""
    conditions = EvasionRunner.get_standard_evasion_conditions()
    assert len(conditions) == 1

    cond = conditions[0]
    assert cond.condition_id == "SLOW_BURN_SPLIT_EVASION"
    assert cond.evasion_strategy == "temporal_dilution"
    assert cond.changed_factor == "rate_multiplier_and_duration"
    assert cond.magnitude == 1.3
    assert cond.start_minute == 50.0
    assert cond.duration_minutes == 30.0


def test_control_evasion_pairing_and_pre_onset_identity(evasion_dataset):
    """Verify transactions before minute 50 are 100% field-by-field identical between Control and Evasion streams."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    onset_time = st + timedelta(minutes=50.0)

    pre_control = [t for t in control_txs if t.timestamp < onset_time]
    pre_evasion = [t for t in evasion_txs if t.timestamp < onset_time]

    assert len(pre_control) == len(pre_evasion)
    for t_c, t_e in zip(pre_control, pre_evasion):
        assert t_c == t_e
        assert t_c.model_dump() == t_e.model_dump()


def test_unaffected_merchant_transaction_identity(evasion_dataset):
    """Verify transactions for unaffected merchant EVASION_M2 match 100% field-by-field across full 120 mins."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset

    m2_control = [t for t in control_txs if t.merchant_id == "EVASION_M2"]
    m2_evasion = [t for t in evasion_txs if t.merchant_id == "EVASION_M2"]

    assert len(m2_control) == len(m2_evasion)
    for t_c, t_e in zip(m2_control, m2_evasion):
        assert t_c == t_e
        assert t_c.model_dump() == t_e.model_dump()


def test_explicit_ground_truth_identity_invariant(evasion_dataset):
    """Verify GroundTruth events match 100% identically across event_id, merchant_id, type, times, severity."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset

    assert len(gt_events) == 1
    e1 = gt_events[0]

    assert e1.event_id == "EVT-EVASION-001"
    assert e1.merchant_id == "EVASION_M1"
    assert e1.anomaly_type == "volume_spike"
    assert e1.start_time == datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)


# =====================================================================
# 3. Frozen Configuration & Immutability Tests
# =====================================================================

def test_frozen_detector_configuration_invariance():
    """Verify EvasionRunner utilizes exact frozen detector configuration with zero parameter mutation."""
    config = FrozenDetectorConfig()
    runner = EvasionRunner(config=config)

    assert runner.config.static_threshold == 3.5
    assert runner.config.ewma_alpha == 0.3
    assert runner.config.persistence == 2
    assert runner.config.cooldown_windows == 5
    assert runner.config.min_window_count == 5
    assert runner.config.temporal_tolerance_seconds == 0.0


def test_detector_immutability_during_evasion(evasion_dataset):
    """Verify detector configuration and parameters remain strictly unchanged after running evasion suite."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset
    config = FrozenDetectorConfig()
    runner = EvasionRunner(config=config)

    runner.run_evasion_suite(control_txs, evasion_txs, gt_events)

    assert runner.config.static_threshold == 3.5
    assert runner.config.ewma_alpha == 0.3
    assert runner.config.persistence == 2
    assert runner.config.cooldown_windows == 5



# =====================================================================
# 4. Evaluator Metrics, Evasion Degradation & Schema Compliance Tests
# =====================================================================

def test_evaluator_metrics_control_vs_evasion_execution(evasion_dataset):
    """Verify EvasionRunner measures detection degradation (Control TP=1, Evasion FN=1, evasion_success_rate=1.0)."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset
    runner = EvasionRunner()

    results = runner.run_evasion_suite(control_txs, evasion_txs, gt_events)
    assert len(results) == 1

    res = results[0]
    assert isinstance(res, EvasionResult)
    assert res.condition_id == "SLOW_BURN_SPLIT_EVASION"

    assert res.control_metrics.tp == 1
    assert res.control_metrics.recall == 1.0
    assert res.control_metrics.f1_score == 0.5

    assert res.evasion_metrics.tp == 0
    assert res.evasion_metrics.fn == 1
    assert res.evasion_metrics.f1_score == 0.0

    assert res.delta_f1 == -0.5
    assert res.delta_recall == -1.0
    assert res.detection_degraded is True
    assert res.evasion_success_rate == 1.0



def test_deterministic_evasion_replay(evasion_dataset):
    """Verify replaying evasion suite produces 100% identical metrics across multiple executions."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset
    runner = EvasionRunner()

    res1 = runner.run_evasion_suite(control_txs, evasion_txs, gt_events)[0]
    res2 = runner.run_evasion_suite(control_txs, evasion_txs, gt_events)[0]

    assert res1.model_dump() == res2.model_dump()


def test_evasion_result_pydantic_schema_compliance(evasion_dataset):
    """Verify EvasionResult complies strictly with Pydantic schema contracts."""
    manifest, control_txs, evasion_txs, gt_events = evasion_dataset
    runner = EvasionRunner()

    res = runner.run_evasion_suite(control_txs, evasion_txs, gt_events)[0]

    assert isinstance(res.condition_id, str)
    assert isinstance(res.delta_f1, float)
    assert isinstance(res.detection_degraded, bool)
    assert isinstance(res.evasion_success_rate, float)
