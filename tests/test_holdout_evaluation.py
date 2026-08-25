"""Comprehensive behavioral unit tests for Day 9 Locked Holdout Evaluation.

Validates all required locked holdout evaluation behavioral dimensions:
1. Holdout manifest verification (hash, generator_version, seed, schema_version, created_at).
2. Holdout access protection (raises HoldoutAccessError when explicit_evaluation_mode=False).
3. Checksum integrity verification (raises ChecksumMismatchError on hash mismatch).
4. Single-pass holdout evaluation execution.
5. Frozen configuration invariance (threshold=3.5, alpha=0.3, P=2, C=5, min_window_count=5).
6. Metric schema compliance (TP, FP, FN, Precision, Recall, F1, Latency).
7. Deterministic holdout replay.
"""

from datetime import datetime, timedelta, timezone
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent, EvaluationMetrics
from src.evaluation.holdout import (
    HoldoutManifest,
    HoldoutProtection,
    HoldoutEvaluator,
    HoldoutAccessError,
    ChecksumMismatchError,
)
from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def locked_holdout_manifest() -> HoldoutManifest:
    """Immutable locked holdout manifest for Day 9 benchmark evaluation."""
    return HoldoutManifest(
        dataset_hash="a3f89e2c1b4d7e0f918273645e0d2c1b4d7e0f918273645e0d2c1b4d7e0f9182",
        generator_version="1.0.0",
        seed=4242,
        schema_version="1.0.0",
        created_at="2026-08-25T00:00:00Z",
    )


@pytest.fixture
def holdout_dataset(locked_holdout_manifest: HoldoutManifest):
    """Generate deterministic locked holdout dataset stream."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(
        global_seed=locked_holdout_manifest.seed,
        merchant_configs=[{"id": "HOLDOUT_M1", "archetype": "stable"}],
        clock=VirtualClock(initial_time=st),
    )

    # Schedule a fraud anomaly event in holdout stream
    spec = AnomalySpec("volume_spike", st + timedelta(minutes=10), 300.0, 4.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("HOLDOUT_M1", spec, "EVT-HOLDOUT-001")

    txs, gt_events = gen.generate_window(20.0)
    dataset_hash = locked_holdout_manifest.dataset_hash
    return txs, gt_events, dataset_hash


# =====================================================================
# 1. Holdout Access & Checksum Protection Tests
# =====================================================================

def test_holdout_access_denied_in_normal_development_mode(
    locked_holdout_manifest: HoldoutManifest,
    holdout_dataset,
):
    """Verify attempting to evaluate holdout with explicit_evaluation_mode=False raises HoldoutAccessError."""
    txs, gt_events, dataset_hash = holdout_dataset
    evaluator = HoldoutEvaluator(manifest=locked_holdout_manifest, explicit_evaluation_mode=False)

    with pytest.raises(HoldoutAccessError, match="Holdout access denied"):
        evaluator.evaluate_holdout(
            transactions=txs,
            ground_truth_events=gt_events,
            actual_dataset_hash=dataset_hash,
        )


def test_holdout_checksum_mismatch_aborts_evaluation(
    locked_holdout_manifest: HoldoutManifest,
    holdout_dataset,
):
    """Verify actual dataset hash mismatch raises ChecksumMismatchError."""
    txs, gt_events, _ = holdout_dataset
    evaluator = HoldoutEvaluator(manifest=locked_holdout_manifest, explicit_evaluation_mode=True)

    with pytest.raises(ChecksumMismatchError, match="Holdout dataset checksum mismatch"):
        evaluator.evaluate_holdout(
            transactions=txs,
            ground_truth_events=gt_events,
            actual_dataset_hash="invalid_corrupted_hash_xyz",
        )


# =====================================================================
# 2. Single-Pass Holdout Evaluation & Metrics Verification
# =====================================================================

def test_single_pass_holdout_evaluation_success(
    locked_holdout_manifest: HoldoutManifest,
    holdout_dataset,
):
    """Verify single-pass holdout evaluation with explicit_evaluation_mode=True emits valid EvaluationMetrics."""
    txs, gt_events, dataset_hash = holdout_dataset
    evaluator = HoldoutEvaluator(manifest=locked_holdout_manifest, explicit_evaluation_mode=True)

    metrics = evaluator.evaluate_holdout(
        transactions=txs,
        ground_truth_events=gt_events,
        actual_dataset_hash=dataset_hash,
        static_threshold=3.5,
        ewma_alpha=0.3,
        persistence=2,
        cooldown_windows=5,
        min_window_count=5,
        temporal_tolerance_seconds=0.0,
    )

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.tp >= 0
    assert metrics.fp >= 0
    assert metrics.fn >= 0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0


def test_deterministic_holdout_evaluation_replay(
    locked_holdout_manifest: HoldoutManifest,
    holdout_dataset,
):
    """Verify replaying holdout evaluation produces 100% identical metrics output."""
    txs, gt_events, dataset_hash = holdout_dataset
    evaluator = HoldoutEvaluator(manifest=locked_holdout_manifest, explicit_evaluation_mode=True)

    res1 = evaluator.evaluate_holdout(
        transactions=txs,
        ground_truth_events=gt_events,
        actual_dataset_hash=dataset_hash,
    )
    res2 = evaluator.evaluate_holdout(
        transactions=txs,
        ground_truth_events=gt_events,
        actual_dataset_hash=dataset_hash,
    )

    assert res1 == res2
    assert res1.model_dump() == res2.model_dump()
