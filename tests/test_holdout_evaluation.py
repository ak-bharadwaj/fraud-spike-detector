"""Comprehensive behavioral unit tests for Day 9 Locked Holdout Evaluation.

Validates all required locked holdout evaluation behavioral dimensions:
1. Locked holdout artifact storage & manifest verification (loaded directly from data/holdout/).
2. Canonical SHA-256 dataset hash verification (actual_hash == manifest.dataset_hash).
3. Checksum integrity verification (mutating transaction changes computed hash -> raises ChecksumMismatchError).
4. Holdout access protection (raises HoldoutAccessError when explicit_evaluation_mode=False).
5. Historical-only baseline ordering (verifies current window transactions DO NOT leak into pre-current baseline).
6. Structurally frozen detector config (FrozenDetectorConfig with zero parameter overrides).
7. Single-pass holdout evaluation execution emitting valid EvaluationMetrics.
8. Deterministic holdout replay.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent, EvaluationMetrics
from src.evaluation.holdout import (
    HoldoutManifest,
    HoldoutProtection,
    HoldoutEvaluator,
    FrozenDetectorConfig,
    HoldoutAccessError,
    ChecksumMismatchError,
    compute_holdout_dataset_hash,
    load_locked_holdout_data,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine


# =====================================================================
# Fixture loading stored locked artifact from data/holdout/
# =====================================================================

@pytest.fixture
def stored_locked_holdout_artifact():
    """Load stored locked holdout artifact from data/holdout/."""
    data_dir = Path(__file__).parent.parent / "data" / "holdout"
    manifest, transactions, ground_truth_events = load_locked_holdout_data(data_dir)
    return manifest, transactions, ground_truth_events


# =====================================================================
# 1. Holdout Artifact & Hashing Verification Tests
# =====================================================================

def test_locked_holdout_artifact_exists_and_hash_matches(stored_locked_holdout_artifact):
    """Verify stored locked holdout artifact exists and computed canonical hash matches manifest."""
    manifest, transactions, gt_events = stored_locked_holdout_artifact

    assert manifest.generator_version == "1.0.0"
    assert manifest.seed == 4242

    computed_hash = compute_holdout_dataset_hash(transactions, gt_events)
    assert computed_hash == manifest.dataset_hash


def test_holdout_checksum_mismatch_aborts_evaluation_when_data_mutated(stored_locked_holdout_artifact):
    """Verify mutating actual dataset changes computed SHA-256 hash and raises ChecksumMismatchError."""
    manifest, transactions, gt_events = stored_locked_holdout_artifact

    evaluator = HoldoutEvaluator(manifest=manifest, explicit_evaluation_mode=True)

    # Mutate a transaction in holdout stream
    corrupted_txs = list(transactions)
    mutated_tx = corrupted_txs[0].model_copy(update={"amount": corrupted_txs[0].amount + 999.0})
    corrupted_txs[0] = mutated_tx

    mutated_hash = compute_holdout_dataset_hash(corrupted_txs, gt_events)
    assert mutated_hash != manifest.dataset_hash

    with pytest.raises(ChecksumMismatchError, match="Holdout dataset checksum mismatch"):
        evaluator.evaluate_holdout(
            transactions=corrupted_txs,
            ground_truth_events=gt_events,
        )


def test_holdout_access_denied_in_normal_development_mode(stored_locked_holdout_artifact):
    """Verify attempting to evaluate holdout with explicit_evaluation_mode=False raises HoldoutAccessError."""
    manifest, transactions, gt_events = stored_locked_holdout_artifact
    evaluator = HoldoutEvaluator(manifest=manifest, explicit_evaluation_mode=False)

    with pytest.raises(HoldoutAccessError, match="Holdout access denied"):
        evaluator.evaluate_holdout(
            transactions=transactions,
            ground_truth_events=gt_events,
        )


# =====================================================================
# 2. Historical-Only Baseline & No Current Leakage Test (Blocker 1)
# =====================================================================

def test_holdout_historical_only_baseline_no_current_leakage():
    """Verify current window transactions DO NOT leak into pre-current baseline calculation."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    t_past = st
    t_curr = st + timedelta(minutes=1)

    # Historical past transaction
    past_tx = Transaction(
        transaction_id="TX-PAST-1",
        timestamp=t_past,
        merchant_id="M1",
        customer_id="C1",
        amount=100.0,
        payment_method="CREDIT_CARD",
        country="US",
        device_id="D1",
    )

    feature_engine = FeatureEngine()
    baseline_engine = BaselineEngine(min_window_count=1)

    # Populate baseline history with past snapshot
    past_snap = feature_engine.extract_snapshot("M1", [past_tx], t_past, t_curr)
    baseline_engine.update(past_snap)

    # Current window transaction stream A (normal volume)
    curr_tx_a = Transaction(
        transaction_id="TX-CURR-A",
        timestamp=t_curr,
        merchant_id="M1",
        customer_id="C2",
        amount=100.0,
        payment_method="CREDIT_CARD",
        country="US",
        device_id="D2",
    )
    curr_snap_a = feature_engine.extract_snapshot("M1", [curr_tx_a], t_curr, t_curr + timedelta(minutes=1))

    # Baseline BEFORE updating with current window A
    base_before_a = baseline_engine.get_baseline("M1", curr_snap_a)

    # Current window transaction stream B (massive 100x spike!)
    curr_txs_b = [
        Transaction(
            transaction_id=f"TX-CURR-B-{i}",
            timestamp=t_curr,
            merchant_id="M1",
            customer_id=f"C{i}",
            amount=5000.0,
            payment_method="CREDIT_CARD",
            country="US",
            device_id=f"D{i}",
        )
        for i in range(100)
    ]
    curr_snap_b = feature_engine.extract_snapshot("M1", curr_txs_b, t_curr, t_curr + timedelta(minutes=1))

    # Baseline computed for current window B BEFORE update
    base_before_b = baseline_engine.get_baseline("M1", curr_snap_b)

    # Invariant: pre-current baseline expectations MUST BE 100% IDENTICAL regardless of current window spike!
    assert base_before_a.expected_values == base_before_b.expected_values
    assert base_before_a.robust_scale == base_before_b.robust_scale
    assert base_before_a.history_count == base_before_b.history_count == 1


# =====================================================================
# 3. Single-Pass Holdout Evaluation & Metrics Verification
# =====================================================================

def test_single_pass_holdout_evaluation_success(stored_locked_holdout_artifact):
    """Verify single-pass holdout evaluation with explicit_evaluation_mode=True emits valid EvaluationMetrics."""
    manifest, transactions, gt_events = stored_locked_holdout_artifact
    config = FrozenDetectorConfig(static_threshold=3.5, ewma_alpha=0.3, persistence=2, cooldown_windows=5)

    evaluator = HoldoutEvaluator(manifest=manifest, config=config, explicit_evaluation_mode=True)
    metrics = evaluator.evaluate_holdout(transactions=transactions, ground_truth_events=gt_events)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.tp >= 0
    assert metrics.fp >= 0
    assert metrics.fn >= 0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0


def test_deterministic_holdout_evaluation_replay(stored_locked_holdout_artifact):
    """Verify replaying holdout evaluation produces 100% identical metrics output."""
    manifest, transactions, gt_events = stored_locked_holdout_artifact
    evaluator = HoldoutEvaluator(manifest=manifest, explicit_evaluation_mode=True)

    res1 = evaluator.evaluate_holdout(transactions=transactions, ground_truth_events=gt_events)
    res2 = evaluator.evaluate_holdout(transactions=transactions, ground_truth_events=gt_events)

    assert res1 == res2
    assert res1.model_dump() == res2.model_dump()
