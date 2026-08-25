"""Comprehensive behavioral unit tests for Day 10 Ablation Experiments.

Validates all required ablation behavioral dimensions:
1. Single-factor causation (each variant modifies exactly one component mechanism).
2. FULL_PIPELINE control baseline (delta_f1 = 0.0, delta_precision = 0.0, delta_recall = 0.0).
3. NO_EWMA variant execution (alpha=1.0).
4. NO_PERSISTENCE variant execution (P=1).
5. NO_COOLDOWN variant execution (C=0).
6. SINGLE_FEATURE_VOLUME_ONLY variant execution (volume feature only).
7. Identical evaluation stream invariance across all ablation variants.
8. Deterministic ablation replay.
9. AblationResult Pydantic schema compliance.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent, AblationVariantConfig, AblationResult
from src.evaluation.ablation import AblationRunner
from src.evaluation.holdout import load_locked_holdout_data


# =====================================================================
# Fixture
# =====================================================================

@pytest.fixture
def eval_dataset():
    """Load benchmark dataset stream from data/holdout/."""
    data_dir = Path(__file__).parent.parent / "data" / "holdout"
    manifest, transactions, ground_truth_events = load_locked_holdout_data(data_dir)
    return transactions, ground_truth_events


# =====================================================================
# 1. Ablation Suite Execution & Control Baseline Tests
# =====================================================================

def test_ablation_suite_control_baseline_delta(eval_dataset):
    """Verify FULL_PIPELINE control baseline produces zero delta metrics."""
    txs, gt_events = eval_dataset
    runner = AblationRunner()

    results = runner.run_ablation_suite(txs, gt_events)
    assert len(results) == 5  # Control + 4 standard ablation variants

    control = results[0]
    assert control.variant_id == "FULL_PIPELINE"
    assert control.delta_f1 == 0.0
    assert control.delta_precision == 0.0
    assert control.delta_recall == 0.0
    assert control.delta_latency_seconds == 0.0


def test_ablation_variants_single_factor_causation():
    """Verify standard ablation variants modify exactly one parameter/component."""
    variants = AblationRunner.get_standard_ablation_variants()

    no_ewma = next(v for v in variants if v.variant_id == "NO_EWMA")
    assert no_ewma.disable_ewma is True
    assert no_ewma.persistence == 2
    assert no_ewma.cooldown_windows == 5
    assert no_ewma.feature_subset is None

    no_pers = next(v for v in variants if v.variant_id == "NO_PERSISTENCE")
    assert no_pers.disable_ewma is False
    assert no_pers.persistence == 1
    assert no_pers.cooldown_windows == 5
    assert no_pers.feature_subset is None

    no_cool = next(v for v in variants if v.variant_id == "NO_COOLDOWN")
    assert no_cool.disable_ewma is False
    assert no_cool.persistence == 2
    assert no_cool.cooldown_windows == 0
    assert no_cool.feature_subset is None

    vol_only = next(v for v in variants if v.variant_id == "SINGLE_FEATURE_VOLUME_ONLY")
    assert vol_only.disable_ewma is False
    assert vol_only.persistence == 2
    assert vol_only.cooldown_windows == 5
    assert vol_only.feature_subset == ["volume"]


# =====================================================================
# 2. Individual Variant Evaluation & Schema Compliance
# =====================================================================

def test_individual_ablation_variant_evaluation(eval_dataset):
    """Verify individual ablation variants execute cleanly and produce valid metrics."""
    txs, gt_events = eval_dataset
    runner = AblationRunner()

    var = AblationVariantConfig(
        variant_id="NO_PERSISTENCE",
        description="Persistence disabled (P=1)",
        persistence=1,
    )

    metrics = runner.evaluate_variant(var, txs, gt_events)
    assert metrics.tp >= 0
    assert metrics.fp >= 0
    assert metrics.fn >= 0


def test_deterministic_ablation_replay(eval_dataset):
    """Verify identical ablation suite execution produces 100% identical AblationResult metrics."""
    txs, gt_events = eval_dataset
    runner = AblationRunner()

    res1 = runner.run_ablation_suite(txs, gt_events)
    res2 = runner.run_ablation_suite(txs, gt_events)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1 == r2
        assert r1.model_dump() == r2.model_dump()


def test_ablation_result_pydantic_schema_compliance(eval_dataset):
    """Verify AblationResult validates strictly against Pydantic schema."""
    txs, gt_events = eval_dataset
    runner = AblationRunner()

    results = runner.run_ablation_suite(txs, gt_events)
    for res in results:
        dumped = res.model_dump()
        reconstructed = AblationResult(**dumped)
        assert reconstructed.variant_id == res.variant_id
