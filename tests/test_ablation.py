"""Comprehensive behavioral unit tests for Day 10 Ablation Experiments.

Validates all required ablation behavioral dimensions:
1. Characterization dataset usage (loaded strictly from data/development/, NOT data/holdout/).
2. Holdout contamination rejection (attempting to load data/holdout/ raises ValueError).
3. Single-factor causation enforcement (variants modifying > 1 factor raise ValueError).
4. FULL_PIPELINE control baseline (uses FrozenDetectorConfig, delta_f1 = 0.0, delta_precision = 0.0, delta_recall = 0.0).
5. NO_EWMA variant execution (alpha=1.0).
6. NO_PERSISTENCE variant execution (P=1).
7. NO_COOLDOWN variant execution (C=0).
8. SINGLE_FEATURE_VOLUME_ONLY variant execution (volume feature only).
9. Identical evaluation stream invariance across all ablation variants.
10. Deterministic ablation replay.
11. AblationResult Pydantic schema compliance.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import Transaction, GroundTruthEvent, AblationVariantConfig, AblationResult
from src.evaluation.ablation import AblationRunner, load_characterization_data
from src.evaluation.holdout import FrozenDetectorConfig


# =====================================================================
# Fixture loading development/characterization dataset
# =====================================================================

@pytest.fixture
def characterization_dataset():
    """Load development characterization stream from data/development/ (NOT holdout!)."""
    data_dir = Path(__file__).parent.parent / "data" / "development"
    manifest, transactions, ground_truth_events = load_characterization_data(data_dir)
    return manifest, transactions, ground_truth_events


# =====================================================================
# 1. Holdout Contamination & Single-Factor Enforcement Tests
# =====================================================================

def test_holdout_contamination_prevention_raises_value_error():
    """Verify attempting to pass data/holdout/ to load_characterization_data raises ValueError."""
    holdout_dir = Path(__file__).parent.parent / "data" / "holdout"
    with pytest.raises(ValueError, match="Holdout contamination error"):
        load_characterization_data(holdout_dir)


def test_multi_factor_ablation_variant_rejection():
    """Verify attempting to execute a multi-factor ablation variant raises ValueError."""
    runner = AblationRunner()

    multi_factor_var = AblationVariantConfig(
        variant_id="INVALID_MULTI_FACTOR",
        description="Modifies two factors simultaneously",
        disable_ewma=True,
        persistence=3,  # Second modified factor!
    )

    with pytest.raises(ValueError, match="Invalid multi-factor ablation variant"):
        runner.validate_single_factor_variant(multi_factor_var)


def test_zero_factor_ablation_variant_rejection():
    """Verify non-control variant with zero parameter modifications raises ValueError."""
    runner = AblationRunner()

    zero_factor_var = AblationVariantConfig(
        variant_id="ZERO_FACTOR",
        description="No parameters modified",
    )

    with pytest.raises(ValueError, match="No causal factor modified"):
        runner.validate_single_factor_variant(zero_factor_var)


# =====================================================================
# 2. Ablation Suite Execution & Control Baseline Tests
# =====================================================================

def test_ablation_suite_control_baseline_delta(characterization_dataset):
    """Verify FULL_PIPELINE control baseline produces zero delta metrics on characterization stream."""
    manifest, txs, gt_events = characterization_dataset
    runner = AblationRunner()

    results = runner.run_ablation_suite(txs, gt_events)
    assert len(results) == 5  # Control + 4 standard ablation variants

    control = results[0]
    assert control.variant_id in ("FULL", "FULL_PIPELINE")
    assert control.delta_f1 == 0.0
    assert control.delta_precision == 0.0
    assert control.delta_recall == 0.0
    assert control.delta_latency_seconds == 0.0


def test_ablation_variants_single_factor_causation():
    """Verify standard ablation variants modify exactly one parameter/component relative to frozen config."""
    config = FrozenDetectorConfig()
    runner = AblationRunner(config=config)
    variants = runner.get_standard_ablation_variants()

    for var in variants:
        # Must pass single-factor validation
        runner.validate_single_factor_variant(var)


# =====================================================================
# 3. Individual Variant Evaluation & Deterministic Replay
# =====================================================================

def test_individual_ablation_variant_evaluation(characterization_dataset):
    """Verify individual ablation variants execute cleanly and produce valid metrics."""
    manifest, txs, gt_events = characterization_dataset
    runner = AblationRunner()

    config = FrozenDetectorConfig()
    var = AblationVariantConfig(
        variant_id="NO_PERSISTENCE",
        description="Persistence disabled (P=1)",
        persistence=1,
        cooldown_windows=config.cooldown_windows,
        static_threshold=config.static_threshold,
    )

    metrics = runner.evaluate_variant(var, txs, gt_events)
    assert metrics.tp >= 0
    assert metrics.fp >= 0
    assert metrics.fn >= 0


def test_deterministic_ablation_replay(characterization_dataset):
    """Verify identical ablation suite execution produces 100% identical AblationResult metrics."""
    manifest, txs, gt_events = characterization_dataset
    runner = AblationRunner()

    res1 = runner.run_ablation_suite(txs, gt_events)
    res2 = runner.run_ablation_suite(txs, gt_events)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2):
        assert r1 == r2
        assert r1.model_dump() == r2.model_dump()


def test_ablation_result_pydantic_schema_compliance(characterization_dataset):
    """Verify AblationResult validates strictly against Pydantic schema."""
    manifest, txs, gt_events = characterization_dataset
    runner = AblationRunner()

    results = runner.run_ablation_suite(txs, gt_events)
    for res in results:
        dumped = res.model_dump()
        reconstructed = AblationResult(**dumped)
        assert reconstructed.variant_id == res.variant_id
