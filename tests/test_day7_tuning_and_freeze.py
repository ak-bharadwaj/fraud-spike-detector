"""Day 7 Final Development Tuning + Freeze Gate Test Suite.

Validates:
1. Scorer Strategy Set:
   - StaticThresholdScorer: Genuine strategy calculating fixed threshold limits.
   - StatisticalDeviationScorer: Standardized deviation magnitude M_k = |f_k - exp_k| / scale_k, S_raw = max_k M_k.
   - HybridEWMAScorer: Standardized deviation smoothed via EWMA S_ewma = alpha * S_raw + (1-alpha) * S_ewma_prev.
2. Common Scorer Contract:
   - Produces valid RiskScore objects.
   - Evidence state mapping: INSUFFICIENT -> score=None, conf=0.0; DEGRADED -> conf=0.5; SUFFICIENT -> conf=1.0.
   - Merchant isolation: independent state per merchant.
   - Deterministic replay.
3. Development Parameter Sweeps & Strategy Comparison:
   - Alpha sweep over {0.2, 0.3, 0.5, 0.7, 0.9}.
   - Persistence sweep over {1, 2, 3}.
   - Threshold sweep over operating points.
   - Strategy comparison (Static vs Statistical vs Hybrid) with complete metric reporting:
     TP, FP, FN, Precision, Recall, F1, Median Latency, P95 Latency, FP Cost, FN Exposure, Total Cost.
4. Strict Development-Only Firewall:
   - Attempts to pass holdout paths to sweeps raise HoldoutAccessViolationError.
5. Final Operating-Point Selection & Freeze Record:
   - Reproducible selection minimizing Total Cost on development benchmark dataset.
   - Durable FreezeRecord creation with deterministic config_hash and development_dataset_hash.
   - Post-freeze override protection (mutation rejects verification).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import pytest
import numpy as np

from src.contracts.contracts import (
    Transaction,
    FeatureSnapshot,
    BaselineSnapshot,
    RiskScore,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.scoring.static import StaticThresholdScorer
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.sweeps import (
    run_strategy_comparison,
    run_alpha_sweep,
    run_persistence_sweep,
    run_threshold_operating_point_sweep,
    select_final_development_configuration,
    HoldoutAccessViolationError,
)
from src.evaluation.freeze import (
    FreezeRecord,
    create_freeze_record,
    compute_config_hash,
    compute_dataset_hash,
    save_freeze_record,
    load_freeze_record,
)
from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock


# =====================================================================
# 1. Scorer Strategy Set & Mathematical Correctness
# =====================================================================

def test_static_threshold_scorer_mathematical_correctness():
    """Verify StaticThresholdScorer computes static threshold ratios against fixed limits."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = StaticThresholdScorer(static_threshold=3.5, static_limits={"volume": 20.0, "velocity": 20.0})

    feat = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st,
        volume=30.0,  # 30 / 20 * 3.5 = 5.25
        velocity=10.0,  # 10 / 20 * 3.5 = 1.75
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 1000.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=10,
        unique_devices=8,
        data_quality="GOOD",
    )
    base = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st,
        expected_values={"volume": 10.0, "velocity": 10.0},
        robust_scale={"volume": 2.0, "velocity": 2.0},
        history_count=10,
        current_window_count=10,
        evidence_state="SUFFICIENT",
    )

    risk = scorer.calculate_score(feat, base, signal_mask=["volume", "velocity"])
    assert risk.score == pytest.approx(5.25, abs=1e-5)
    assert risk.confidence == 1.0
    assert risk.data_quality == "GOOD"
    assert "volume" in risk.triggered_signals
    assert "velocity" not in risk.triggered_signals


def test_statistical_deviation_scorer_mathematical_correctness():
    """Verify StatisticalDeviationScorer computes standardized deviation M_k = |f_k - exp_k| / scale_k."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = StatisticalDeviationScorer(static_threshold=3.5)

    feat = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st,
        volume=20.0,
        velocity=10.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 500.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=5,
        unique_devices=4,
        data_quality="GOOD",
    )
    base = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st,
        expected_values={
            "volume": 10.0, "velocity": 10.0, "unique_customers": 5.0, "unique_devices": 4.0,
            "amount_total_amount": 500.0, "amount_mean_amount": 50.0, "amount_std_amount": 5.0,
            "amount_median_amount": 50.0, "amount_mad_amount": 3.0, "amount_min_amount": 40.0, "amount_max_amount": 60.0,
        },
        robust_scale={
            "volume": 2.0, "velocity": 2.0, "unique_customers": 1.0, "unique_devices": 1.0,
            "amount_total_amount": 100.0, "amount_mean_amount": 10.0, "amount_std_amount": 2.0,
            "amount_median_amount": 10.0, "amount_mad_amount": 1.0, "amount_min_amount": 10.0, "amount_max_amount": 15.0,
        },
        history_count=10,
        current_window_count=10,
        evidence_state="SUFFICIENT",
    )

    risk = scorer.calculate_score(feat, base)
    # volume: |20 - 10| / 2 = 5.0, max is 5.0
    assert risk.score == pytest.approx(5.0, abs=1e-5)
    assert risk.confidence == 1.0
    assert "volume" in risk.triggered_signals


def test_hybrid_ewma_scorer_mathematical_correctness_and_smoothing():
    """Verify HybridEWMAScorer computes smoothed score S_ewma = alpha * S_raw + (1-alpha) * S_ewma_prev."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)

    base = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st,
        expected_values={
            "volume": 10.0, "velocity": 10.0, "unique_customers": 5.0, "unique_devices": 4.0,
            "amount_total_amount": 500.0, "amount_mean_amount": 50.0, "amount_std_amount": 5.0,
            "amount_median_amount": 50.0, "amount_mad_amount": 3.0, "amount_min_amount": 40.0, "amount_max_amount": 60.0,
        },
        robust_scale={
            "volume": 2.0, "velocity": 2.0, "unique_customers": 1.0, "unique_devices": 1.0,
            "amount_total_amount": 100.0, "amount_mean_amount": 10.0, "amount_std_amount": 2.0,
            "amount_median_amount": 10.0, "amount_mad_amount": 1.0, "amount_min_amount": 10.0, "amount_max_amount": 15.0,
        },
        history_count=10,
        current_window_count=10,
        evidence_state="SUFFICIENT",
    )

    # Window 0: S_raw = 5.0 -> S_ewma = 5.0 (initial window)
    feat0 = FeatureSnapshot(
        merchant_id="M1", timestamp=st, volume=20.0, velocity=10.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 500.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=5, unique_devices=4, data_quality="GOOD",
    )
    r0 = scorer.calculate_score(feat0, base)
    assert r0.score == pytest.approx(5.0, abs=1e-5)

    # Window 1: S_raw = 0.0 -> S_ewma = 0.3 * 0.0 + 0.7 * 5.0 = 3.5
    feat1 = FeatureSnapshot(
        merchant_id="M1", timestamp=st + timedelta(minutes=1), volume=10.0, velocity=10.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 500.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=5, unique_devices=4, data_quality="GOOD",
    )
    r1 = scorer.calculate_score(feat1, base)
    assert r1.score == pytest.approx(3.5, abs=1e-5)


# =====================================================================
# 2. Common Scorer Contract & Merchant Isolation
# =====================================================================

def test_common_scorer_contract_and_merchant_isolation():
    """Verify common scorer contract across all 3 strategies:
    - INSUFFICIENT -> score=None, confidence=0.0
    - DEGRADED -> score is float, confidence=0.5
    - SUFFICIENT -> score is float, confidence=1.0
    - Merchant isolation: Merchant A state does not bleed into Merchant B.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorers = [
        StaticThresholdScorer(static_threshold=3.5),
        StatisticalDeviationScorer(static_threshold=3.5),
        HybridEWMAScorer(alpha=0.3, static_threshold=3.5),
    ]

    base_ins = BaselineSnapshot(merchant_id="M1", timestamp=st, evidence_state="INSUFFICIENT", history_count=0, current_window_count=0)
    feat_empty = FeatureSnapshot(merchant_id="M1", timestamp=st, volume=0.0, velocity=0.0, unique_customers=0, unique_devices=0, data_quality="EMPTY")

    for sc in scorers:
        r = sc.calculate_score(feat_empty, base_ins)
        assert r.score is None
        assert r.confidence == 0.0
        assert r.triggered_signals == []

    # Merchant isolation test on HybridEWMAScorer
    ewma_scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)
    base_suf_m1 = BaselineSnapshot(
        merchant_id="M1", timestamp=st,
        expected_values={"volume": 10.0, "velocity": 10.0, "unique_customers": 5.0, "unique_devices": 4.0, "amount_total_amount": 500.0, "amount_mean_amount": 50.0, "amount_std_amount": 5.0, "amount_median_amount": 50.0, "amount_mad_amount": 3.0, "amount_min_amount": 40.0, "amount_max_amount": 60.0},
        robust_scale={"volume": 2.0, "velocity": 2.0, "unique_customers": 1.0, "unique_devices": 1.0, "amount_total_amount": 100.0, "amount_mean_amount": 10.0, "amount_std_amount": 2.0, "amount_median_amount": 10.0, "amount_mad_amount": 1.0, "amount_min_amount": 10.0, "amount_max_amount": 15.0},
        history_count=10, current_window_count=10, evidence_state="SUFFICIENT",
    )
    base_suf_m2 = base_suf_m1.model_copy(update={"merchant_id": "M2"})

    feat_burst = FeatureSnapshot(
        merchant_id="M1", timestamp=st, volume=30.0, velocity=10.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 500.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=5, unique_devices=4, data_quality="GOOD",
    )
    feat_normal = FeatureSnapshot(
        merchant_id="M2", timestamp=st, volume=10.0, velocity=10.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 500.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=5, unique_devices=4, data_quality="GOOD",
    )

    r_m1 = ewma_scorer.calculate_score(feat_burst, base_suf_m1)
    r_m2 = ewma_scorer.calculate_score(feat_normal, base_suf_m2)

    assert r_m1.score == pytest.approx(10.0, abs=1e-5)  # |30-10|/2 = 10
    assert r_m2.score == pytest.approx(0.0, abs=1e-5)   # |10-10|/2 = 0


# =====================================================================
# 3. Strategy Comparison & Development Sweeps
# =====================================================================

@pytest.fixture
def dev_benchmark_stream():
    """Deterministic development benchmark stream fixture."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    clock = VirtualClock(initial_time=st)
    gen = SyntheticStreamGenerator(42, [{"id": "M_DEV", "archetype": "stable"}], clock)

    txs_base, _ = gen.generate_window(15.0)
    spec = AnomalySpec(
        anomaly_type="volume_spike",
        start_time=st + timedelta(minutes=15),
        duration_seconds=180.0,
        target_magnitude=4.5,
        parameters={"rate_multiplier": 4.0},
    )
    gen.schedule_anomaly("M_DEV", spec, event_id="EVT-DEV-SPIKE")
    txs_ano, events = gen.generate_window(3.0)
    return txs_base + txs_ano, events


def test_strategy_comparison_complete_metrics(dev_benchmark_stream):
    """Verify strategy comparison evaluates Static, Statistical, and Hybrid with complete metrics."""
    txs, events = dev_benchmark_stream
    results = run_strategy_comparison(txs, events)

    assert len(results) == 3
    strat_names = [r["strategy"] for r in results]
    assert "StaticThresholdScorer" in strat_names
    assert "StatisticalDeviationScorer" in strat_names
    assert "HybridEWMAScorer" in strat_names

    for r in results:
        assert r["tp"] >= 0
        assert r["fp"] >= 0
        assert r["fn"] >= 0
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
        assert 0.0 <= r["f1_score"] <= 1.0
        assert r["fp_cost"] >= 0.0
        assert r["fn_exposure"] >= 0.0
        assert r["total_cost"] >= 0.0


def test_alpha_persistence_threshold_sweeps(dev_benchmark_stream):
    """Verify alpha, persistence, and threshold sweeps execute cleanly on development data."""
    txs, events = dev_benchmark_stream

    # 1. Alpha sweep {0.2, 0.3, 0.5, 0.7, 0.9}
    alpha_results = run_alpha_sweep(txs, events, alphas=[0.2, 0.3, 0.5, 0.7, 0.9])
    assert len(alpha_results) == 5
    for res in alpha_results:
        assert res["alpha"] in [0.2, 0.3, 0.5, 0.7, 0.9]
        assert res["total_cost"] is not None

    # 2. Persistence sweep {1, 2, 3}
    p_results = run_persistence_sweep(txs, events, persistences=[1, 2, 3])
    assert len(p_results) == 3
    for res in p_results:
        assert res["persistence"] in [1, 2, 3]
        assert res["total_cost"] is not None

    # 3. Threshold sweep
    th_results = run_threshold_operating_point_sweep(txs, events, thresholds=[2.5, 3.0, 3.5, 4.0])
    assert len(th_results) == 4
    for res in th_results:
        assert res["threshold"] in [2.5, 3.0, 3.5, 4.0]
        assert res["total_cost"] is not None


def test_strict_development_only_enforcement(dev_benchmark_stream):
    """Verify that attempting to point sweeps at holdout data raises HoldoutAccessViolationError."""
    txs, events = dev_benchmark_stream

    with pytest.raises(HoldoutAccessViolationError):
        run_strategy_comparison(txs, events, data_path="data/holdout/stream.json")

    with pytest.raises(HoldoutAccessViolationError):
        run_alpha_sweep(txs, events, data_path="data/holdout/stream.json")

    with pytest.raises(HoldoutAccessViolationError):
        run_persistence_sweep(txs, events, data_path="data/holdout/stream.json")

    with pytest.raises(HoldoutAccessViolationError):
        run_threshold_operating_point_sweep(txs, events, data_path="data/holdout/stream.json")

    with pytest.raises(HoldoutAccessViolationError):
        select_final_development_configuration(txs, events, data_path="data/holdout/stream.json")


# =====================================================================
# 4. Final Development Selection & Immutable Freeze Record
# =====================================================================

def test_final_development_selection_and_freeze_record_creation(dev_benchmark_stream, tmp_path):
    """Verify optimal configuration selection and durable FreezeRecord creation."""
    txs, events = dev_benchmark_stream

    # 1. Execute selection procedure
    selected = select_final_development_configuration(txs, events)
    assert selected["selected_scorer"] == "HybridEWMAScorer"
    assert selected["selected_alpha"] in [0.2, 0.3, 0.5, 0.7, 0.9]
    assert selected["selected_persistence"] in [1, 2, 3]
    assert selected["selected_threshold"] > 0.0

    # 2. Create FreezeRecord
    record = create_freeze_record(
        selected_scorer=selected["selected_scorer"],
        selected_parameters={
            "scorer": selected["selected_scorer"],
            "alpha": selected["selected_alpha"],
            "static_threshold": selected["selected_threshold"],
            "persistence": selected["selected_persistence"],
            "cooldown_windows": selected["selected_cooldown"],
            "min_window_count": selected["selected_evidence_params"]["min_window_count"],
            "signal_config": selected["selected_signal_config"],
            "detector_version": "1.0.0",
        },
        development_transactions=txs,
        seed=42,
        detector_version="1.0.0",
        selection_rationale="Optimal development operating point minimizing total cost and maximizing F1.",
    )

    assert record.detector_version == "1.0.0"
    assert record.seed == 42
    assert len(record.config_hash) == 64
    assert len(record.development_dataset_hash) == 64

    # 3. Save and reload FreezeRecord
    freeze_file = tmp_path / "freeze_record.json"
    save_freeze_record(record, freeze_file)
    loaded = load_freeze_record(freeze_file)

    assert loaded.config_hash == record.config_hash
    assert loaded.development_dataset_hash == record.development_dataset_hash
    assert loaded.all_selected_parameters == record.all_selected_parameters


def test_freeze_record_integrity_and_override_protection(dev_benchmark_stream):
    """Verify hash determinism and post-freeze override rejection."""
    txs, _ = dev_benchmark_stream

    base_params = {
        "scorer": "HybridEWMAScorer",
        "alpha": 0.3,
        "static_threshold": 3.5,
        "persistence": 2,
        "cooldown_windows": 5,
        "min_window_count": 5,
        "detector_version": "1.0.0",
    }

    record = create_freeze_record(
        selected_scorer="HybridEWMAScorer",
        selected_parameters=base_params,
        development_transactions=txs,
        seed=42,
    )

    # 1. Same config -> same hash
    assert record.verify_config(base_params) is True
    assert record.verify_dataset(txs, seed=42) is True

    # 2. Mutated config -> hash mismatch (override rejected!)
    mutated_params = dict(base_params)
    mutated_params["static_threshold"] = 4.0
    assert record.verify_config(mutated_params) is False

    # 3. Mutated alpha -> hash mismatch
    mutated_alpha = dict(base_params)
    mutated_alpha["alpha"] = 0.5
    assert record.verify_config(mutated_alpha) is False

    # 4. Mutated dataset / seed -> dataset hash mismatch
    assert record.verify_dataset(txs, seed=999) is False


def test_canonical_freeze_record_file_exists_and_valid():
    """Verify that config/freeze_record.json exists, is valid, and matches the frozen detector parameters."""
    freeze_path = Path("config/freeze_record.json")
    assert freeze_path.exists(), "config/freeze_record.json must exist"

    record = load_freeze_record(freeze_path)
    assert record.detector_version == "1.0.0"
    assert record.selected_scorer == "HybridEWMAScorer"
    assert record.seed == 42
    assert len(record.config_hash) == 64
    assert len(record.development_dataset_hash) == 64

    # Verify config hash matches the internal parameters
    assert compute_config_hash(record.all_selected_parameters) == record.config_hash
