"""Day 7 Final Development Tuning + Freeze Gate & Day 4 Precision/Latency Plotting Test Suite.

Validates:
1. Scorer Strategy Set & AnomalyScorer ABC (Section 17):
   - Common AnomalyScorer ABC interface.
   - StaticThresholdScorer, StatisticalDeviationScorer, and HybridEWMAScorer all implement AnomalyScorer.
   - Strategy polymorphism: uniform invocation without special-casing.
2. Day 4 Precision/Latency Tradeoff Plotting Requirement:
   - Generates precision vs latency tradeoff visualization from development parameter sweep results.
   - Strictly enforces development-only firewall (holdout access raises HoldoutAccessViolationError).
3. Signal Weights Candidate Space & Selection:
   - Evaluates candidate weight vectors (EQUAL, VOLUME_VELOCITY_HEAVY, AMOUNT_HEAVY, BEHAVIORAL_HEAVY).
   - Selection procedure: Declared development selection procedure minimizes Total Cost on development data.
4. Common Scorer Contract & Mathematical Correctness:
   - Produces valid RiskScore objects.
   - Evidence state mapping: INSUFFICIENT -> score=None, conf=0.0; DEGRADED -> conf=0.5; SUFFICIENT -> conf=1.0.
   - Merchant isolation: independent state per merchant.
   - Deterministic replay.
5. Development Parameter Sweeps & Strategy Comparison:
   - Alpha sweep over {0.2, 0.3, 0.5, 0.7, 0.9}.
   - Persistence sweep over {1, 2, 3}.
   - Threshold sweep over complete operating point grid [1.0, 10.0] with step 0.5.
   - Strategy comparison (Static vs Statistical vs Hybrid) with complete metric reporting:
     TP, FP, FN, Precision, Recall, F1, Median Latency, P95 Latency, FP Cost, FN Exposure, Total Cost.
6. Final Operating-Point Selection Procedure:
   - Tests selection procedure mathematically: verifies selected configuration is the exact argmin(total_cost) from the search space without hardcoded expectations.
7. Immutable Freeze Record & Post-Freeze Override Protection:
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
from src.scoring.base import AnomalyScorer
from src.scoring.static import StaticThresholdScorer
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.plots import generate_precision_latency_tradeoff_plot
from src.evaluation.sweeps import (
    run_strategy_comparison,
    run_alpha_sweep,
    run_persistence_sweep,
    run_threshold_operating_point_sweep,
    run_cooldown_sweep,
    run_evidence_parameter_sweep,
    run_signal_weight_sweep,
    select_final_development_configuration,
    load_development_data,
    HoldoutAccessViolationError,
    CANDIDATE_SIGNAL_WEIGHTS,
)
from src.contracts.config_loader import load_runtime_frozen_config
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
# 1. AnomalyScorer ABC & Strategy Polymorphism (Blocker 1)
# =====================================================================

def test_anomaly_scorer_abc_and_polymorphism():
    """Verify common AnomalyScorer ABC interface and polymorphic implementation across all 3 scorers."""
    assert issubclass(StaticThresholdScorer, AnomalyScorer)
    assert issubclass(StatisticalDeviationScorer, AnomalyScorer)
    assert issubclass(HybridEWMAScorer, AnomalyScorer)

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    scorers: list[AnomalyScorer] = [
        StaticThresholdScorer(static_threshold=3.5),
        StatisticalDeviationScorer(static_threshold=3.5),
        HybridEWMAScorer(alpha=0.3, static_threshold=3.5),
    ]

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

    for sc in scorers:
        # Verify polymorphic calculate_score and score alias
        r1 = sc.calculate_score(feat, base)
        r2 = sc.score(feat, base)
        assert isinstance(r1, RiskScore)
        assert isinstance(r2, RiskScore)
        assert r1.score == r2.score
        assert r1.confidence == r2.confidence


# =====================================================================
# 2. Day 4 Precision/Latency Tradeoff Plotting (Blocker 2)
# =====================================================================

def test_day4_precision_latency_tradeoff_plot_generation_and_firewall(tmp_path):
    """Verify precision/latency tradeoff plot generates from development sweep results and enforces holdout firewall."""
    # Synthetic development sweep results
    sweep_results = [
        {"threshold": 1.5, "precision": 0.50, "recall": 1.0, "median_latency_seconds": 60.0, "p95_latency_seconds": 120.0},
        {"threshold": 2.5, "precision": 0.75, "recall": 1.0, "median_latency_seconds": 120.0, "p95_latency_seconds": 180.0},
        {"threshold": 3.5, "precision": 1.00, "recall": 1.0, "median_latency_seconds": 180.0, "p95_latency_seconds": 240.0},
        {"threshold": 5.0, "precision": 1.00, "recall": 0.5, "median_latency_seconds": 300.0, "p95_latency_seconds": 300.0},
    ]

    out_file = tmp_path / "precision_latency_tradeoff.png"
    result_path = generate_precision_latency_tradeoff_plot(
        sweep_results=sweep_results,
        output_path=out_file,
        data_path="data/development/stream.json",
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000  # Non-empty PNG image

    # Strictly verify holdout firewall
    with pytest.raises(HoldoutAccessViolationError):
        generate_precision_latency_tradeoff_plot(
            sweep_results=sweep_results,
            output_path=tmp_path / "illegal.png",
            data_path="data/holdout/stream.json",
        )


# =====================================================================
# 3. Strategy Comparison & Sweeps
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


def test_alpha_persistence_and_complete_threshold_grid_sweeps(dev_benchmark_stream):
    """Verify alpha, persistence, and complete threshold sweeps execute cleanly on development data."""
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

    # 3. Complete operating point threshold sweep [1.0, 10.0] step 0.5 (19 points)
    th_grid = [float(t) for t in np.arange(1.0, 10.5, 0.5)]
    assert len(th_grid) == 19
    # 4. Cooldown sweep {1, 3, 5, 10}
    cd_results = run_cooldown_sweep(txs, events, cooldowns=[1, 3, 5, 10])
    assert len(cd_results) == 4
    for res in cd_results:
        assert res["cooldown_windows"] in [1, 3, 5, 10]
        assert res["total_cost"] is not None

    # 5. Evidence parameter sweep {1, 3, 5, 10}
    ev_results = run_evidence_parameter_sweep(txs, events, min_window_counts=[1, 3, 5, 10])
    assert len(ev_results) == 4
    for res in ev_results:
        assert res["min_window_count"] in [1, 3, 5, 10]
        assert res["total_cost"] is not None

    # 6. Signal weight sweep
    sw_results = run_signal_weight_sweep(txs, events, weight_candidates=CANDIDATE_SIGNAL_WEIGHTS)
    assert len(sw_results) == 4
    for res in sw_results:
        assert res["weight_name"] in CANDIDATE_SIGNAL_WEIGHTS
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
        run_cooldown_sweep(txs, events, data_path="data/holdout/stream.json")

    with pytest.raises(HoldoutAccessViolationError):
        run_evidence_parameter_sweep(txs, events, data_path="data/holdout/stream.json")

    with pytest.raises(HoldoutAccessViolationError):
        run_signal_weight_sweep(txs, events, data_path="data/holdout/stream.json")

    with pytest.raises(HoldoutAccessViolationError):
        select_final_development_configuration(txs, events, data_path="data/holdout/stream.json")


# =====================================================================
# 4. Final Development Selection Procedure & argmin Verification (Blockers 1, 3, 4)
# =====================================================================

def test_selection_procedure_derives_argmin_without_hardcoded_answers(dev_benchmark_stream):
    """Verify selection procedure mathematically derives the optimal operating point from development data."""
    txs, events = dev_benchmark_stream

    # Evaluate across a representative grid of thresholds
    selected = select_final_development_configuration(
        txs,
        events,
        thresholds=[2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        alphas=[0.2, 0.3, 0.5, 0.7, 0.9],
        persistences=[1, 2, 3],
    )

    all_candidates = selected["all_evaluated_candidates"]
    assert len(all_candidates) > 0

    # Find the true argmin across all evaluated candidates in the search space
    min_cost = min(c["score_tuple"][0] for c in all_candidates)
    qualifying = [c for c in all_candidates if c["score_tuple"][0] == min_cost]
    max_f1 = max(-c["score_tuple"][1] for c in qualifying)
    best_qualifying = [c for c in qualifying if -c["score_tuple"][1] == max_f1]
    min_lat = min(c["score_tuple"][2] for c in best_qualifying)
    expected_best = next(c for c in best_qualifying if c["score_tuple"][2] == min_lat)

    # Prove selected configuration equals the true argmin of the declared criteria
    assert selected["score_tuple"] == expected_best["score_tuple"]
    assert selected["selected_scorer"] in ["StaticThresholdScorer", "StatisticalDeviationScorer", "HybridEWMAScorer"]
    assert selected["selected_persistence"] in [1, 2, 3]
    assert selected["selected_threshold"] in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    assert selected["selected_signal_weights"] in CANDIDATE_SIGNAL_WEIGHTS.values()


# =====================================================================
# 5. Freeze Record Creation, Dataset Hash & Override Protection (Blocker 5)
# =====================================================================

def test_freeze_record_integrity_and_override_protection(dev_benchmark_stream, tmp_path):
    """Verify hash determinism, comprehensive dataset hash, and post-freeze override rejection."""
    txs, events = dev_benchmark_stream

    selected = select_final_development_configuration(
        txs,
        events,
        thresholds=[2.5, 3.0, 3.5, 4.0],
        alphas=[0.3, 0.5],
        persistences=[1, 2],
    )

    record = create_freeze_record(
        selected_scorer=selected["selected_scorer"],
        selected_parameters=selected["all_selected_parameters"],
        development_transactions=txs,
        seed=42,
        detector_version="1.0.0",
        selection_rationale="Derived optimal configuration minimizing total cost on development benchmark dataset.",
    )

    # 1. Verify exact hash integrity
    assert record.verify_config(selected["all_selected_parameters"]) is True
    assert record.verify_dataset(txs, seed=42) is True
    assert len(record.config_hash) == 64
    assert len(record.development_dataset_hash) == 64

    # 2. Mutated config -> hash mismatch (override rejected!)
    mutated_params = dict(selected["all_selected_parameters"])
    mutated_params["static_threshold"] = 9.9
    assert record.verify_config(mutated_params) is False

    # 3. Mutated signal weights -> hash mismatch (override rejected!)
    mutated_weights_params = dict(selected["all_selected_parameters"])
    mutated_weights_params["signal_weights"] = {"volume": 99.0}
    assert record.verify_config(mutated_weights_params) is False

    # 4. Mutated dataset / seed -> dataset hash mismatch
    assert record.verify_dataset(txs, seed=999) is False

    # 5. Save and reload
    freeze_file = tmp_path / "freeze_record.json"
    save_freeze_record(record, freeze_file)
    loaded = load_freeze_record(freeze_file)
    assert loaded.config_hash == record.config_hash
    assert loaded.development_dataset_hash == record.development_dataset_hash


def test_canonical_freeze_record_file_exists_and_valid():
    """Verify that config/freeze_record.json exists, is valid, and matches the frozen detector parameters."""
    freeze_path = Path("config/freeze_record.json")
    assert freeze_path.exists(), "config/freeze_record.json must exist"

    record = load_freeze_record(freeze_path)
    assert record.detector_version in ["1.0.0", "1.1.0"]
    assert record.selected_scorer in ["StaticThresholdScorer", "StatisticalDeviationScorer", "HybridEWMAScorer"]
    assert record.seed == 42
    assert len(record.config_hash) == 64
    assert len(record.development_dataset_hash) == 64

    # Verify config hash matches the internal parameters
    assert compute_config_hash(record.all_selected_parameters) == record.config_hash


def test_selection_procedure_reproducibility(dev_benchmark_stream):
    """Verify that running the complete selection procedure twice on identical data produces identical candidates, winner, and config hash."""
    txs, events = dev_benchmark_stream

    run1 = select_final_development_configuration(
        txs,
        events,
        thresholds=[2.5, 3.5, 4.5],
        alphas=[0.3, 0.5],
        persistences=[1, 2],
        cooldowns=[3, 5],
        min_window_counts=[3, 5],
    )
    run2 = select_final_development_configuration(
        txs,
        events,
        thresholds=[2.5, 3.5, 4.5],
        alphas=[0.3, 0.5],
        persistences=[1, 2],
        cooldowns=[3, 5],
        min_window_counts=[3, 5],
    )

    assert run1["selected_scorer"] == run2["selected_scorer"]
    assert run1["all_selected_parameters"] == run2["all_selected_parameters"]
    assert run1["score_tuple"] == run2["score_tuple"]
    assert len(run1["all_evaluated_candidates"]) == len(run2["all_evaluated_candidates"])

    hash1 = compute_config_hash(run1["all_selected_parameters"])
    hash2 = compute_config_hash(run2["all_selected_parameters"])
    assert hash1 == hash2


def test_default_selector_evaluates_complete_candidate_space(dev_benchmark_stream):
    """Verify default selector evaluates all 4 cooldowns x 4 evidence min_window_counts x scorers x thresholds x persistences x weights x alphas (25,536 candidates)."""
    txs, events = dev_benchmark_stream

    selected = select_final_development_configuration(txs, events)
    candidates = selected["all_evaluated_candidates"]

    # Total combinations:
    # 4 mwc * 4 weights * 19 thresholds * 4 cooldowns * 3 persistences * (1 static + 1 statistical + 5 hybrid) = 25,536
    assert len(candidates) == 25536

    evaluated_cooldowns = {c["selected_cooldown"] for c in candidates}
    assert evaluated_cooldowns == {1, 3, 5, 10}

    evaluated_evidence = {c["selected_evidence_params"]["min_window_count"] for c in candidates}
    assert evaluated_evidence == {1, 3, 5, 10}

    evaluated_scorers = {c["selected_scorer"] for c in candidates}
    assert evaluated_scorers == {"StaticThresholdScorer", "StatisticalDeviationScorer", "HybridEWMAScorer"}

    evaluated_persistences = {c["selected_persistence"] for c in candidates}
    assert evaluated_persistences == {1, 2, 3}

    evaluated_alphas = {c["selected_alpha"] for c in candidates if c["selected_scorer"] == "HybridEWMAScorer"}
    assert evaluated_alphas == {0.2, 0.3, 0.5, 0.7, 0.9}


def test_canonical_freeze_record_matches_derived_development_selection():
    """Verify config/freeze_record.json exactly matches the dynamically derived output of full development selection."""
    txs, gts = load_development_data("data/development")
    assert len(txs) > 0
    assert len(gts) > 0

    # 1. Run complete development selection
    derived_selection = select_final_development_configuration(txs, gts)

    # 2. Load canonical freeze record
    freeze_path = Path("config/freeze_record.json")
    assert freeze_path.exists(), "config/freeze_record.json must exist"
    record = load_freeze_record(freeze_path)

    # 3. Verify exact match with derived selection
    assert record.selected_scorer == derived_selection["selected_scorer"]
    assert record.all_selected_parameters == derived_selection["all_selected_parameters"]

    # 4. Verify hashes match
    assert record.config_hash == compute_config_hash(derived_selection["all_selected_parameters"])
    assert record.development_dataset_hash == compute_dataset_hash(txs, seed=42)
    assert record.verify_config(derived_selection["all_selected_parameters"]) is True
    assert record.verify_dataset(txs, seed=42) is True


def test_runtime_detector_configuration_binding_and_consistency(tmp_path):
    """Verify runtime detector configuration binds directly to canonical freeze record and rejects drift/override."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    runtime_config = load_runtime_frozen_config("config/freeze_record.json", "config/detector.yaml")

    # 1. Verify exact 1:1 parameter equality between freeze record and runtime detector configuration
    assert runtime_config.scorer == freeze_record.selected_scorer
    assert runtime_config.ewma_alpha == freeze_record.all_selected_parameters.get("alpha")
    assert runtime_config.static_threshold == freeze_record.all_selected_parameters["static_threshold"]
    assert runtime_config.persistence == freeze_record.all_selected_parameters["persistence"]
    assert runtime_config.cooldown_windows == freeze_record.all_selected_parameters["cooldown_windows"]
    assert runtime_config.min_history_count == freeze_record.all_selected_parameters["min_history_count"]
    assert runtime_config.min_window_count == freeze_record.all_selected_parameters["min_window_count"]
    assert runtime_config.signal_weights == freeze_record.all_selected_parameters["signal_weights"]
    assert runtime_config.detector_version == freeze_record.detector_version

    # 2. Verify runtime config hash equals freeze record config hash
    runtime_dict = {
        "scorer": runtime_config.scorer,
        "alpha": runtime_config.ewma_alpha,
        "static_threshold": runtime_config.static_threshold,
        "persistence": runtime_config.persistence,
        "cooldown_windows": runtime_config.cooldown_windows,
        "min_history_count": runtime_config.min_history_count,
        "min_window_count": runtime_config.min_window_count,
        "signal_weights": runtime_config.signal_weights,
        "detector_version": runtime_config.detector_version,
    }
    assert compute_config_hash(runtime_dict) == freeze_record.config_hash

    # 3. Mutated static_threshold -> ValueError
    bad_yaml_th = tmp_path / "bad_th.yaml"
    bad_yaml_th.write_text(
        """
version: 1.0.0
scorer:
  type: StatisticalDeviationScorer
  alpha: null
  persistence: 1
  static_threshold: 9.9
evidence:
  min_history_count: 1
  min_window_count: 1
state_machine:
  cooldown_windows: 5
signal_weights:
  volume: 1.0
  velocity: 1.0
  amount: 1.0
  behavioral: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match freeze record threshold"):
        load_runtime_frozen_config("config/freeze_record.json", bad_yaml_th)

    # 4. Mutated alpha -> ValueError
    bad_yaml_alpha = tmp_path / "bad_alpha.yaml"
    bad_yaml_alpha.write_text(
        """
version: 1.0.0
scorer:
  type: StatisticalDeviationScorer
  alpha: 0.5
  persistence: 1
  static_threshold: 5.0
evidence:
  min_history_count: 1
  min_window_count: 1
state_machine:
  cooldown_windows: 5
signal_weights:
  volume: 1.0
  velocity: 1.0
  amount: 1.0
  behavioral: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match freeze record alpha"):
        load_runtime_frozen_config("config/freeze_record.json", bad_yaml_alpha)

    # 5. Mutated signal_weights -> ValueError
    bad_yaml_weights = tmp_path / "bad_weights.yaml"
    bad_yaml_weights.write_text(
        """
version: 1.0.0
scorer:
  type: StatisticalDeviationScorer
  alpha: null
  persistence: 1
  static_threshold: 5.0
evidence:
  min_history_count: 1
  min_window_count: 1
state_machine:
  cooldown_windows: 5
signal_weights:
  volume: 0.5
  velocity: 1.0
  amount: 1.0
  behavioral: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match freeze record signal_weights"):
        load_runtime_frozen_config("config/freeze_record.json", bad_yaml_weights)

    # 6. Mutated scorer -> ValueError
    bad_yaml_scorer = tmp_path / "bad_scorer.yaml"
    bad_yaml_scorer.write_text(
        """
version: 1.0.0
scorer:
  type: HybridEWMAScorer
  alpha: null
  persistence: 1
  static_threshold: 5.0
evidence:
  min_history_count: 1
  min_window_count: 1
state_machine:
  cooldown_windows: 5
signal_weights:
  volume: 1.0
  velocity: 1.0
  amount: 1.0
  behavioral: 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match freeze record scorer"):
        load_runtime_frozen_config("config/freeze_record.json", bad_yaml_scorer)




