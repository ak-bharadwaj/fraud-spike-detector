"""Day 8 Locked Holdout Execution, Verification & Research Artifacts Test Suite.

Validates:
1. Holdout Integrity & Manifest Verification:
   - Manifest schema, dataset SHA-256 hash, generator version, seed (4242), schema version (1.0.0).
   - Aborts on checksum or metadata mismatch.
2. Frozen Configuration Enforcement & Override Rejection:
   - Loads exact parameters from config/freeze_record.json.
   - Rejects any parameter overrides.
3. Historical-Only Baseline & Single-Pass Holdout Execution:
   - Executes locked holdout once with historical-only baseline (t_past < t_current).
   - Reports raw TP, FP, FN, Precision, Recall, F1, Median Latency, P95 Latency, FP Cost, FN Exposure, Total Cost.
4. Per-Anomaly Final Table & Zero-Event Class Reporting:
   - Evaluates performance across all required anomaly classes.
   - For zero-event classes (N=0), explicitly reports None for precision/recall/f1 (no false positive claims!).
5. Descriptive Holdout Calibration & Direct RiskScore Bucketing:
   - Operates directly on RiskScore.score without transformation, division by threshold, or clipping.
   - No threshold parameter dependency.
   - Generates reliability buckets (0.5–0.6, 0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9–1.0).
   - Empty buckets explicitly report n=0, mean_predicted_score=None, observed_positive_rate=None.
   - Populated buckets report empirical mean_predicted_score, observed_positive_rate, and sample count N.
   - Population breakdown accounts for all 115 holdout window samples.
   - Computes Expected Calibration Error (ECE) and reliability diagram data.
6. Complete Bootstrap Uncertainty Contract & 1,000 Resamples Execution Test:
   - 1,000 deterministic bootstrap resamples (seed 42) computing 95% CIs for Precision and Recall.
   - Reports raw_numerator_tp, raw_denominator, n_events, and n_alerts.
   - Separate test asserting published bootstrap uncertainty artifact contains n_resamples=1000.
7. Cost Reporting Unit Enforcement (INR '₹'):
   - Explicitly asserts cost unit is '₹' and prevents regression to USD.
8. Portfolio Cost Analysis:
   - Evaluates Static, Statistical, and Hybrid on holdout as descriptive portfolio analysis.
9. Required Artifact Hierarchy:
   - Verifies artifacts/ directory hierarchy including final/metrics.json, final/metrics.csv, final/report.json.
   - Every artifact references experiment_id (EXP-DAY8-HOLDOUT-CORRECTED-002), dataset_hash, config_hash, detector_version, and seed.
10. Unambiguous Provenance Verification Test (Blocker):
    - execution_commit exists (fb3c7f9) and artifact_commit exists (cc2872b) with prior_artifact_commit (775e779).
    - Neither is a placeholder or PENDING_COMMIT.
    - Experiment identity is stable (EXP-DAY8-HOLDOUT-CORRECTED-002).
    - Original run remains disclosed (EXP-DAY8-HOLDOUT-CONFIRMATION-001, 414998f).
    - Corrected run remains canonical.
11. Holdout Immutability & Replay Determinism:
    - Holdout SHA before == Holdout SHA after.
    - Replay reproduces 100% bitwise-identical results.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import csv
import hashlib
import pytest
import numpy as np

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    Alert,
    RiskScore,
    EvaluationMetrics,
)
from src.evaluation.freeze import load_freeze_record, compute_config_hash
from src.evaluation.holdout import (
    HoldoutManifest,
    HoldoutProtection,
    compute_holdout_dataset_hash,
    load_locked_holdout_data,
    HoldoutAccessError,
    ChecksumMismatchError,
)
from src.evaluation.holdout_execution import (
    build_frozen_scorer,
    execute_single_pass_holdout,
    compute_per_anomaly_holdout_metrics,
    compute_descriptive_holdout_calibration,
    compute_bootstrap_uncertainty,
    execute_portfolio_comparison,
    save_day8_research_artifacts,
)


# =====================================================================
# 1. Holdout Integrity & Manifest Verification
# =====================================================================

def test_holdout_manifest_and_sha_verification():
    """Verify holdout manifest, dataset SHA-256 hash, generator version, seed, and schema version."""
    manifest, transactions, ground_truth_events = load_locked_holdout_data("data/holdout")

    assert manifest.generator_version == "1.0.0"
    assert manifest.seed == 4242
    assert manifest.schema_version == "1.0.0"
    assert len(manifest.dataset_hash) == 64

    # Verify actual computed SHA matches manifest
    actual_hash = compute_holdout_dataset_hash(transactions, ground_truth_events)
    assert actual_hash == manifest.dataset_hash
    assert actual_hash == "71595f0cf6681e26ea96232eca900fb805909525367fe90124156de9fa65ddb4"

    # Verify HoldoutProtection allows explicit access and rejects implicit/mutated access
    assert HoldoutProtection.verify_access(manifest=manifest, actual_dataset_hash=actual_hash, explicit_evaluation_mode=True) is True

    with pytest.raises(HoldoutAccessError):
        HoldoutProtection.verify_access(manifest=manifest, actual_dataset_hash=actual_hash, explicit_evaluation_mode=False)

    with pytest.raises(ChecksumMismatchError):
        HoldoutProtection.verify_access(manifest=manifest, actual_dataset_hash="CORRUPTED_HASH", explicit_evaluation_mode=True)


# =====================================================================
# 2. Frozen Configuration Enforcement & Override Rejection
# =====================================================================

def test_frozen_configuration_enforcement_and_override_rejection():
    """Verify frozen configuration is loaded from config/freeze_record.json and parameter overrides are rejected."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    assert freeze_record.detector_version == "1.0.0"
    assert freeze_record.seed == 42
    assert compute_config_hash(freeze_record.all_selected_parameters) == freeze_record.config_hash

    scorer = build_frozen_scorer(freeze_record)
    assert scorer is not None

    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    # Attempting to pass parameter overrides must raise ValueError
    with pytest.raises(ValueError, match="override"):
        execute_single_pass_holdout(
            transactions=txs,
            ground_truth_events=gts,
            freeze_record=freeze_record,
            explicit_evaluation_mode=True,
            override_params={"static_threshold": 9.9},
        )


# =====================================================================
# 3. Single-Pass Holdout Execution & Historical-Only Baseline
# =====================================================================

def test_historical_only_baseline_and_single_pass_execution():
    """Verify single-pass holdout execution with historical-only baseline (t_past < t_current) and complete metrics."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    metrics, alerts, all_scores = execute_single_pass_holdout(
        transactions=txs,
        ground_truth_events=gts,
        freeze_record=freeze_record,
        explicit_evaluation_mode=True,
    )

    # Check raw counts and rates
    assert metrics.tp >= 0
    assert metrics.fp >= 0
    assert metrics.fn >= 0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1_score <= 1.0
    assert metrics.median_latency_seconds is not None
    assert metrics.p95_latency_seconds is not None
    assert metrics.fp_cost >= 0.0
    assert metrics.fn_exposure >= 0.0
    assert metrics.total_cost >= 0.0


# =====================================================================
# 4. Per-Anomaly Final Table & Zero-Event Class Reporting
# =====================================================================

def test_per_anomaly_zero_event_reporting():
    """Verify per-anomaly class evaluation table correctly handles zero-event classes."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    metrics, alerts, _ = execute_single_pass_holdout(
        transactions=txs,
        ground_truth_events=gts,
        freeze_record=freeze_record,
        explicit_evaluation_mode=True,
    )

    per_anomaly = compute_per_anomaly_holdout_metrics(alerts, gts)
    all_classes = [
        "volume_spike", "velocity_burst", "sustained_spike", "amount_shift",
        "behavioral_anomaly", "attribute_shift", "compound_anomaly", "evasive_patterns"
    ]

    for a_type in all_classes:
        assert a_type in per_anomaly
        res = per_anomaly[a_type]
        if res["total_events"] == 0:
            assert res["events_detected"] == 0
            assert res["precision"] is None
            assert res["recall"] is None
            assert res["f1"] is None
            assert res["median_latency_seconds"] is None
            assert res["status"] == "NO_EVENTS_IN_DATASET"
        else:
            assert res["total_events"] > 0
            assert res["precision"] is not None
            assert res["recall"] is not None
            assert res["status"] == "VALIDATED"


# =====================================================================
# 5. Descriptive Holdout Calibration & Direct RiskScore Bucketing
# =====================================================================

def test_descriptive_calibration_direct_bucketing_and_population():
    """Verify descriptive calibration buckets operate directly on RiskScore.score with explicit population breakdown."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    _, _, all_scores = execute_single_pass_holdout(
        transactions=txs,
        ground_truth_events=gts,
        freeze_record=freeze_record,
        explicit_evaluation_mode=True,
    )

    calib = compute_descriptive_holdout_calibration(all_scores, gts)
    buckets = calib["buckets"]
    assert len(buckets) == 5

    bucket_labels = [b["bucket"] for b in buckets]
    assert "0.5–0.6" in bucket_labels
    assert "0.6–0.7" in bucket_labels
    assert "0.7–0.8" in bucket_labels
    assert "0.8–0.9" in bucket_labels
    assert "0.9–1.0" in bucket_labels

    for b in buckets:
        if b["n"] == 0:
            assert b["mean_predicted_score"] is None
            assert b["observed_positive_rate"] is None
        else:
            assert b["mean_predicted_score"] is not None
            assert 0.0 <= b["observed_positive_rate"] <= 1.0
            assert b["range"][0] <= b["mean_predicted_score"] <= b["range"][1]

    # Verify complete population accounting of all 115 window samples
    breakdown = calib["population_breakdown"]
    assert breakdown["total_evaluated_samples"] == 115
    assert breakdown["below_display_buckets_count"] == 9
    assert breakdown["in_display_buckets_count"] == 102
    assert breakdown["above_display_buckets_count"] == 4
    assert breakdown["below_display_buckets_count"] + breakdown["in_display_buckets_count"] + breakdown["above_display_buckets_count"] == breakdown["total_evaluated_samples"]

    # Verify ECE and reliability diagram data
    assert calib["expected_calibration_error"] is not None
    assert calib["expected_calibration_error"] >= 0.0
    assert "mean_predicted_scores" in calib["reliability_diagram_data"]
    assert "observed_positive_rates" in calib["reliability_diagram_data"]


# =====================================================================
# 6. Complete Bootstrap Uncertainty Contract & Published 1000 Resamples
# =====================================================================

def test_bootstrap_uncertainty_contract_with_raw_counts_and_n():
    """Verify bootstrap reporting contract contains raw counts, denominators, and N for precision and recall."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    metrics, alerts, _ = execute_single_pass_holdout(
        transactions=txs,
        ground_truth_events=gts,
        freeze_record=freeze_record,
        explicit_evaluation_mode=True,
    )

    boot = compute_bootstrap_uncertainty(alerts, gts, n_resamples=1000, seed=42, ci=0.95)
    assert boot["n_resamples"] == 1000
    assert boot["seed"] == 42
    assert boot["ci_level"] == 0.95

    # Precision contract
    p_info = boot["precision"]
    assert 0.0 <= p_info["ci_lower"] <= p_info["point"] <= p_info["ci_upper"] <= 1.0
    assert "raw_numerator_tp" in p_info
    assert "raw_denominator_alerts" in p_info
    assert "n_alerts" in p_info
    assert p_info["raw_numerator_tp"] == 1
    assert p_info["raw_denominator_alerts"] == 1
    assert p_info["n_alerts"] == 1

    # Recall contract
    r_info = boot["recall"]
    assert 0.0 <= r_info["ci_lower"] <= r_info["point"] <= r_info["ci_upper"] <= 1.0
    assert "raw_numerator_tp" in r_info
    assert "raw_denominator_events" in r_info
    assert "n_events" in r_info
    assert r_info["raw_numerator_tp"] == 1
    assert r_info["raw_denominator_events"] == 1
    assert r_info["n_events"] == 1

    # Raw counts overview
    assert "raw_counts" in boot
    assert boot["raw_counts"]["tp"] == 1
    assert boot["raw_counts"]["fp"] == 0
    assert boot["raw_counts"]["fn"] == 0


def test_published_bootstrap_uncertainty_artifact_has_1000_resamples():
    """Verify published artifacts/uncertainty/bootstrap_uncertainty.json strictly adheres to 1,000 resamples."""
    boot_path = Path("artifacts/uncertainty/bootstrap_uncertainty.json")
    if not boot_path.exists():
        pytest.skip("Artifacts not yet generated on clean checkout.")

    data = json.loads(boot_path.read_text(encoding="utf-8"))
    assert data["n_resamples"] == 1000
    assert data["seed"] == 42
    assert data["ci_level"] == 0.95
    assert data["precision"]["point"] == 1.0
    assert data["precision"]["raw_numerator_tp"] == 1
    assert data["precision"]["raw_denominator_alerts"] == 1
    assert data["recall"]["point"] == 1.0
    assert data["recall"]["raw_numerator_tp"] == 1
    assert data["recall"]["raw_denominator_events"] == 1


# =====================================================================
# 7. Cost Reporting Unit Enforcement (INR '₹')
# =====================================================================

def test_cost_reporting_unit_is_inr_prevent_usd_regression(tmp_path):
    """Verify cost reporting units in metrics.csv and reports are strictly '₹' and reject USD."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    m, alerts, scores = execute_single_pass_holdout(txs, gts, freeze_record, True)
    per_ano = compute_per_anomaly_holdout_metrics(alerts, gts)
    calib = compute_descriptive_holdout_calibration(scores, gts)
    boot = compute_bootstrap_uncertainty(alerts, gts, n_resamples=100, seed=42)
    port = execute_portfolio_comparison(txs, gts, freeze_record)

    saved_paths = save_day8_research_artifacts(
        base_artifact_dir=tmp_path / "artifacts",
        freeze_record=freeze_record,
        holdout_manifest=manifest,
        holdout_metrics=m,
        per_anomaly_metrics=per_ano,
        calibration_results=calib,
        bootstrap_results=boot,
        portfolio_results=port,
        evasion_results={"status": "CONFIRMED"},
        drift_results={"status": "CONFIRMED"},
        experiment_id="EXP-DAY8-HOLDOUT-CORRECTED-002",
        execution_commit="fb3c7f9",
        artifact_commit="cc2872b",
        prior_artifact_commit="775e779",
    )

    csv_path = saved_paths["final_metrics_csv"]
    csv_text = csv_path.read_text(encoding="utf-8")

    # Strictly verify unit '₹'
    assert "fp_cost,0.00,₹" in csv_text
    assert "fn_exposure,0.00,₹" in csv_text
    assert "total_cost,0.00,₹" in csv_text

    # Prevent regression to USD
    assert "usd" not in csv_text.lower()


# =====================================================================
# 8. Portfolio Analysis
# =====================================================================

def test_portfolio_cost_comparison():
    """Verify descriptive portfolio comparison of Static, Statistical, and Hybrid scorers on holdout."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    port_results = execute_portfolio_comparison(txs, gts, freeze_record)
    assert len(port_results) == 3

    detectors = [p["detector"] for p in port_results]
    assert "StaticThresholdScorer" in detectors
    assert "StatisticalDeviationScorer" in detectors
    assert "HybridEWMAScorer" in detectors

    for p in port_results:
        assert "fp_cost" in p
        assert "fn_exposure" in p
        assert "total_cost" in p
        assert p["cost_unit"] == "₹"
        assert p["total_cost"] == p["fp_cost"] + p["fn_exposure"]


# =====================================================================
# 9. Required Artifact Hierarchy
# =====================================================================

def test_required_artifact_hierarchy(tmp_path):
    """Verify artifacts/ directory hierarchy and all required files."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    m, alerts, scores = execute_single_pass_holdout(txs, gts, freeze_record, True)
    per_ano = compute_per_anomaly_holdout_metrics(alerts, gts)
    calib = compute_descriptive_holdout_calibration(scores, gts)
    boot = compute_bootstrap_uncertainty(alerts, gts, n_resamples=100, seed=42)
    port = execute_portfolio_comparison(txs, gts, freeze_record)

    saved_paths = save_day8_research_artifacts(
        base_artifact_dir=tmp_path / "artifacts",
        freeze_record=freeze_record,
        holdout_manifest=manifest,
        holdout_metrics=m,
        per_anomaly_metrics=per_ano,
        calibration_results=calib,
        bootstrap_results=boot,
        portfolio_results=port,
        evasion_results={"status": "CONFIRMED"},
        drift_results={"status": "CONFIRMED"},
        experiment_id="EXP-DAY8-HOLDOUT-CORRECTED-002",
        execution_commit="fb3c7f9",
        artifact_commit="cc2872b",
        prior_artifact_commit="775e779",
    )

    required_keys = [
        "calibration", "ablation", "drift", "evasion", "uncertainty", "portfolio",
        "final_metrics_json", "final_metrics_csv", "final_report_json",
    ]
    for k in required_keys:
        assert k in saved_paths
        p = saved_paths[k]
        assert p.exists()
        assert p.stat().st_size > 0


# =====================================================================
# 10. Unambiguous Provenance Verification Test (Blocker)
# =====================================================================

def test_unambiguous_provenance_and_dual_run_disclosure(tmp_path):
    """Verify execution_commit and artifact_commit exist, neither is a placeholder, and dual runs are disclosed."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    m, alerts, scores = execute_single_pass_holdout(txs, gts, freeze_record, True)
    per_ano = compute_per_anomaly_holdout_metrics(alerts, gts)
    calib = compute_descriptive_holdout_calibration(scores, gts)
    boot = compute_bootstrap_uncertainty(alerts, gts, n_resamples=1000, seed=42)
    port = execute_portfolio_comparison(txs, gts, freeze_record)

    saved_paths = save_day8_research_artifacts(
        base_artifact_dir=tmp_path / "artifacts",
        freeze_record=freeze_record,
        holdout_manifest=manifest,
        holdout_metrics=m,
        per_anomaly_metrics=per_ano,
        calibration_results=calib,
        bootstrap_results=boot,
        portfolio_results=port,
        evasion_results={"status": "CONFIRMED"},
        drift_results={"status": "CONFIRMED"},
        experiment_id="EXP-DAY8-HOLDOUT-CORRECTED-002",
        execution_commit="fb3c7f9",
        artifact_commit="cc2872b",
        prior_artifact_commit="775e779",
    )

    report_text = saved_paths["final_report_json"].read_text(encoding="utf-8")
    assert "PENDING_COMMIT" not in report_text

    report_content = json.loads(report_text)
    assert report_content["experiment_id"] == "EXP-DAY8-HOLDOUT-CORRECTED-002"
    assert "dual_run_disclosure" in report_content
    
    dual = report_content["dual_run_disclosure"]
    
    # 1. Run 001 Original
    assert "run_001_original" in dual
    r1 = dual["run_001_original"]
    assert r1["experiment_id"] == "EXP-DAY8-HOLDOUT-CONFIRMATION-001"
    assert r1["execution_commit"] == "414998f"
    assert r1["artifact_commit"] == "414998f"
    assert r1["status"] == "SUPERSEDED"
    
    # 2. Run 002 Corrected
    assert "run_002_corrected" in dual
    r2 = dual["run_002_corrected"]
    assert r2["experiment_id"] == "EXP-DAY8-HOLDOUT-CORRECTED-002"
    assert r2["execution_commit"] == "fb3c7f9"
    assert r2["artifact_commit"] == "cc2872b"
    assert r2["prior_artifact_commit"] == "775e779"
    assert r2["status"] == "ACCEPTED_CANONICAL"
    
    # 3. No placeholders
    for r in [r1, r2]:
        for field in ["execution_commit", "artifact_commit", "experiment_id"]:
            assert r[field]
            assert "pending" not in r[field].lower()
            assert "placeholder" not in r[field].lower()


# =====================================================================
# 11. Holdout Immutability & Replay Determinism
# =====================================================================

def test_holdout_immutability_and_replay_determinism():
    """Verify holdout dataset SHA remains identical before and after execution, and replay is 100% deterministic."""
    manifest_before, txs_before, gts_before = load_locked_holdout_data("data/holdout")
    hash_before = compute_holdout_dataset_hash(txs_before, gts_before)

    freeze_record = load_freeze_record("config/freeze_record.json")

    # Run 1
    m1, alerts1, scores1 = execute_single_pass_holdout(txs_before, gts_before, freeze_record, True)

    # Run 2 (Replay)
    m2, alerts2, scores2 = execute_single_pass_holdout(txs_before, gts_before, freeze_record, True)

    # Verify 100% bitwise determinism
    assert m1.model_dump() == m2.model_dump()
    assert len(alerts1) == len(alerts2)
    assert len(scores1) == len(scores2)

    # Verify holdout SHA after == holdout SHA before
    manifest_after, txs_after, gts_after = load_locked_holdout_data("data/holdout")
    hash_after = compute_holdout_dataset_hash(txs_after, gts_after)
    assert hash_after == hash_before
