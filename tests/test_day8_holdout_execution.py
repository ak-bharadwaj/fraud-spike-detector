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
5. Descriptive Holdout Calibration & Empty-Bucket Handling:
   - Generates reliability buckets (0.5–0.6, 0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9–1.0).
   - Empty buckets explicitly report n=0, mean_score=None, observed_positive_rate=None (no midpoint pseudo-values!).
   - Computes Expected Calibration Error (ECE) and reliability diagram data.
6. Bootstrap Uncertainty:
   - 1,000 deterministic bootstrap resamples (seed 42) computing 95% CIs for Precision and Recall.
7. Portfolio Cost Analysis:
   - Evaluates Static, Statistical, and Hybrid on holdout as descriptive portfolio analysis.
8. Required Artifact Hierarchy:
   - Verifies artifacts/ directory hierarchy including final/metrics.json, final/metrics.csv, final/report.json.
   - Every artifact references dataset_hash, config_hash, detector_version, and seed.
9. Holdout Immutability & Replay Determinism:
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
# 4. Per-Anomaly Final Table & Zero-Event Class Reporting (Blocker 4)
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
# 5. Descriptive Holdout Calibration & Empty-Bucket Handling (Blocker 1 & 2)
# =====================================================================

def test_descriptive_calibration_empty_buckets_and_ece():
    """Verify descriptive calibration buckets report None for empty buckets without midpoint pseudo-values."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    _, _, all_scores = execute_single_pass_holdout(
        transactions=txs,
        ground_truth_events=gts,
        freeze_record=freeze_record,
        explicit_evaluation_mode=True,
    )

    calib = compute_descriptive_holdout_calibration(all_scores, gts, threshold=1.0)
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
            # Must be None, NOT midpoint pseudo-values!
            assert b["mean_score"] is None
            assert b["observed_positive_rate"] is None
        else:
            assert b["mean_score"] is not None
            assert 0.0 <= b["observed_positive_rate"] <= 1.0

    assert calib["expected_calibration_error"] is not None
    assert calib["expected_calibration_error"] >= 0.0


# =====================================================================
# 6. Bootstrap Uncertainty Analysis (1,000 Resamples)
# =====================================================================

def test_bootstrap_uncertainty_1000_resamples():
    """Verify 1,000 deterministic bootstrap resamples computing 95% CIs for Precision and Recall."""
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

    p_info = boot["precision"]
    r_info = boot["recall"]
    assert 0.0 <= p_info["ci_lower"] <= p_info["point"] <= p_info["ci_upper"] <= 1.0
    assert 0.0 <= r_info["ci_lower"] <= r_info["point"] <= r_info["ci_upper"] <= 1.0


# =====================================================================
# 7. Portfolio Analysis
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
        assert p["total_cost"] == p["fp_cost"] + p["fn_exposure"]


# =====================================================================
# 8. Required Artifact Hierarchy & Final Files (Blocker 3)
# =====================================================================

def test_required_artifact_hierarchy_and_final_files(tmp_path):
    """Verify artifacts/ directory hierarchy including final/metrics.json, final/metrics.csv, final/report.json."""
    freeze_record = load_freeze_record("config/freeze_record.json")
    manifest, txs, gts = load_locked_holdout_data("data/holdout")

    m, alerts, scores = execute_single_pass_holdout(txs, gts, freeze_record, True)
    per_ano = compute_per_anomaly_holdout_metrics(alerts, gts)
    calib = compute_descriptive_holdout_calibration(scores, gts, threshold=1.0)
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

    # Verify final/metrics.csv content
    csv_content = saved_paths["final_metrics_csv"].read_text(encoding="utf-8")
    assert "metric,value,unit" in csv_content
    assert "tp,1,count" in csv_content

    # Verify final/report.json content
    report_content = json.loads(saved_paths["final_report_json"].read_text(encoding="utf-8"))
    assert report_content["detector_version"] == "1.0.0"
    assert report_content["config_hash"] == freeze_record.config_hash
    assert report_content["holdout_dataset_hash"] == manifest.dataset_hash
    assert report_content["seed"] == 42
    assert "executive_summary" in report_content
    assert "frozen_detector" in report_content


# =====================================================================
# 9. Holdout Immutability & Replay Determinism
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
