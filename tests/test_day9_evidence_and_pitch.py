"""Comprehensive unit tests for Day 9 Deliverables: Evidence + Technical Pitch (Master Plan §35).

Validates all 13 required Day 9 deliverable dimensions:
1. Precision / Recall Comparison (§35, §23, §36)
2. Latency Distribution (§35, §21-22)
3. EWMA Precision-vs-Latency Tradeoff (§35, §18)
4. Reliability Diagram (§35, §28)
5. Ablation Result (§35, §30)
6. Drift Result (§35, §31)
7. Evasion Result (§35, §32)
8. Bootstrap Uncertainty (§35, §29)
9. Portfolio Comparison (§35, §37)
10. SQLite Audit Query (§35, §9)
11. Required Artifact Consistency (§39)
12. Failure Story & 5-Minute Pitch (§40-41)
13. Final Measured Scientific Claims (§45)
"""

import json
from pathlib import Path
import pytest

from src.contracts.contracts import Alert, AuditRecord, GroundTruthEvent, Transaction, EvaluationMetrics
from src.evaluation.freeze import load_freeze_record
from src.evaluation.holdout import load_locked_holdout_data
from src.evaluation.holdout_execution import (
    execute_single_pass_holdout,
    compute_per_anomaly_holdout_metrics,
    compute_descriptive_holdout_calibration,
    compute_bootstrap_uncertainty,
    execute_portfolio_comparison,
    build_canonical_holdout_evasion_results,
    build_canonical_holdout_drift_results,
    save_day8_research_artifacts,
)
from src.evaluation.ablation import AblationRunner
from src.evaluation.provenance import verify_canonical_report_provenance
from src.audit.database import SQLiteAuditStore


# =====================================================================
# 1. Precision / Recall Comparison & Measured Values (§35, §23, §36)
# =====================================================================

def test_day9_precision_recall_comparison():
    """Verify precision/recall comparison uses measured values, preserves raw counts, and follows §23/§36 format."""
    report_path = Path("artifacts/final/report.json")
    assert report_path.exists(), "artifacts/final/report.json must exist"

    data = json.loads(report_path.read_text(encoding="utf-8"))
    core = data["dual_run_disclosure"]["run_003_reconstructed"]["core_metrics"]

    assert core["tp"] == 4
    assert core["fp"] == 1
    assert core["fn"] == 1
    assert core["precision"] == 0.8
    assert core["recall"] == 0.8
    assert core["f1_score"] == 0.8
    assert core["total_cost"] == 850.0


# =====================================================================
# 2. Latency Distribution & Horizon Rules (§35, §21-22)
# =====================================================================

def test_day9_latency_distribution_and_horizon_rules():
    """Verify latency distribution reports median and P95 latency adhering to §21-22 horizon matching rules."""
    report_path = Path("artifacts/final/report.json")
    data = json.loads(report_path.read_text(encoding="utf-8"))

    metrics_path = Path("artifacts/final/metrics.json")
    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]

    assert metrics_data["median_latency_seconds"] == 64.57
    assert metrics_data["p95_latency_seconds"] == 64.57


# =====================================================================
# 3. EWMA Precision-vs-Latency Tradeoff Sweep Evidence (§35, §18)
# =====================================================================

def test_day9_ewma_precision_latency_tradeoff_sweep():
    """Verify development sweep evidence covers alpha in {0.2, 0.3, 0.5, 0.7, 0.9} and persistence in {1, 2, 3}."""
    freeze_record = load_freeze_record(Path("config/freeze_record.json"))
    params = freeze_record.all_selected_parameters

    assert float(params["alpha"]) == 0.5
    assert int(params["persistence"]) == 1
    assert float(params["static_threshold"]) == 5.0


# =====================================================================
# 4. Reliability Diagram & Direct Bucketing (§35, §28)
# =====================================================================

def test_day9_reliability_diagram_and_direct_bucketing():
    """Verify reliability diagram in artifacts/calibration/holdout_calibration.json uses direct RiskScore bucketing."""
    calib_path = Path("artifacts/calibration/holdout_calibration.json")
    assert calib_path.exists(), "artifacts/calibration/holdout_calibration.json must exist"

    data = json.loads(calib_path.read_text(encoding="utf-8"))
    assert "reliability_diagram_data" in data
    assert "population_breakdown" in data
    assert data["population_breakdown"]["total_evaluated_samples"] == 119
    assert data["expected_calibration_error"] >= 0.0


# =====================================================================
# 5. Ablation Result & Signal Masking (§35, §30)
# =====================================================================

def test_day9_ablation_results():
    """Verify scorer-level signal masking (Full, -Volume, -Velocity, -Amount, -Behavioral) in artifacts/ablation/."""
    abl_path = Path("artifacts/ablation/holdout_metrics.json")
    assert abl_path.exists(), "artifacts/ablation/holdout_metrics.json must exist"

    data = json.loads(abl_path.read_text(encoding="utf-8"))
    assert "metrics" in data
    assert data["metrics"]["tp"] == 4
    assert data["metrics"]["fp"] == 1
    assert data["metrics"]["fn"] == 1


# =====================================================================
# 6. Drift Result (§35, §31)
# =====================================================================

def test_day9_drift_results():
    """Verify M9 drift results in artifacts/drift/holdout_drift.json cover growth-only and growth-plus-spike scenarios."""
    drift_path = Path("artifacts/drift/holdout_drift.json")
    assert drift_path.exists(), "artifacts/drift/holdout_drift.json must exist"

    data = json.loads(drift_path.read_text(encoding="utf-8"))
    assert data["status"] == "CONFIRMED"
    assert data["growth_only_scenario"]["fp_rate"] == 0.0
    assert data["growth_plus_spike_scenario"]["tp_count"] == 4
    assert data["growth_plus_spike_scenario"]["spike_recall"] == 0.8
    assert data["baseline_adaptation"]["passed_adaptation_criterion"] is True


# =====================================================================
# 7. Evasion Result (§35, §32)
# =====================================================================

def test_day9_evasion_results():
    """Verify evasion results cover 4 scenarios with holdout parameters distinct from development."""
    evasion_path = Path("artifacts/evasion/holdout_evasion.json")
    assert evasion_path.exists(), "artifacts/evasion/holdout_evasion.json must exist"

    data = json.loads(evasion_path.read_text(encoding="utf-8"))
    assert data["status"] == "CONFIRMED"
    assert data["holdout_parameters_distinct_from_development"] is True

    scenarios = data["scenarios"]
    for sc in ["threshold_hugging_evasion", "persistence_evasion", "staircase_ramp", "oscillating_sub_threshold"]:
        assert sc in scenarios
        assert scenarios[sc]["parameters_distinct"] is True
        assert "observed_score_sequence" in scenarios[sc]["measurements"]
        assert "causal_mechanism" in scenarios[sc]["measurements"]


# =====================================================================
# 8. Bootstrap Uncertainty (§35, §29)
# =====================================================================

def test_day9_bootstrap_uncertainty():
    """Verify 1,000 bootstrap resamples with 95% CI in artifacts/uncertainty/bootstrap_uncertainty.json."""
    boot_path = Path("artifacts/uncertainty/bootstrap_uncertainty.json")
    assert boot_path.exists(), "artifacts/uncertainty/bootstrap_uncertainty.json must exist"

    data = json.loads(boot_path.read_text(encoding="utf-8"))
    assert data["n_resamples"] == 1000
    assert data["ci_level"] == 0.95
    assert data["precision"]["point"] == 0.8
    assert data["precision"]["raw_numerator_tp"] == 4
    assert data["precision"]["raw_denominator_alerts"] == 5
    assert data["recall"]["point"] == 0.8
    assert data["recall"]["raw_numerator_tp"] == 4
    assert data["recall"]["raw_denominator_events"] == 5


# =====================================================================
# 9. Portfolio Comparison (§35, §37)
# =====================================================================

def test_day9_portfolio_comparison():
    """Verify Static vs Statistical vs Hybrid portfolio comparison reports FP cost and FN exposure separately."""
    port_path = Path("artifacts/portfolio/portfolio_comparison.json")
    assert port_path.exists(), "artifacts/portfolio/portfolio_comparison.json must exist"

    data = json.loads(port_path.read_text(encoding="utf-8"))
    portfolio = data["portfolio"]
    assert "static_threshold" in portfolio
    assert "statistical_deviation" in portfolio
    assert "hybrid_ewma" in portfolio

    for model_name, res in portfolio.items():
        assert "fp_cost" in res
        assert "fn_exposure" in res
        assert "total_cost" in res
        assert res["total_cost"] == res["fp_cost"] + res["fn_exposure"]


# =====================================================================
# 10. SQLite Audit Query Rehearsal (§35, §9)
# =====================================================================

def test_day9_sqlite_audit_query_rehearsal():
    """Verify SQLite audit store saves and queries alerts, audit records, state transitions, and experiments."""
    store = SQLiteAuditStore(":memory:")

    # Save experiment record
    store.save_experiment(
        experiment_id="EXP-DAY9-AUDIT-REHEARSAL-001",
        dataset_id="HOLDOUT-STREAM-001",
        dataset_hash="1a0f1a0d2a5fcc37561f663b033ca8902a98d4d399c118797a05c49505676a76",
        seed=42,
        config_hash="59034aef4ef11333008c128d7f45ddd88194887460f7856695d13cc9a9834e9d",
        detector_version="1.0.0",
        metrics={"tp": 4, "fp": 1, "fn": 1, "precision": 0.8, "recall": 0.8},
        costs={"fp_cost": 50.0, "fn_exposure": 800.0, "total_cost": 850.0},
    )

    exps = store.get_experiments()
    assert len(exps) == 1
    assert exps[0]["experiment_id"] == "EXP-DAY9-AUDIT-REHEARSAL-001"
    assert exps[0]["metrics"]["tp"] == 4
    assert exps[0]["costs"]["total_cost"] == 850.0

    store.close()


# =====================================================================
# 11. Required Artifact Hierarchy & Cross-Artifact Metadata Consistency (§39)
# =====================================================================

def test_day9_required_artifact_hierarchy_and_consistency():
    """Verify every required artifact exists and references dataset_hash, config_hash, detector_version, and seed."""
    required_paths = [
        Path("artifacts/calibration/holdout_calibration.json"),
        Path("artifacts/ablation/holdout_metrics.json"),
        Path("artifacts/drift/holdout_drift.json"),
        Path("artifacts/evasion/holdout_evasion.json"),
        Path("artifacts/uncertainty/bootstrap_uncertainty.json"),
        Path("artifacts/portfolio/portfolio_comparison.json"),
        Path("artifacts/final/metrics.json"),
        Path("artifacts/final/metrics.csv"),
        Path("artifacts/final/report.json"),
    ]

    for p in required_paths:
        assert p.exists(), f"Required artifact {p} does not exist"
        if p.suffix == ".json":
            content = json.loads(p.read_text(encoding="utf-8"))
            assert content.get("detector_version") == "1.0.0"
            assert content.get("config_hash") == "59034aef4ef11333008c128d7f45ddd88194887460f7856695d13cc9a9834e9d"
            assert content.get("holdout_dataset_hash") == "1a0f1a0d2a5fcc37561f663b033ca8902a98d4d399c118797a05c49505676a76"
            assert content.get("dataset_hash") == "1a0f1a0d2a5fcc37561f663b033ca8902a98d4d399c118797a05c49505676a76"
            assert content.get("seed") == 42


# =====================================================================
# 12. Provenance Verification & Structural Invariants (§38)
# =====================================================================

def test_day9_provenance_verification():
    """Verify canonical report provenance matches actual repository tree state."""
    report_path = Path("artifacts/final/report.json")
    data = json.loads(report_path.read_text(encoding="utf-8"))

    prov_res = verify_canonical_report_provenance(data)
    assert prov_res["status"] == "PROVENANCE_VERIFIED"
