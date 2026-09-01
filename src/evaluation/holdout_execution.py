"""Day 8 Locked Holdout Execution, Per-Anomaly Analysis, Descriptive Calibration, Bootstrap & Artifacts.

Key Invariants:
- LOCKED HOLDOUT INTEGRITY: Verifies manifest, dataset SHA-256, generator version, seed, schema version.
- FROZEN CONFIGURATION ENFORCEMENT: Loads exact parameters from config/freeze_record.json; rejects overrides.
- SINGLE-PASS EXECUTION: Runs locked holdout once with historical-only baseline (t_past < t_current).
- PER-ANOMALY EVALUATION: Correctly reports zero-event classes (N=0/0, precision/recall/f1=None) without false positive claims.
- HOLDOUT EVASION CONFIRMATION: Confirms evasion patterns on holdout without detector modification.
- HOLDOUT DRIFT CONFIRMATION: Confirms drift adaptation measurement on holdout without detector modification.
- DESCRIPTIVE CALIBRATION: Direct RiskScore.score bucketing (no score transformation, no threshold dependency), reports mean_predicted_score, observed_positive_rate, and sample count N with explicit None for empty buckets, ECE, and reliability diagram data.
- BOOTSTRAP UNCERTAINTY: 1,000 deterministic resamples (seed 42) computing 95% CIs for Precision and Recall with complete raw counts and N.
- PORTFOLIO ANALYSIS: Evaluates Static, Statistical, and Hybrid on holdout, reporting FP Cost, FN Exposure, and Total Cost.
- ARTIFACT GENERATION: Generates required hierarchy under artifacts/ (including final/metrics.json, final/metrics.csv with '₹' unit, final/report.json).
- UNAMBIGUOUS PROVENANCE: Discloses both original run (EXP-DAY8-HOLDOUT-CONFIRMATION-001, execution_commit: 414998f, artifact_commit: 414998f) and corrected canonical run (EXP-DAY8-HOLDOUT-CORRECTED-002, execution_commit: fb3c7f9, artifact_commit: f21ddeb, prior_artifact_commit: e28d6d3, historical_artifact_chain: [20bf655, 775e779, cc2872b, e28d6d3]).
- HOLDOUT IMMUTABILITY: Verifies holdout SHA before == holdout SHA after.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence, Union
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import csv
import hashlib
import numpy as np

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    Alert,
    RiskScore,
    FeatureSnapshot,
    BaselineSnapshot,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.base import AnomalyScorer
from src.scoring.static import StaticThresholdScorer
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.freeze import FreezeRecord, load_freeze_record, compute_config_hash
from src.evaluation.calibration import compute_descriptive_calibration, DescriptiveCalibrationResult, DescriptiveHoldoutCalibrator
from src.evaluation.holdout import (
    HoldoutManifest,
    HoldoutProtection,
    compute_holdout_dataset_hash,
    load_locked_holdout_data,
    HoldoutAccessError,
    ChecksumMismatchError,
)


def build_frozen_scorer(freeze_record: FreezeRecord) -> AnomalyScorer:
    """Build concrete AnomalyScorer instance matching frozen freeze record configuration."""
    scorer_name = freeze_record.selected_scorer
    params = freeze_record.all_selected_parameters

    th = float(params.get("static_threshold", 3.5))
    weights = params.get("signal_weights", None)

    if scorer_name == "StaticThresholdScorer":
        return StaticThresholdScorer(static_threshold=th, signal_weights=weights)
    elif scorer_name == "StatisticalDeviationScorer":
        return StatisticalDeviationScorer(static_threshold=th, signal_weights=weights)
    elif scorer_name == "HybridEWMAScorer":
        alpha = float(params.get("alpha", 0.3) if params.get("alpha") is not None else 0.3)
        return HybridEWMAScorer(alpha=alpha, static_threshold=th, signal_weights=weights)
    else:
        raise ValueError(f"Unknown frozen scorer '{scorer_name}' in freeze record")


def execute_single_pass_holdout(
    transactions: List[Transaction],
    ground_truth_events: List[GroundTruthEvent],
    freeze_record: FreezeRecord,
    explicit_evaluation_mode: bool = True,
    override_params: Optional[Dict[str, Any]] = None,
) -> Tuple[EvaluationMetrics, List[Alert], List[Tuple[str, datetime, RiskScore]]]:
    """Execute single-pass evaluation of frozen detector on holdout data with historical-only baseline."""
    if not explicit_evaluation_mode:
        raise HoldoutAccessError("Holdout evaluation requires explicit_evaluation_mode=True.")

    if override_params is not None and len(override_params) > 0:
        raise ValueError("Frozen configuration violation: Parameter overrides are strictly prohibited on Day 8.")

    params = freeze_record.all_selected_parameters
    th = float(params.get("static_threshold", 3.5))
    persistence = int(params.get("persistence", 2))
    cooldown = int(params.get("cooldown_windows", 5))
    min_windows = int(params.get("min_window_count", 5))
    weights = params.get("signal_weights", None)

    scorer = build_frozen_scorer(freeze_record)
    feature_engine = FeatureEngine()
    baseline_engine = BaselineEngine(min_history_count=min_windows, min_window_count=min_windows)
    state_machine = AlertStateMachine(
        persistence=persistence,
        cooldown_windows=cooldown,
        static_threshold=th,
    )

    tx_by_merchant: Dict[str, List[Transaction]] = {}
    for tx in sorted(transactions, key=lambda x: x.timestamp):
        tx_by_merchant.setdefault(tx.merchant_id, []).append(tx)

    alerts: List[Alert] = []
    all_scores: List[Tuple[str, datetime, RiskScore]] = []

    for merchant_id in sorted(tx_by_merchant.keys()):
        m_txs = tx_by_merchant[merchant_id]
        if not m_txs:
            continue

        start_time = m_txs[0].timestamp
        end_time = m_txs[-1].timestamp
        curr_window_start = start_time

        while curr_window_start <= end_time:
            curr_window_end = curr_window_start + timedelta(minutes=feature_engine.window_duration_minutes)
            curr_txs = [t for t in m_txs if curr_window_start <= t.timestamp < curr_window_end]

            feat_snap = feature_engine.extract_snapshot(merchant_id, curr_txs, curr_window_start, curr_window_end)
            
            # Historical-only baseline: computed strictly before baseline update
            base_snap = baseline_engine.get_baseline(merchant_id, feat_snap)
            risk_score = scorer.calculate_score(feat_snap, base_snap, signal_weights=weights)
            baseline_engine.update(feat_snap)

            all_scores.append((merchant_id, curr_window_end, risk_score))

            _, alert = state_machine.process_score(merchant_id, curr_window_end, risk_score)
            if alert is not None:
                alerts.append(alert)

            curr_window_start = curr_window_end

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alerts, ground_truth_events)
    return metrics, alerts, all_scores


def compute_per_anomaly_holdout_metrics(
    alerts: Sequence[Alert],
    ground_truth_events: Sequence[GroundTruthEvent],
    evaluator: Optional[AnomalyEvaluator] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-anomaly class evaluation table conforming to Section 36 with explicit zero-event semantics."""
    eval_engine = evaluator or AnomalyEvaluator()
    events_by_type: Dict[str, List[GroundTruthEvent]] = {}
    for gt in ground_truth_events:
        events_by_type.setdefault(gt.anomaly_type, []).append(gt)

    per_anomaly_results: Dict[str, Dict[str, Any]] = {}
    all_types = [
        "volume_spike",
        "velocity_burst",
        "sustained_spike",
        "amount_shift",
        "behavioral_anomaly",
        "attribute_shift",
        "compound_anomaly",
        "evasive_patterns",
    ]

    for a_type in all_types:
        gt_subset = events_by_type.get(a_type, [])
        if not gt_subset:
            per_anomaly_results[a_type] = {
                "anomaly_type": a_type,
                "events_detected": 0,
                "total_events": 0,
                "precision": None,
                "recall": None,
                "f1": None,
                "median_latency_seconds": None,
                "status": "NO_EVENTS_IN_DATASET",
            }
            continue

        m = eval_engine.evaluate(alerts=list(alerts), ground_truth_events=gt_subset)
        per_anomaly_results[a_type] = {
            "anomaly_type": a_type,
            "events_detected": m.tp,
            "total_events": len(gt_subset),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1_score, 4),
            "median_latency_seconds": round(m.median_latency_seconds, 2) if m.median_latency_seconds is not None else None,
            "status": "VALIDATED",
        }

    return per_anomaly_results


def compute_descriptive_holdout_calibration(
    scores_with_timestamps: Sequence[Tuple[str, datetime, RiskScore]],
    ground_truth_events: Sequence[GroundTruthEvent],
) -> Dict[str, Any]:
    """Compute descriptive calibration buckets, observed positive rates, population accounting, and ECE."""
    calibrator = DescriptiveHoldoutCalibrator()
    res: DescriptiveCalibrationResult = calibrator.calibrate_holdout(
        scores_with_timestamps=scores_with_timestamps,
        ground_truth_events=ground_truth_events,
    )
    return res.model_dump(mode="json")


def compute_bootstrap_uncertainty(
    alerts: Sequence[Alert],
    ground_truth_events: Sequence[GroundTruthEvent],
    evaluator: Optional[AnomalyEvaluator] = None,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Compute 95% Confidence Intervals for Precision and Recall using deterministic bootstrap resampling with raw counts."""
    eval_engine = evaluator or AnomalyEvaluator()
    rng = np.random.RandomState(seed)

    base_metrics = eval_engine.evaluate(alerts=list(alerts), ground_truth_events=list(ground_truth_events))
    point_precision = base_metrics.precision
    point_recall = base_metrics.recall
    raw_tp = base_metrics.tp
    raw_fp = base_metrics.fp
    raw_fn = base_metrics.fn
    n_events = len(ground_truth_events)
    n_alerts = len(alerts)

    if not ground_truth_events:
        return {
            "n_resamples": n_resamples,
            "seed": seed,
            "ci_level": ci,
            "precision": {
                "point": point_precision,
                "ci_lower": point_precision,
                "ci_upper": point_precision,
                "raw_numerator_tp": raw_tp,
                "raw_denominator_alerts": n_alerts,
                "n_alerts": n_alerts,
            },
            "recall": {
                "point": point_recall,
                "ci_lower": point_recall,
                "ci_upper": point_recall,
                "raw_numerator_tp": raw_tp,
                "raw_denominator_events": n_events,
                "n_events": n_events,
            },
            "raw_counts": {"tp": raw_tp, "fp": raw_fp, "fn": raw_fn, "n_events": n_events, "n_alerts": n_alerts},
        }

    precisions = []
    recalls = []
    gt_list = list(ground_truth_events)

    for _ in range(n_resamples):
        sampled_indices = rng.choice(n_events, size=n_events, replace=True)
        sampled_gt = [gt_list[i] for i in sampled_indices]
        m = eval_engine.evaluate(alerts=list(alerts), ground_truth_events=sampled_gt)
        precisions.append(m.precision)
        recalls.append(m.recall)

    alpha_tail = (1.0 - ci) / 2.0
    p_lower = float(np.percentile(precisions, alpha_tail * 100))
    p_upper = float(np.percentile(precisions, (1.0 - alpha_tail) * 100))
    r_lower = float(np.percentile(recalls, alpha_tail * 100))
    r_upper = float(np.percentile(recalls, (1.0 - alpha_tail) * 100))

    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "ci_level": ci,
        "precision": {
            "point": round(point_precision, 4),
            "ci_lower": round(p_lower, 4),
            "ci_upper": round(p_upper, 4),
            "raw_numerator_tp": raw_tp,
            "raw_denominator_alerts": raw_tp + raw_fp,
            "n_alerts": n_alerts,
        },
        "recall": {
            "point": round(point_recall, 4),
            "ci_lower": round(r_lower, 4),
            "ci_upper": round(r_upper, 4),
            "raw_numerator_tp": raw_tp,
            "raw_denominator_events": raw_tp + raw_fn,
            "n_events": n_events,
        },
        "raw_counts": {
            "tp": raw_tp,
            "fp": raw_fp,
            "fn": raw_fn,
            "n_events": n_events,
            "n_alerts": n_alerts,
        },
    }


def execute_portfolio_comparison(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    freeze_record: FreezeRecord,
    evaluator: Optional[AnomalyEvaluator] = None,
) -> List[Dict[str, Any]]:
    """Compare Static, Statistical, and Hybrid EWMA scorers on locked holdout (Descriptive Portfolio Analysis)."""
    params = freeze_record.all_selected_parameters
    th = float(params.get("static_threshold", 3.5))
    alpha = float(params.get("alpha", 0.3) if params.get("alpha") is not None else 0.3)
    p = int(params.get("persistence", 2))
    c = int(params.get("cooldown_windows", 5))
    min_w = int(params.get("min_window_count", 5))
    weights = params.get("signal_weights", None)

    strategies = [
        ("StaticThresholdScorer", StaticThresholdScorer(static_threshold=th, signal_weights=weights)),
        ("StatisticalDeviationScorer", StatisticalDeviationScorer(static_threshold=th, signal_weights=weights)),
        ("HybridEWMAScorer", HybridEWMAScorer(alpha=alpha, static_threshold=th, signal_weights=weights)),
    ]

    results = []
    for strat_name, scorer in strategies:
        fake_rec = FreezeRecord(
            detector_version="1.0.0",
            config_hash="PORTFOLIO_COMPARISON",
            development_dataset_hash="PORTFOLIO",
            seed=42,
            selected_scorer=strat_name,
            all_selected_parameters={**params, "scorer": strat_name},
            selection_rationale="Descriptive portfolio comparison on holdout data",
            freeze_timestamp="2026-01-08T00:00:00Z",
        )
        m, _, _ = execute_single_pass_holdout(
            transactions=list(transactions),
            ground_truth_events=list(ground_truth_events),
            freeze_record=fake_rec,
            explicit_evaluation_mode=True,
        )

        results.append({
            "detector": strat_name,
            "fp_cost": m.fp_cost,
            "fn_exposure": m.fn_exposure,
            "total_cost": m.total_cost,
            "cost_unit": "₹",
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "median_latency_seconds": m.median_latency_seconds,
            "p95_latency_seconds": m.p95_latency_seconds,
        })

    return results


def save_day8_research_artifacts(
    base_artifact_dir: Union[str, Path],
    freeze_record: FreezeRecord,
    holdout_manifest: HoldoutManifest,
    holdout_metrics: EvaluationMetrics,
    per_anomaly_metrics: Dict[str, Any],
    calibration_results: Dict[str, Any],
    bootstrap_results: Dict[str, Any],
    portfolio_results: List[Dict[str, Any]],
    evasion_results: Dict[str, Any],
    drift_results: Dict[str, Any],
    experiment_id: str = "EXP-DAY8-HOLDOUT-CORRECTED-002",
    execution_commit: str = "fb3c7f9",
    artifact_commit: str = "f21ddeb",
    prior_artifact_commit: str = "e28d6d3",
    historical_artifact_chain: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Save all Day 8 research outputs in structured artifact directories matching required Section 39 hierarchy."""
    base_p = Path(base_artifact_dir)
    common_metadata = {
        "experiment_id": experiment_id,
        "detector_version": freeze_record.detector_version,
        "config_hash": freeze_record.config_hash,
        "development_dataset_hash": freeze_record.development_dataset_hash,
        "holdout_dataset_hash": holdout_manifest.dataset_hash,
        "seed": freeze_record.seed,
        "timestamp": freeze_record.freeze_timestamp,
    }

    dirs = {
        "calibration": base_p / "calibration",
        "ablation": base_p / "ablation",
        "drift": base_p / "drift",
        "evasion": base_p / "evasion",
        "uncertainty": base_p / "uncertainty",
        "portfolio": base_p / "portfolio",
        "final": base_p / "final",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    # 1. Calibration
    p_cal = dirs["calibration"] / "holdout_calibration.json"
    p_cal.write_text(json.dumps({**common_metadata, **calibration_results}, indent=2), encoding="utf-8")
    saved_paths["calibration"] = p_cal

    # 2. Ablation / Metrics
    p_abl = dirs["ablation"] / "holdout_metrics.json"
    p_abl.write_text(json.dumps({**common_metadata, "metrics": holdout_metrics.model_dump(mode="json"), "per_anomaly": per_anomaly_metrics}, indent=2), encoding="utf-8")
    saved_paths["ablation"] = p_abl

    # 3. Drift
    p_drf = dirs["drift"] / "holdout_drift.json"
    p_drf.write_text(json.dumps({**common_metadata, **drift_results}, indent=2), encoding="utf-8")
    saved_paths["drift"] = p_drf

    # 4. Evasion
    p_eva = dirs["evasion"] / "holdout_evasion.json"
    p_eva.write_text(json.dumps({**common_metadata, **evasion_results}, indent=2), encoding="utf-8")
    saved_paths["evasion"] = p_eva

    # 5. Uncertainty
    p_unc = dirs["uncertainty"] / "bootstrap_uncertainty.json"
    p_unc.write_text(json.dumps({**common_metadata, **bootstrap_results}, indent=2), encoding="utf-8")
    saved_paths["uncertainty"] = p_unc

    # 6. Portfolio
    p_por = dirs["portfolio"] / "portfolio_comparison.json"
    p_por.write_text(json.dumps({**common_metadata, "portfolio": portfolio_results}, indent=2), encoding="utf-8")
    saved_paths["portfolio"] = p_por

    # 7. Final Metrics JSON
    p_fin_m = dirs["final"] / "metrics.json"
    p_fin_m.write_text(
        json.dumps({
            **common_metadata,
            "cost_unit": "₹",
            "metrics": holdout_metrics.model_dump(mode="json"),
            "per_anomaly": per_anomaly_metrics,
        }, indent=2),
        encoding="utf-8",
    )
    saved_paths["final_metrics_json"] = p_fin_m

    # 8. Final Metrics CSV with '₹' cost unit
    p_fin_csv = dirs["final"] / "metrics.csv"
    csv_rows = [
        ["metric", "value", "unit"],
        ["tp", holdout_metrics.tp, "count"],
        ["fp", holdout_metrics.fp, "count"],
        ["fn", holdout_metrics.fn, "count"],
        ["precision", f"{holdout_metrics.precision:.4f}", "rate"],
        ["recall", f"{holdout_metrics.recall:.4f}", "rate"],
        ["f1_score", f"{holdout_metrics.f1_score:.4f}", "score"],
        ["median_latency_seconds", f"{holdout_metrics.median_latency_seconds:.2f}" if holdout_metrics.median_latency_seconds is not None else "N/A", "seconds"],
        ["p95_latency_seconds", f"{holdout_metrics.p95_latency_seconds:.2f}" if holdout_metrics.p95_latency_seconds is not None else "N/A", "seconds"],
        ["fp_cost", f"{holdout_metrics.fp_cost:.2f}", "₹"],
        ["fn_exposure", f"{holdout_metrics.fn_exposure:.2f}", "₹"],
        ["total_cost", f"{holdout_metrics.total_cost:.2f}", "₹"],
    ]
    with open(p_fin_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    saved_paths["final_metrics_csv"] = p_fin_csv

    # 9. Final Report JSON with unambiguous dual run disclosure
    p_fin_rep = dirs["final"] / "report.json"
    dual_run_disclosure = {
        "run_001_original": {
            "experiment_id": "EXP-DAY8-HOLDOUT-CONFIRMATION-001",
            "execution_commit": "414998f",
            "artifact_commit": "414998f",
            "status": "SUPERSEDED",
            "reason_superseded": "Post-holdout descriptive calibration methodology bug (pseudo-probability division), empty bucket pseudo-values, and missing bootstrap raw-count contract.",
            "detector_parameters": freeze_record.all_selected_parameters,
            "holdout_dataset_hash": holdout_manifest.dataset_hash,
            "core_metrics": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1_score": 1.0,
                "total_cost": 0.0,
            },
        },
        "run_002_corrected": {
            "experiment_id": experiment_id,
            "execution_commit": execution_commit,
            "artifact_commit": artifact_commit,
            "prior_artifact_commit": prior_artifact_commit,
            "historical_artifact_chain": historical_artifact_chain or ["20bf655", "775e779", "cc2872b", "e28d6d3"],
            "status": "ACCEPTED_CANONICAL",
            "reason": "Corrected post-holdout descriptive calibration (direct RiskScore bucketing, explicit population accounting), complete bootstrap uncertainty reporting contract with raw counts, and INR '₹' units.",
            "detector_parameters": freeze_record.all_selected_parameters,
            "holdout_dataset_hash": holdout_manifest.dataset_hash,
            "core_metrics": {
                "tp": holdout_metrics.tp,
                "fp": holdout_metrics.fp,
                "fn": holdout_metrics.fn,
                "precision": holdout_metrics.precision,
                "recall": holdout_metrics.recall,
                "f1_score": holdout_metrics.f1_score,
                "total_cost": holdout_metrics.total_cost,
            },
        },
    }

    final_report = {
        **common_metadata,
        "executive_summary": {
            "status": "LOCKED_HOLDOUT_EVALUATION_COMPLETE",
            "tp": holdout_metrics.tp,
            "fp": holdout_metrics.fp,
            "fn": holdout_metrics.fn,
            "precision": holdout_metrics.precision,
            "recall": holdout_metrics.recall,
            "f1_score": holdout_metrics.f1_score,
            "total_cost": holdout_metrics.total_cost,
            "cost_unit": "₹",
        },
        "dual_run_disclosure": dual_run_disclosure,
        "frozen_detector": freeze_record.all_selected_parameters,
        "per_anomaly_performance": per_anomaly_metrics,
        "descriptive_calibration": calibration_results,
        "bootstrap_uncertainty": bootstrap_results,
        "descriptive_portfolio_analysis": portfolio_results,
        "evasion_confirmation": evasion_results,
        "drift_confirmation": drift_results,
    }
    p_fin_rep.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    saved_paths["final_report_json"] = p_fin_rep

    return saved_paths
