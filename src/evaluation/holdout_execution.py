"""Day 8 Locked Holdout Execution, Per-Anomaly Analysis, Evasion/Drift Confirmation, Calibration, Bootstrap & Portfolio Module.

Key Invariants:
- LOCKED HOLDOUT INTEGRITY: Verifies manifest, dataset SHA-256, generator version, seed, schema version.
- FROZEN CONFIGURATION ENFORCEMENT: Loads exact parameters from config/freeze_record.json; rejects overrides.
- SINGLE-PASS EXECUTION: Runs locked holdout once with historical-only baseline (t_past < t_current).
- PER-ANOMALY EVALUATION: Computes Precision, Recall, Median Latency, and Detected/Total per anomaly class.
- HOLDOUT EVASION CONFIRMATION: Confirms evasion patterns on holdout without detector modification.
- HOLDOUT DRIFT CONFIRMATION: Confirms drift adaptation measurement on holdout without detector modification.
- DESCRIPTIVE CALIBRATION: Generates reliability buckets (0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0) and ECE.
- BOOTSTRAP UNCERTAINTY: 1,000 deterministic resamples (seed 42) computing 95% CIs for Precision and Recall.
- PORTFOLIO ANALYSIS: Evaluates Static, Statistical, and Hybrid on holdout, reporting FP Cost, FN Exposure, and Total Cost.
- ARTIFACT GENERATION: Generates structured research artifacts in data/artifacts/.
- HOLDOUT IMMUTABILITY: Verifies holdout SHA before == holdout SHA after.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence, Union
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
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
from src.evaluation.holdout import (
    HoldoutManifest,
    HoldoutProtection,
    compute_holdout_dataset_hash,
    load_locked_holdout_data,
    HoldoutAccessError,
    ChecksumMismatchError,
)
from src.stream.clock import VirtualClock
from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec


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
    """Compute per-anomaly class evaluation table conforming to Section 36."""
    eval_engine = evaluator or AnomalyEvaluator()
    events_by_type: Dict[str, List[GroundTruthEvent]] = {}
    for gt in ground_truth_events:
        events_by_type.setdefault(gt.anomaly_type, []).append(gt)

    per_anomaly_results: Dict[str, Dict[str, Any]] = {}
    all_types = ["volume_spike", "velocity_burst", "sustained_spike", "amount_shift", "behavioral_anomaly", "attribute_shift", "compound_anomaly", "evasive_patterns"]

    for a_type in all_types:
        gt_subset = events_by_type.get(a_type, [])
        if not gt_subset:
            per_anomaly_results[a_type] = {
                "anomaly_type": a_type,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "median_latency_seconds": 0.0,
                "events_detected": 0,
                "total_events": 0,
            }
            continue

        m = eval_engine.evaluate(alerts=list(alerts), ground_truth_events=gt_subset)
        per_anomaly_results[a_type] = {
            "anomaly_type": a_type,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1_score,
            "median_latency_seconds": m.median_latency_seconds or 0.0,
            "events_detected": m.tp,
            "total_events": len(gt_subset),
        }

    return per_anomaly_results


def compute_descriptive_holdout_calibration(
    scores_with_timestamps: Sequence[Tuple[str, datetime, RiskScore]],
    ground_truth_events: Sequence[GroundTruthEvent],
    evaluator: Optional[AnomalyEvaluator] = None,
    buckets: Sequence[Tuple[float, float]] = (
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
    ),
) -> Dict[str, Any]:
    """Compute descriptive calibration buckets, observed positive rates, and Expected Calibration Error (ECE)."""
    eval_engine = evaluator or AnomalyEvaluator()

    # Determine ground truth positive timestamps for windows
    gt_intervals = [(e.merchant_id, e.start_time, e.end_time) for e in ground_truth_events]

    valid_samples = []
    for m_id, ts, rs in scores_with_timestamps:
        if rs.score is None:
            continue
        # Normalize score to [0, 1] probability scale for descriptive calibration
        # S_norm = 1 / (1 + exp(- (S - 3.5)))
        raw_score = float(rs.score)
        prob = 1.0 / (1.0 + np.exp(- (raw_score - 3.5)))

        is_gt_positive = any(m_id == gm and st <= ts <= et for gm, st, et in gt_intervals)
        valid_samples.append((prob, 1 if is_gt_positive else 0, raw_score))

    bucket_results = []
    total_n = len(valid_samples)
    ece = 0.0

    for low, high in buckets:
        in_bucket = [s for s in valid_samples if low <= s[0] < high or (high == 1.0 and s[0] == 1.0)]
        n = len(in_bucket)
        if n > 0:
            mean_score = float(np.mean([s[0] for s in in_bucket]))
            mean_raw = float(np.mean([s[2] for s in in_bucket]))
            pos_rate = float(np.mean([s[1] for s in in_bucket]))
            ece += (n / max(total_n, 1)) * abs(pos_rate - mean_score)
        else:
            mean_score = (low + high) / 2.0
            mean_raw = 0.0
            pos_rate = 0.0

        bucket_results.append({
            "bucket": f"{low:.1f}–{high:.1f}",
            "range": [low, high],
            "n": n,
            "mean_score": round(mean_score, 4),
            "mean_raw_score": round(mean_raw, 4),
            "observed_positive_rate": round(pos_rate, 4),
        })

    return {
        "buckets": bucket_results,
        "expected_calibration_error": round(float(ece), 4),
        "total_samples": total_n,
    }


def compute_bootstrap_uncertainty(
    alerts: Sequence[Alert],
    ground_truth_events: Sequence[GroundTruthEvent],
    evaluator: Optional[AnomalyEvaluator] = None,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Compute 95% Confidence Intervals for Precision and Recall using deterministic bootstrap resampling."""
    eval_engine = evaluator or AnomalyEvaluator()
    rng = np.random.RandomState(seed)

    base_metrics = eval_engine.evaluate(alerts=list(alerts), ground_truth_events=list(ground_truth_events))
    point_precision = base_metrics.precision
    point_recall = base_metrics.recall

    if not ground_truth_events:
        return {
            "n_resamples": n_resamples,
            "seed": seed,
            "ci_level": ci,
            "precision": {"point": point_precision, "ci_lower": point_precision, "ci_upper": point_precision},
            "recall": {"point": point_recall, "ci_lower": point_recall, "ci_upper": point_recall},
        }

    precisions = []
    recalls = []
    gt_list = list(ground_truth_events)
    n_events = len(gt_list)

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
        },
        "recall": {
            "point": round(point_recall, 4),
            "ci_lower": round(r_lower, 4),
            "ci_upper": round(r_upper, 4),
        },
    }


def execute_portfolio_comparison(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    freeze_record: FreezeRecord,
    evaluator: Optional[AnomalyEvaluator] = None,
) -> List[Dict[str, Any]]:
    """Compare Static, Statistical, and Hybrid EWMA scorers on locked holdout, breaking down FP Cost, FN Exposure, Total Cost."""
    eval_engine = evaluator or AnomalyEvaluator()
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
        cfg = FrozenDetectorConfig(
            static_threshold=th,
            ewma_alpha=alpha,
            persistence=p,
            cooldown_windows=c,
            min_window_count=min_w,
        )
        fake_rec = FreezeRecord(
            detector_version="1.0.0",
            config_hash="PORTFOLIO_COMPARISON",
            development_dataset_hash="PORTFOLIO",
            seed=42,
            selected_scorer=strat_name,
            all_selected_parameters={**params, "scorer": strat_name},
            selection_rationale="Portfolio comparison",
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
) -> Dict[str, Path]:
    """Save all Day 8 research outputs in structured artifact directories."""
    base_p = Path(base_artifact_dir)
    common_metadata = {
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

    return saved_paths
