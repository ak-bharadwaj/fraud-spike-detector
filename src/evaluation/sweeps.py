"""Development parameter sweep and strategy comparison module.

Key Invariants:
- STRICT DEVELOPMENT DATA ONLY: strictly prohibits access to holdout data paths ('data/holdout/').
- Evaluates complete strategy set:
  1. StaticThresholdScorer
  2. StatisticalDeviationScorer
  3. HybridEWMAScorer
- Complete candidate space (all defaults evaluated simultaneously):
  - alpha: {0.2, 0.3, 0.5, 0.7, 0.9}
  - persistence: {1, 2, 3}
  - threshold: complete operating-point sweep over [1.0, 10.0] with step 0.5 (19 points)
  - signal weights: candidate feature group weight vectors (EQUAL, VOLUME_VELOCITY_HEAVY, AMOUNT_HEAVY, BEHAVIORAL_HEAVY)
  - cooldown: candidate cooldown windows {1, 3, 5, 10}
  - evidence parameters: candidate min_window_count {1, 3, 5, 10}
- Complete metric reporting for all candidates:
  - TP, FP, FN, Precision, Recall, F1, Median Latency, P95 Latency, FP Cost, FN Exposure, Total Cost.
- Selection procedure:
  - Derives winning scorer, alpha, threshold, persistence, cooldown, signal weights, and evidence parameters.
  - Minimizes Total Cost (FP Cost + FN Exposure) with tie-breaking on higher F1 and lower latency.
"""

from typing import List, Dict, Any, Optional, Sequence, Tuple, Union
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import numpy as np

from src.contracts.contracts import (
    Transaction,
    FeatureSnapshot,
    BaselineSnapshot,
    RiskScore,
    Alert,
    GroundTruthEvent,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.detector.pipeline import StreamingDetectorPipeline
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.state.alert_state_machine import AlertStateMachine
from src.scoring.static import StaticThresholdScorer
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.evaluation.evaluator import AnomalyEvaluator

CANDIDATE_SIGNAL_WEIGHTS = {
    "EQUAL": {"volume": 1.0, "velocity": 1.0, "amount": 1.0, "behavioral": 1.0},
    "VOLUME_VELOCITY_HEAVY": {"volume": 1.5, "velocity": 1.5, "amount": 1.0, "behavioral": 1.0},
    "AMOUNT_HEAVY": {"volume": 1.0, "velocity": 1.0, "amount": 1.5, "behavioral": 1.0},
    "BEHAVIORAL_HEAVY": {"volume": 1.0, "velocity": 1.0, "amount": 1.0, "behavioral": 1.5},
}


class HoldoutAccessViolationError(PermissionError):
    """Raised when development parameter sweep attempts to access holdout data."""
    pass


def _verify_development_only_data(data_path: Optional[str] = None) -> None:
    """Ensure data path is not pointing to locked holdout directory."""
    if data_path and "holdout" in str(data_path).lower():
        raise HoldoutAccessViolationError("Parameter sweep is strictly prohibited on holdout data. Use development data only.")


def load_development_data(data_dir: Union[str, Path] = "data/development") -> Tuple[List[Transaction], List[GroundTruthEvent]]:
    """Load canonical development dataset transactions and ground truth events."""
    d_path = Path(data_dir)
    _verify_development_only_data(str(d_path))

    tx_path = d_path / "transactions.json"
    gt_path = d_path / "ground_truth.json"

    if not tx_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Development artifact missing in {d_path}")

    tx_raw = json.loads(tx_path.read_text(encoding="utf-8"))
    transactions = [
        Transaction(
            transaction_id=t["id"],
            timestamp=datetime.fromisoformat(t["ts"]),
            merchant_id=t["m_id"],
            customer_id=t["c_id"],
            amount=float(t["amt"]),
            payment_method=t["pm"],
            country=t["country"],
            device_id=t["d_id"],
        )
        for t in tx_raw
    ]

    gt_raw = json.loads(gt_path.read_text(encoding="utf-8"))
    ground_truth_events = [
        GroundTruthEvent(
            event_id=e["id"],
            merchant_id=e["m_id"],
            anomaly_type=e["type"],
            start_time=datetime.fromisoformat(e["st"]),
            end_time=datetime.fromisoformat(e["et"]),
            severity=float(e["sev"]),
            parameters=e.get("params", {
                "excess_transaction_count": max(1.0, float(round(10.0 * e["sev"]))),
                "mean_transaction_amount": 50.0,
                "exposure_factor": 1.0,
            }),
        )
        for e in gt_raw
    ]

    return transactions, ground_truth_events


def run_strategy_comparison(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compare Static, Statistical, and Hybrid EWMA scoring strategies on development data."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()

    strategies = [
        ("StaticThresholdScorer", StaticThresholdScorer(static_threshold=base_cfg.static_threshold)),
        ("StatisticalDeviationScorer", StatisticalDeviationScorer(static_threshold=base_cfg.static_threshold)),
        ("HybridEWMAScorer", HybridEWMAScorer(alpha=base_cfg.ewma_alpha or 0.3, static_threshold=base_cfg.static_threshold)),
    ]

    results = []
    for strat_name, scorer in strategies:
        pipeline = StreamingDetectorPipeline(config=base_cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "strategy": strat_name,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "metrics": metrics,
        })

    return results


def run_alpha_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    alphas: Sequence[float] = (0.2, 0.3, 0.5, 0.7, 0.9),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over EWMA alpha smoothing factors on development data."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    results = []

    for alpha in alphas:
        scorer = HybridEWMAScorer(alpha=float(alpha), static_threshold=base_cfg.static_threshold)
        pipeline = StreamingDetectorPipeline(config=base_cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "alpha": float(alpha),
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "metrics": metrics,
        })

    return results


def run_persistence_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    persistences: Sequence[int] = (1, 2, 3),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over candidate persistence values on development data."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    results = []

    for p in persistences:
        cfg = base_cfg.model_copy(update={"persistence": int(p)})
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha or 0.3, static_threshold=cfg.static_threshold)
        pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "persistence": int(p),
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "metrics": metrics,
        })

    return results


def run_ewma_precision_latency_tradeoff_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    alphas: Sequence[float] = (0.2, 0.3, 0.5, 0.7, 0.9),
    persistences: Sequence[int] = (1, 2, 3),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute development parameter sweep over EWMA alpha smoothing and persistence grid (15 operating points) (Master Plan §18/§35)."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    results = []

    for alpha in alphas:
        for p in persistences:
            cfg = base_cfg.model_copy(update={"persistence": int(p), "ewma_alpha": float(alpha)})
            scorer = HybridEWMAScorer(alpha=float(alpha), static_threshold=cfg.static_threshold)
            pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
            alerts = pipeline.process_transactions(transactions)
            metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

            results.append({
                "alpha": float(alpha),
                "persistence": int(p),
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
                "precision": float(metrics.precision),
                "recall": float(metrics.recall),
                "f1_score": float(metrics.f1_score),
                "median_latency_seconds": float(metrics.median_latency_seconds) if metrics.median_latency_seconds is not None else None,
                "p95_latency_seconds": float(metrics.p95_latency_seconds) if metrics.p95_latency_seconds is not None else None,
                "fp_cost": float(metrics.fp_cost),
                "fn_exposure": float(metrics.fn_exposure),
                "total_cost": float(metrics.total_cost),
            })

    return results


def run_threshold_operating_point_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    thresholds: Optional[Sequence[float]] = None,
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over static decision threshold operating points [1.0, 10.0] with step 0.5 on development data."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    
    if thresholds is None:
        thresholds = [float(t) for t in np.arange(1.0, 10.5, 0.5)]

    results = []
    for th in thresholds:
        cfg = base_cfg.model_copy(update={"static_threshold": float(th)})
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha or 0.3, static_threshold=cfg.static_threshold)
        pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "threshold": float(th),
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "metrics": metrics,
        })

    return results


def run_cooldown_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    cooldowns: Sequence[int] = (1, 3, 5, 10),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over candidate cooldown windows on development data."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    results = []

    for cd in cooldowns:
        cfg = base_cfg.model_copy(update={"cooldown_windows": int(cd)})
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha or 0.3, static_threshold=cfg.static_threshold)
        pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "cooldown_windows": int(cd),
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "metrics": metrics,
        })

    return results


def run_evidence_parameter_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    min_window_counts: Sequence[int] = (1, 3, 5, 10),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over candidate baseline min_window_count on development data."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    results = []

    for mwc in min_window_counts:
        cfg = base_cfg.model_copy(update={"min_window_count": int(mwc), "min_history_count": int(mwc)})
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha or 0.3, static_threshold=cfg.static_threshold)
        pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "min_window_count": int(mwc),
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "metrics": metrics,
        })

    return results


def run_signal_weight_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    weight_candidates: Optional[Dict[str, Dict[str, float]]] = None,
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
    data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over candidate signal weight configurations on development data."""
    _verify_development_only_data(data_path)
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    candidates = weight_candidates or CANDIDATE_SIGNAL_WEIGHTS
    results = []

    for name, weights in candidates.items():
        scorer = HybridEWMAScorer(alpha=base_cfg.ewma_alpha or 0.3, static_threshold=base_cfg.static_threshold, signal_weights=weights)
        pipeline = StreamingDetectorPipeline(config=base_cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "weight_name": name,
            "signal_weights": weights,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "metrics": metrics,
        })

    return results


def _extract_stream_snapshots(
    transactions: Sequence[Transaction],
    min_window_count: int = 5,
) -> List[Tuple[FeatureSnapshot, BaselineSnapshot]]:
    """Helper to extract (FeatureSnapshot, BaselineSnapshot) sequence for fast sweep evaluation."""
    feature_engine = FeatureEngine()
    baseline_engine = BaselineEngine(min_history_count=min_window_count, min_window_count=min_window_count)

    sorted_txs = sorted(transactions, key=lambda t: t.timestamp)
    if not sorted_txs:
        return []

    merchant_buffers: Dict[str, List[Transaction]] = {}
    merchant_window_starts: Dict[str, datetime] = {}
    snapshots: List[Tuple[FeatureSnapshot, BaselineSnapshot]] = []

    for tx in sorted_txs:
        m_id = tx.merchant_id
        if m_id not in merchant_window_starts:
            merchant_window_starts[m_id] = tx.timestamp.replace(second=0, microsecond=0)
            merchant_buffers[m_id] = []

        w_start = merchant_window_starts[m_id]
        w_end = w_start + timedelta(minutes=1)

        while tx.timestamp >= w_end:
            buf = merchant_buffers.get(m_id, [])
            feat = feature_engine.extract_snapshot(m_id, buf, w_start, w_end)
            base = baseline_engine.get_baseline(m_id, feat)
            snapshots.append((feat, base))
            baseline_engine.update(feat)
            merchant_buffers[m_id] = []
            merchant_window_starts[m_id] = w_end
            w_start = w_end
            w_end = w_start + timedelta(minutes=1)

        merchant_buffers[m_id].append(tx)

    for m_id in sorted(merchant_buffers.keys()):
        buf = merchant_buffers[m_id]
        w_start = merchant_window_starts[m_id]
        w_end = w_start + timedelta(minutes=1)
        feat = feature_engine.extract_snapshot(m_id, buf, w_start, w_end)
        base = baseline_engine.get_baseline(m_id, feat)
        snapshots.append((feat, base))
        baseline_engine.update(feat)

    return snapshots


def select_final_development_configuration(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    thresholds: Optional[Sequence[float]] = None,
    alphas: Sequence[float] = (0.2, 0.3, 0.5, 0.7, 0.9),
    persistences: Sequence[int] = (1, 2, 3),
    cooldowns: Sequence[int] = (1, 3, 5, 10),
    min_window_counts: Sequence[int] = (1, 3, 5, 10),
    weight_candidates: Optional[Dict[str, Dict[str, float]]] = None,
    detector_version: str = "1.1.0",
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Select optimal detector configuration from development grid search across ALL strategies minimizing Total Cost.

    Full Search Space:
    - Scorers:
      1. StaticThresholdScorer
      2. StatisticalDeviationScorer
      3. HybridEWMAScorer
    - Alphas: {0.2, 0.3, 0.5, 0.7, 0.9} (for Hybrid)
    - Persistence: {1, 2, 3}
    - Thresholds: [1.0, 10.0] step 0.5 (19 points)
    - Signal Weights: Equal, Volume/Velocity Heavy, Amount Heavy, Behavioral Heavy (4 vectors)
    - Cooldown: {1, 3, 5, 10} (4 windows)
    - Evidence min_window_count: {1, 3, 5, 10} (4 counts)
    """
    _verify_development_only_data(data_path)
    evaluator = AnomalyEvaluator()

    if thresholds is None:
        threshold_grid = [float(t) for t in np.arange(1.0, 10.5, 0.5)]
    else:
        threshold_grid = [float(t) for t in thresholds]

    weights_dict = weight_candidates or CANDIDATE_SIGNAL_WEIGHTS

    best_candidate: Optional[Dict[str, Any]] = None
    best_score_tuple = (float("inf"), -1.0, float("inf"))  # (total_cost, -f1_score, median_latency)
    all_evaluated: List[Dict[str, Any]] = []

    for mwc in min_window_counts:
        snapshots = _extract_stream_snapshots(transactions, min_window_count=mwc)

        # 1. Search StaticThresholdScorer
        for w_name, weights in weights_dict.items():
            for th in threshold_grid:
                scorer = StaticThresholdScorer(static_threshold=th, signal_weights=weights)
                risk_scores = [
                    (feat.merchant_id, feat.timestamp, scorer.calculate_score(feat, base, signal_weights=weights))
                    for feat, base in snapshots
                ]
                for cd in cooldowns:
                    for p in persistences:
                        sm = AlertStateMachine(persistence=p, cooldown_windows=cd, static_threshold=th)
                        alerts = []
                        for m_id, ts, rs in risk_scores:
                            _, alt = sm.process_score(m_id, ts, rs)
                            if alt is not None:
                                alerts.append(alt)
                        m = evaluator.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

                        tot_cost = m.total_cost if m.total_cost is not None else float("inf")
                        lat = m.median_latency_seconds if m.median_latency_seconds is not None else float("inf")
                        score_tuple = (tot_cost, -m.f1_score, lat)

                        cand = {
                            "selected_scorer": "StaticThresholdScorer",
                            "selected_alpha": None,
                            "selected_threshold": th,
                            "selected_persistence": p,
                            "selected_cooldown": cd,
                            "selected_evidence_params": {"min_window_count": mwc, "min_history_count": mwc},
                            "selected_signal_weights": weights,
                            "all_selected_parameters": {
                                "scorer": "StaticThresholdScorer",
                                "alpha": None,
                                "static_threshold": th,
                                "persistence": p,
                                "cooldown_windows": cd,
                                "min_history_count": mwc,
                                "min_window_count": mwc,
                                "signal_weights": weights,
                                "detector_version": detector_version,
                            },
                            "metrics": m,
                            "score_tuple": score_tuple,
                        }
                        all_evaluated.append(cand)
                        if score_tuple < best_score_tuple or best_candidate is None:
                            best_score_tuple = score_tuple
                            best_candidate = cand

        # 2. Search StatisticalDeviationScorer
        for w_name, weights in weights_dict.items():
            for th in threshold_grid:
                scorer = StatisticalDeviationScorer(static_threshold=th, signal_weights=weights)
                risk_scores = [
                    (feat.merchant_id, feat.timestamp, scorer.calculate_score(feat, base, signal_weights=weights))
                    for feat, base in snapshots
                ]
                for cd in cooldowns:
                    for p in persistences:
                        sm = AlertStateMachine(persistence=p, cooldown_windows=cd, static_threshold=th)
                        alerts = []
                        for m_id, ts, rs in risk_scores:
                            _, alt = sm.process_score(m_id, ts, rs)
                            if alt is not None:
                                alerts.append(alt)
                        m = evaluator.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

                        tot_cost = m.total_cost if m.total_cost is not None else float("inf")
                        lat = m.median_latency_seconds if m.median_latency_seconds is not None else float("inf")
                        score_tuple = (tot_cost, -m.f1_score, lat)

                        cand = {
                            "selected_scorer": "StatisticalDeviationScorer",
                            "selected_alpha": None,
                            "selected_threshold": th,
                            "selected_persistence": p,
                            "selected_cooldown": cd,
                            "selected_evidence_params": {"min_window_count": mwc, "min_history_count": mwc},
                            "selected_signal_weights": weights,
                            "all_selected_parameters": {
                                "scorer": "StatisticalDeviationScorer",
                                "alpha": None,
                                "static_threshold": th,
                                "persistence": p,
                                "cooldown_windows": cd,
                                "min_history_count": mwc,
                                "min_window_count": mwc,
                                "signal_weights": weights,
                                "detector_version": detector_version,
                            },
                            "metrics": m,
                            "score_tuple": score_tuple,
                        }
                        all_evaluated.append(cand)
                        if score_tuple < best_score_tuple or best_candidate is None:
                            best_score_tuple = score_tuple
                            best_candidate = cand

        # 3. Search HybridEWMAScorer
        for w_name, weights in weights_dict.items():
            for alpha in alphas:
                for th in threshold_grid:
                    scorer = HybridEWMAScorer(alpha=alpha, static_threshold=th, signal_weights=weights)
                    risk_scores = [
                        (feat.merchant_id, feat.timestamp, scorer.calculate_score(feat, base, signal_weights=weights))
                        for feat, base in snapshots
                    ]
                    for cd in cooldowns:
                        for p in persistences:
                            sm = AlertStateMachine(persistence=p, cooldown_windows=cd, static_threshold=th)
                            alerts = []
                            for m_id, ts, rs in risk_scores:
                                _, alt = sm.process_score(m_id, ts, rs)
                                if alt is not None:
                                    alerts.append(alt)
                            m = evaluator.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

                            tot_cost = m.total_cost if m.total_cost is not None else float("inf")
                            lat = m.median_latency_seconds if m.median_latency_seconds is not None else float("inf")
                            score_tuple = (tot_cost, -m.f1_score, lat)

                            cand = {
                                "selected_scorer": "HybridEWMAScorer",
                                "selected_alpha": alpha,
                                "selected_threshold": th,
                                "selected_persistence": p,
                                "selected_cooldown": cd,
                                "selected_evidence_params": {"min_window_count": mwc, "min_history_count": mwc},
                                "selected_signal_weights": weights,
                                "all_selected_parameters": {
                                    "scorer": "HybridEWMAScorer",
                                    "alpha": alpha,
                                    "static_threshold": th,
                                    "persistence": p,
                                    "cooldown_windows": cd,
                                    "min_history_count": mwc,
                                    "min_window_count": mwc,
                                    "signal_weights": weights,
                                    "detector_version": detector_version,
                                },
                                "metrics": m,
                                "score_tuple": score_tuple,
                            }
                            all_evaluated.append(cand)
                            if score_tuple < best_score_tuple or best_candidate is None:
                                best_score_tuple = score_tuple
                                best_candidate = cand

    if best_candidate is not None:
        best_candidate["all_evaluated_candidates"] = all_evaluated

    return best_candidate or {}
