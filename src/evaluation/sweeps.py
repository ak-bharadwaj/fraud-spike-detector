"""Development parameter sweep module for evaluating alpha, persistence, and threshold operating points.

Key Invariants:
- STRICT DEVELOPMENT DATA ONLY: strictly prohibits access to holdout data paths ('data/holdout/').
- Parameter sweeps:
  - alpha: {0.2, 0.3, 0.5, 0.7, 0.9}
  - persistence: {1, 2, 3}
  - threshold: operating-point sweep
- Reports for each sweep point:
  - Precision
  - Recall
  - F1 Score
  - Median Latency
  - P95 Latency
  - FP Cost
  - FN Exposure
  - Total Cost
"""

from typing import List, Dict, Any, Optional, Sequence
from pathlib import Path

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.detector.pipeline import StreamingDetectorPipeline
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.evaluation.evaluator import AnomalyEvaluator


class HoldoutAccessViolationError(PermissionError):
    """Raised when development parameter sweep attempts to access holdout data."""
    pass


def _verify_development_only_data(data_path: Optional[str] = None) -> None:
    """Ensure data path is not pointing to locked holdout directory."""
    if data_path and "holdout" in str(data_path).lower():
        raise HoldoutAccessViolationError("Parameter sweep is strictly prohibited on holdout data. Use development data only.")


def run_alpha_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    alphas: Sequence[float] = (0.2, 0.3, 0.5, 0.7, 0.9),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over EWMA alpha smoothing factors on development data."""
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
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
        })

    return results


def run_persistence_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    persistences: Sequence[int] = (1, 2, 3),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over candidate persistence values on development data."""
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    results = []

    for p in persistences:
        cfg = base_cfg.model_copy(update={"persistence": int(p)})
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha, static_threshold=cfg.static_threshold)
        pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "persistence": int(p),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
        })

    return results


def run_threshold_operating_point_sweep(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    thresholds: Sequence[float] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
    base_config: Optional[FrozenDetectorConfig] = None,
    evaluator: Optional[AnomalyEvaluator] = None,
) -> List[Dict[str, Any]]:
    """Execute parameter sweep over static decision threshold operating points on development data."""
    base_cfg = base_config or FrozenDetectorConfig()
    eval_engine = evaluator or AnomalyEvaluator()
    results = []

    for th in thresholds:
        cfg = base_cfg.model_copy(update={"static_threshold": float(th)})
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha, static_threshold=cfg.static_threshold)
        pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
        alerts = pipeline.process_transactions(transactions)
        metrics: EvaluationMetrics = eval_engine.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

        results.append({
            "threshold": float(th),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "median_latency_seconds": metrics.median_latency_seconds,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "fp_cost": metrics.fp_cost,
            "fn_exposure": metrics.fn_exposure,
            "total_cost": metrics.total_cost,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
        })

    return results
