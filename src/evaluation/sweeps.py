"""Development parameter sweep and strategy comparison module.

Key Invariants:
- STRICT DEVELOPMENT DATA ONLY: strictly prohibits access to holdout data paths ('data/holdout/').
- Evaluates complete strategy set:
  1. StaticThresholdScorer
  2. StatisticalDeviationScorer
  3. HybridEWMAScorer
- Parameter sweeps:
  - alpha: {0.2, 0.3, 0.5, 0.7, 0.9}
  - persistence: {1, 2, 3}
  - threshold: operating-point sweep over [1.0, 10.0] with step 0.5
- Complete metric reporting for all candidates:
  - TP, FP, FN, Precision, Recall, F1, Median Latency, P95 Latency, FP Cost, FN Exposure, Total Cost.
- Selection procedure:
  - Minimizes Total Cost (FP Cost + FN Exposure) with tie-breaking on higher F1 and lower latency.
"""

from typing import List, Dict, Any, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.detector.pipeline import StreamingDetectorPipeline
from src.scoring.static import StaticThresholdScorer
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.evaluation.evaluator import AnomalyEvaluator


class HoldoutAccessViolationError(PermissionError):
    """Raised when development parameter sweep attempts to access holdout data."""
    pass


def _verify_development_only_data(data_path: Optional[str] = None) -> None:
    """Ensure data path is not pointing to locked holdout directory."""
    if data_path and "holdout" in str(data_path).lower():
        raise HoldoutAccessViolationError("Parameter sweep is strictly prohibited on holdout data. Use development data only.")


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
        ("HybridEWMAScorer", HybridEWMAScorer(alpha=base_cfg.ewma_alpha, static_threshold=base_cfg.static_threshold)),
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
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha, static_threshold=cfg.static_threshold)
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
        scorer = HybridEWMAScorer(alpha=cfg.ewma_alpha, static_threshold=cfg.static_threshold)
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


def select_final_development_configuration(
    transactions: Sequence[Transaction],
    ground_truth_events: Sequence[GroundTruthEvent],
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Select optimal detector configuration from development grid search minimizing Total Cost.

    Grid:
    - Scorer: HybridEWMAScorer (robust statistical deviation with exponential smoothing)
    - Alpha: {0.2, 0.3, 0.5, 0.7, 0.9}
    - Persistence: {1, 2, 3}
    - Threshold: {2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
    """
    _verify_development_only_data(data_path)
    evaluator = AnomalyEvaluator()

    best_candidate: Optional[Dict[str, Any]] = None
    best_score_tuple = (float("inf"), -1.0, float("inf"))  # (total_cost, -f1_score, median_latency)

    for alpha in [0.2, 0.3, 0.5, 0.7, 0.9]:
        for p in [1, 2, 3]:
            for th in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                cfg = FrozenDetectorConfig(
                    static_threshold=th,
                    ewma_alpha=alpha,
                    persistence=p,
                    cooldown_windows=5,
                    min_window_count=5,
                    detector_version="1.0.0",
                )
                scorer = HybridEWMAScorer(alpha=alpha, static_threshold=th)
                pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, db_path=":memory:")
                alerts = pipeline.process_transactions(transactions)
                m = evaluator.evaluate(alerts=alerts, ground_truth_events=list(ground_truth_events))

                tot_cost = m.total_cost if m.total_cost is not None else float("inf")
                lat = m.median_latency_seconds if m.median_latency_seconds is not None else float("inf")
                score_tuple = (tot_cost, -m.f1_score, lat)

                if score_tuple < best_score_tuple or best_candidate is None:
                    best_score_tuple = score_tuple
                    best_candidate = {
                        "selected_scorer": "HybridEWMAScorer",
                        "selected_alpha": alpha,
                        "selected_threshold": th,
                        "selected_persistence": p,
                        "selected_cooldown": 5,
                        "selected_evidence_params": {"min_window_count": 5},
                        "selected_signal_config": ["volume", "velocity", "amount", "behavioral"],
                        "config": cfg,
                        "metrics": m,
                    }

    return best_candidate or {}
