"""Drift characterization module for paired control-vs-drift evaluation and baseline adaptation measurement.

Key Invariants:
- Evaluates detector robustness against distribution drift (e.g. organic baseline growth) under constant configuration.
- Uses common StreamingDetectorPipeline and AnomalyEvaluator.
- Paired evaluation: CONTROL (stable) vs DRIFT (growing) with identical seed, duration, and anomaly schedule.
- Warmup exclusion: initial warmup windows (e.g. w < 6) are excluded from adaptation calculation.
- Quantitative adaptation metrics:
  - empirical_post_drift_rate
  - adapted_baseline_rate
  - relative_adaptation_error = |adapted_baseline_rate - empirical_post_drift_rate| / empirical_post_drift_rate
  - convergence_window_count (windows where window_rel_err <= max_relative_error)
- Prohibits access to locked holdout data (development-only).
"""

from typing import List, Dict, Any, Optional, Sequence, Tuple
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from pydantic import BaseModel, Field

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.detector.pipeline import StreamingDetectorPipeline
from src.evaluation.evaluator import AnomalyEvaluator


class DriftResult(BaseModel):
    """Result contract for paired control vs drift characterization experiment."""
    paired_dataset_id: str
    control_metrics: EvaluationMetrics
    drift_metrics: EvaluationMetrics
    metric_deltas: Dict[str, float]
    empirical_post_drift_rate: float
    adapted_baseline_rate: float
    relative_adaptation_error: float
    convergence_window_count: int
    warmup_exclusion_windows: int
    passed_adaptation_criterion: bool


class DriftRunner:
    """Runner for paired control vs drift characterization and quantitative adaptation evaluation."""

    def __init__(
        self,
        config: Optional[FrozenDetectorConfig] = None,
        evaluator: Optional[AnomalyEvaluator] = None,
    ):
        self.config = config or FrozenDetectorConfig()
        self.evaluator = evaluator or AnomalyEvaluator()

    @staticmethod
    def verify_development_only(data_path: Optional[str] = None) -> None:
        """Ensure drift experiment does not reference locked holdout partition."""
        if data_path and "holdout" in str(data_path).lower():
            raise PermissionError("Drift characterization is strictly prohibited on holdout data. Use development data only.")

    def run_paired_drift_experiment(
        self,
        control_transactions: Sequence[Transaction],
        drift_transactions: Sequence[Transaction],
        control_ground_truth: Sequence[GroundTruthEvent],
        drift_ground_truth: Sequence[GroundTruthEvent],
        merchant_id: str = "M_PAIRED",
        warmup_windows: int = 6,
        max_relative_error: float = 0.20,
        min_converged_windows: int = 8,
        unperturbed_end_window: int = 30,
        paired_dataset_id: str = "DEV-DRIFT-PAIR-01",
        data_path: Optional[str] = None,
    ) -> DriftResult:
        """Run paired control vs drift experiment through the common detector pipeline and measure adaptation."""
        self.verify_development_only(data_path)

        # 1. Run Control Stream
        pipe_ctrl = StreamingDetectorPipeline(config=self.config, db_path=":memory:")
        alerts_ctrl = pipe_ctrl.process_transactions(control_transactions)
        metrics_ctrl = self.evaluator.evaluate(alerts_ctrl, list(control_ground_truth))

        # 2. Run Drift Stream
        pipe_drift = StreamingDetectorPipeline(config=self.config, db_path=":memory:")
        alerts_drift = pipe_drift.process_transactions(drift_transactions)
        metrics_drift = self.evaluator.evaluate(alerts_drift, list(drift_ground_truth))

        # 3. Compute Metric Deltas
        lat_ctrl = metrics_ctrl.median_latency_seconds or 0.0
        lat_drift = metrics_drift.median_latency_seconds or 0.0
        deltas = {
            "delta_tp": float(metrics_drift.tp - metrics_ctrl.tp),
            "delta_fp": float(metrics_drift.fp - metrics_ctrl.fp),
            "delta_fn": float(metrics_drift.fn - metrics_ctrl.fn),
            "delta_precision": float(metrics_drift.precision - metrics_ctrl.precision),
            "delta_recall": float(metrics_drift.recall - metrics_ctrl.recall),
            "delta_f1": float(metrics_drift.f1_score - metrics_ctrl.f1_score),
            "delta_median_latency_seconds": float(lat_drift - lat_ctrl),
            "delta_total_cost": float(metrics_drift.total_cost - metrics_ctrl.total_cost),
        }

        # 4. Measure Baseline Adaptation on Post-Warmup Unperturbed Drift Windows
        audits_drift = pipe_drift.audit_store.get_audit_records(merchant_id)
        post_warmup_audits = audits_drift[warmup_windows:unperturbed_end_window]

        converged_count = 0
        emp_rates = []
        exp_rates = []

        for a in post_warmup_audits:
            emp_vol = float(a["features"]["volume"])
            exp_vol = float(a["baseline"]["expected_values"].get("volume", emp_vol))
            win_rel_err = abs(exp_vol - emp_vol) / max(1.0, emp_vol)

            emp_rates.append(emp_vol)
            exp_rates.append(exp_vol)

            if win_rel_err <= max_relative_error:
                converged_count += 1

        mean_emp_rate = float(np.mean(emp_rates)) if emp_rates else 0.0
        mean_exp_rate = float(np.mean(exp_rates)) if exp_rates else 0.0
        overall_rel_error = abs(mean_exp_rate - mean_emp_rate) / max(1.0, mean_emp_rate) if mean_emp_rate > 0 else 0.0

        passed_criterion = converged_count >= min_converged_windows

        return DriftResult(
            paired_dataset_id=paired_dataset_id,
            control_metrics=metrics_ctrl,
            drift_metrics=metrics_drift,
            metric_deltas=deltas,
            empirical_post_drift_rate=mean_emp_rate,
            adapted_baseline_rate=mean_exp_rate,
            relative_adaptation_error=overall_rel_error,
            convergence_window_count=converged_count,
            warmup_exclusion_windows=warmup_windows,
            passed_adaptation_criterion=passed_criterion,
        )
