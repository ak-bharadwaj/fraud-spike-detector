"""Drift characterization module for paired control-vs-drift evaluation and baseline adaptation measurement.

Key Invariants:
- Evaluates detector robustness against distribution drift (e.g. organic baseline growth) under constant configuration.
- Uses common StreamingDetectorPipeline and AnomalyEvaluator.
- Strict Paired Evaluation Contract:
  - Validates control and drift streams share identical merchant identity, aligned start/end timestamps, identical total duration.
  - Validates 100% GroundTruth event specifications match (event IDs, anomaly types, start times, end times).
  - Validates drift-factor isolation: checks that uncontrolled attributes (e.g. country, payment method, amount scale) are not contaminated.
  - Rejects mismatched inputs with explicit ValueError before pipeline execution.
- Warmup exclusion: initial warmup windows (e.g. w < 6) are excluded from adaptation calculation.
- Quantitative adaptation metrics:
  - reference_empirical_post_drift_rate (independently calculated from raw unperturbed post-warmup drift transactions)
  - empirical_post_drift_rate (feature snapshot rate)
  - adapted_baseline_rate (BaselineEngine expected volume)
  - relative_adaptation_error = |adapted_baseline_rate - reference_empirical_post_drift_rate| / reference_empirical_post_drift_rate
  - convergence_window_count (windows where window_rel_err <= max_relative_error)
- Prohibits access to locked holdout data (development-only).
"""

from typing import List, Dict, Any, Optional, Sequence, Tuple
from pathlib import Path
from datetime import datetime, timedelta, timezone
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
    reference_empirical_post_drift_rate: float
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

    @staticmethod
    def validate_paired_contract(
        control_transactions: Sequence[Transaction],
        drift_transactions: Sequence[Transaction],
        control_ground_truth: Sequence[GroundTruthEvent],
        drift_ground_truth: Sequence[GroundTruthEvent],
        merchant_id: str,
    ) -> None:
        """Enforce strict pairing contract between control and drift inputs."""
        if not control_transactions:
            raise ValueError("Paired drift contract violation: control_transactions is empty")
        if not drift_transactions:
            raise ValueError("Paired drift contract violation: drift_transactions is empty")

        # 1. Validate merchant identity
        ctrl_merchants = {t.merchant_id for t in control_transactions}
        drift_merchants = {t.merchant_id for t in drift_transactions}
        if merchant_id not in ctrl_merchants:
            raise ValueError(f"Paired drift contract violation: merchant '{merchant_id}' not found in control stream")
        if merchant_id not in drift_merchants:
            raise ValueError(f"Paired drift contract violation: merchant '{merchant_id}' not found in drift stream")
        if ctrl_merchants != drift_merchants:
            raise ValueError(f"Paired drift contract violation: merchant sets differ (control: {ctrl_merchants}, drift: {drift_merchants})")

        # 2. Validate time window alignment & duration
        ctrl_min_ts = min(t.timestamp for t in control_transactions)
        ctrl_max_ts = max(t.timestamp for t in control_transactions)
        drift_min_ts = min(t.timestamp for t in drift_transactions)
        drift_max_ts = max(t.timestamp for t in drift_transactions)

        ctrl_w_start = ctrl_min_ts.replace(second=0, microsecond=0)
        drift_w_start = drift_min_ts.replace(second=0, microsecond=0)
        ctrl_w_end = ctrl_max_ts.replace(second=0, microsecond=0)
        drift_w_end = drift_max_ts.replace(second=0, microsecond=0)

        if ctrl_w_start != drift_w_start:
            raise ValueError(
                f"Paired drift contract violation: start time mismatch "
                f"(control starts at {ctrl_w_start}, drift starts at {drift_w_start})"
            )
        if ctrl_w_end != drift_w_end:
            raise ValueError(
                f"Paired drift contract violation: end time mismatch "
                f"(control ends at {ctrl_w_end}, drift ends at {drift_w_end})"
            )

        ctrl_dur = (ctrl_w_end - ctrl_w_start).total_seconds()
        drift_dur = (drift_w_end - drift_w_start).total_seconds()
        if ctrl_dur != drift_dur:
            raise ValueError(
                f"Paired drift contract violation: duration mismatch "
                f"(control duration {ctrl_dur}s != drift duration {drift_dur}s)"
            )

        # 3. Validate GroundTruth event specifications
        if len(control_ground_truth) != len(drift_ground_truth):
            raise ValueError(
                f"Paired drift contract violation: GroundTruth event count mismatch "
                f"(control has {len(control_ground_truth)}, drift has {len(drift_ground_truth)})"
            )

        ctrl_gt_map = {e.event_id: e for e in control_ground_truth}
        for drift_e in drift_ground_truth:
            if drift_e.event_id not in ctrl_gt_map:
                raise ValueError(f"Paired drift contract violation: GroundTruth event ID '{drift_e.event_id}' missing in control GT")
            ctrl_e = ctrl_gt_map[drift_e.event_id]
            if ctrl_e.merchant_id != drift_e.merchant_id:
                raise ValueError(f"Paired drift contract violation for '{drift_e.event_id}': merchant_id mismatch ({ctrl_e.merchant_id} vs {drift_e.merchant_id})")
            if ctrl_e.anomaly_type != drift_e.anomaly_type:
                raise ValueError(f"Paired drift contract violation for '{drift_e.event_id}': anomaly_type mismatch ({ctrl_e.anomaly_type} vs {drift_e.anomaly_type})")
            if ctrl_e.start_time != drift_e.start_time:
                raise ValueError(f"Paired drift contract violation for '{drift_e.event_id}': start_time mismatch ({ctrl_e.start_time} vs {drift_e.start_time})")
            if ctrl_e.end_time != drift_e.end_time:
                raise ValueError(f"Paired drift contract violation for '{drift_e.event_id}': end_time mismatch ({ctrl_e.end_time} vs {drift_e.end_time})")

        # 4. Validate Drift Factor Isolation (reject uncontrolled attribute shifts)
        ctrl_countries = {t.country for t in control_transactions}
        drift_countries = {t.country for t in drift_transactions}
        if ctrl_countries != drift_countries:
            raise ValueError(f"Paired drift contract violation: uncontrolled country attribute shift ({ctrl_countries} vs {drift_countries})")

        ctrl_payments = {t.payment_method for t in control_transactions}
        drift_payments = {t.payment_method for t in drift_transactions}
        if ctrl_payments != drift_payments:
            raise ValueError(f"Paired drift contract violation: uncontrolled payment method shift ({ctrl_payments} vs {drift_payments})")

        ctrl_mean_amt = float(np.mean([t.amount for t in control_transactions]))
        drift_mean_amt = float(np.mean([t.amount for t in drift_transactions]))
        if ctrl_mean_amt > 0:
            amt_shift = abs(drift_mean_amt - ctrl_mean_amt) / ctrl_mean_amt
            if amt_shift > 0.35:
                raise ValueError(f"Paired drift contract violation: uncontrolled amount distribution shift ({amt_shift:.2%} deviation)")

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
        self.validate_paired_contract(
            control_transactions=control_transactions,
            drift_transactions=drift_transactions,
            control_ground_truth=control_ground_truth,
            drift_ground_truth=drift_ground_truth,
            merchant_id=merchant_id,
        )

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

        # 4. Independently Calculate Post-Drift Reference Rate directly from raw drift transactions
        stream_start_ts = min(t.timestamp for t in drift_transactions).replace(second=0, microsecond=0)
        unperturbed_start_ts = stream_start_ts + timedelta(minutes=warmup_windows)
        unperturbed_end_ts = stream_start_ts + timedelta(minutes=unperturbed_end_window)

        raw_unperturbed_txs = [
            t for t in drift_transactions
            if t.merchant_id == merchant_id and unperturbed_start_ts <= t.timestamp < unperturbed_end_ts
        ]
        unperturbed_duration_min = float(unperturbed_end_window - warmup_windows)
        reference_emp_rate = len(raw_unperturbed_txs) / max(1.0, unperturbed_duration_min)

        # 5. Measure Baseline Adaptation on Post-Warmup Unperturbed Drift Windows
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
        overall_rel_error = abs(mean_exp_rate - reference_emp_rate) / max(1.0, reference_emp_rate) if reference_emp_rate > 0 else 0.0

        passed_criterion = converged_count >= min_converged_windows

        return DriftResult(
            paired_dataset_id=paired_dataset_id,
            control_metrics=metrics_ctrl,
            drift_metrics=metrics_drift,
            metric_deltas=deltas,
            reference_empirical_post_drift_rate=reference_emp_rate,
            empirical_post_drift_rate=mean_emp_rate,
            adapted_baseline_rate=mean_exp_rate,
            relative_adaptation_error=overall_rel_error,
            convergence_window_count=converged_count,
            warmup_exclusion_windows=warmup_windows,
            passed_adaptation_criterion=passed_criterion,
        )
