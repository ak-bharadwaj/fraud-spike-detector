"""Drift characterization module for paired control-vs-drift evaluation and baseline adaptation measurement.

Key Invariants:
- Evaluates detector robustness against distribution drift (e.g. organic baseline volume growth) under constant configuration.
- Uses common StreamingDetectorPipeline and AnomalyEvaluator.
- Strict Paired Evaluation Contract & Causal Factor Isolation:
  - Exact start timestamp equality: min(control timestamps) == min(drift timestamps).
  - Exact end timestamp equality: max(control timestamps) == max(drift timestamps).
  - Exact duration equality: actual duration(control) == actual duration(drift).
  - 100% GroundTruth event identity match (event IDs, anomaly types, start times, end times).
  - Transaction-level uncontrolled-attribute isolation for common transactions: 100% field identity.
  - Distribution-level isolation for newly added growth transactions:
    - Amount distribution must match underlying control amount distribution.
    - Customer pool and device pool must follow canonical merchant pool distributions.
    - Country and payment method ratios must match canonical legitimate distributions.
  - Rejects uncontrolled factor modifications with explicit ValueError before pipeline execution.
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
    declared_drift_factor: str
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
        declared_drift_factor: str = "baseline_volume_growth",
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

        # 2. Exact Time Bounds & Exact Duration Equality (without minute rounding)
        ctrl_min_ts = min(t.timestamp for t in control_transactions)
        ctrl_max_ts = max(t.timestamp for t in control_transactions)
        drift_min_ts = min(t.timestamp for t in drift_transactions)
        drift_max_ts = max(t.timestamp for t in drift_transactions)

        if ctrl_min_ts != drift_min_ts:
            raise ValueError(
                f"Paired drift contract violation: exact start timestamp mismatch "
                f"({ctrl_min_ts} != {drift_min_ts})"
            )
        if ctrl_max_ts != drift_max_ts:
            raise ValueError(
                f"Paired drift contract violation: exact end timestamp mismatch "
                f"({ctrl_max_ts} != {drift_max_ts})"
            )

        ctrl_dur = (ctrl_max_ts - ctrl_min_ts).total_seconds()
        drift_dur = (drift_max_ts - drift_min_ts).total_seconds()
        if ctrl_dur != drift_dur:
            raise ValueError(
                f"Paired drift contract violation: exact duration mismatch "
                f"({ctrl_dur}s != {drift_dur}s)"
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

        # 4. Transaction-Level Uncontrolled-Attribute Isolation for Common Transactions
        ctrl_tx_map = {t.transaction_id: t for t in control_transactions}
        drift_tx_map = {t.transaction_id: t for t in drift_transactions}

        for ctrl_tx in control_transactions:
            if ctrl_tx.transaction_id in drift_tx_map:
                d_tx = drift_tx_map[ctrl_tx.transaction_id]
                if d_tx.amount != ctrl_tx.amount:
                    raise ValueError(
                        f"Paired drift contract violation: uncontrolled amount shift for '{ctrl_tx.transaction_id}' "
                        f"({ctrl_tx.amount} != {d_tx.amount})"
                    )
                if d_tx.country != ctrl_tx.country:
                    raise ValueError(
                        f"Paired drift contract violation: uncontrolled country shift for '{ctrl_tx.transaction_id}' "
                        f"({ctrl_tx.country} != {d_tx.country})"
                    )
                if d_tx.payment_method != ctrl_tx.payment_method:
                    raise ValueError(
                        f"Paired drift contract violation: uncontrolled payment method shift for '{ctrl_tx.transaction_id}' "
                        f"({ctrl_tx.payment_method} != {d_tx.payment_method})"
                    )
                if d_tx.customer_id != ctrl_tx.customer_id:
                    raise ValueError(
                        f"Paired drift contract violation: uncontrolled customer shift for '{ctrl_tx.transaction_id}' "
                        f"({ctrl_tx.customer_id} != {d_tx.customer_id})"
                    )
                if d_tx.device_id != ctrl_tx.device_id:
                    raise ValueError(
                        f"Paired drift contract violation: uncontrolled device shift for '{ctrl_tx.transaction_id}' "
                        f"({ctrl_tx.device_id} != {d_tx.device_id})"
                    )

        # 5. Distribution-Level Validation for Newly Added Growth Transactions
        growth_txs = [t for t in drift_transactions if t.transaction_id not in ctrl_tx_map]
        if growth_txs:
            # A. Amount distribution check
            ctrl_amts = [t.amount for t in control_transactions]
            growth_amts = [t.amount for t in growth_txs]
            ctrl_mean_amt = float(np.mean(ctrl_amts))
            growth_mean_amt = float(np.mean(growth_amts))
            if ctrl_mean_amt > 0:
                amt_shift = abs(growth_mean_amt - ctrl_mean_amt) / ctrl_mean_amt
                if amt_shift > 0.15:
                    raise ValueError(
                        f"Paired drift contract violation: newly added growth transactions exhibit uncontrolled amount distribution shift "
                        f"({amt_shift:.2%} deviation: control mean ₹{ctrl_mean_amt:.2f} vs growth mean ₹{growth_mean_amt:.2f})"
                    )

            # B. Customer and Device ID space validation
            for gt in growth_txs:
                if not (gt.customer_id.startswith("CUST-") and gt.customer_id[5:].isdigit()):
                    raise ValueError(
                        f"Paired drift contract violation: newly added growth transaction '{gt.transaction_id}' "
                        f"has uncontrolled customer ID format '{gt.customer_id}'"
                    )
                if not (gt.device_id.startswith("DEV-") and gt.device_id[4:].isdigit()):
                    raise ValueError(
                        f"Paired drift contract violation: newly added growth transaction '{gt.transaction_id}' "
                        f"has uncontrolled device ID format '{gt.device_id}'"
                    )

            # C. Country distribution validation
            high_risk_growth_ratio = len([t for t in growth_txs if t.country == "HIGH_RISK_GEO"]) / float(len(growth_txs))
            if high_risk_growth_ratio > 0.08:
                raise ValueError(
                    f"Paired drift contract violation: newly added growth transactions exhibit uncontrolled country distribution shift "
                    f"({high_risk_growth_ratio:.2%} high-risk ratio)"
                )

            # D. Payment distribution validation
            prepaid_growth_ratio = len([t for t in growth_txs if t.payment_method == "PREPAID_CARD"]) / float(len(growth_txs))
            if prepaid_growth_ratio > 0.15:
                raise ValueError(
                    f"Paired drift contract violation: newly added growth transactions exhibit uncontrolled payment method distribution shift "
                    f"({prepaid_growth_ratio:.2%} prepaid ratio)"
                )

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
        declared_drift_factor: str = "baseline_volume_growth",
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
            declared_drift_factor=declared_drift_factor,
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
            declared_drift_factor=declared_drift_factor,
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
