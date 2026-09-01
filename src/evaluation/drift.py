"""Drift characterization module for paired control-vs-drift evaluation and baseline adaptation measurement.

Key Invariants:
- Evaluates detector robustness against distribution drift (e.g. organic baseline volume growth) under constant configuration.
- Uses common StreamingDetectorPipeline and AnomalyEvaluator.
- Strict Paired Evaluation Contract & Causal Factor Isolation:
  - Validates declared drift factor against frozen supported set (baseline_volume_growth, organic_rate_drift).
  - Exact start timestamp equality: min(control timestamps) == min(drift timestamps).
  - Exact end timestamp equality: max(control timestamps) == max(drift timestamps).
  - Exact duration equality: actual duration(control) == actual duration(drift).
  - 100% GroundTruth event identity match (event IDs, anomaly types, start times, end times).
  - Transaction-level uncontrolled-attribute isolation for common transactions: 100% field identity.
  - Rigorous distribution-level isolation for newly added growth transactions:
    - Customer IDs must belong strictly to canonical merchant customer pool (1..legit_customer_pool_size) with unskewed distribution.
    - Device IDs must belong strictly to canonical merchant device pool (1..legit_device_pool_size) with unskewed distribution.
    - Country and payment distributions must conform to canonical merchant profile within statistical tolerance (TVD <= 0.20).
    - Amount distribution must preserve moments (mean, std, median) within statistical tolerance.
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

from typing import List, Dict, Any, Optional, Sequence, Tuple, Set
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta, timezone
import numpy as np
from pydantic import BaseModel, Field

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.generator.archetypes import MerchantProfile, create_merchant_profile
from src.detector.pipeline import StreamingDetectorPipeline
from src.evaluation.evaluator import AnomalyEvaluator

ALLOWED_DRIFT_FACTORS: Set[str] = {"baseline_volume_growth", "organic_rate_drift"}


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
        merchant_profile: Optional[MerchantProfile] = None,
    ) -> None:
        """Enforce strict pairing contract between control and drift inputs."""
        # 1. Validate Declared Drift Factor
        if declared_drift_factor not in ALLOWED_DRIFT_FACTORS:
            raise ValueError(
                f"Paired drift contract violation: unsupported declared_drift_factor '{declared_drift_factor}'. "
                f"Allowed factors: {sorted(ALLOWED_DRIFT_FACTORS)}"
            )

        if not control_transactions:
            raise ValueError("Paired drift contract violation: control_transactions is empty")
        if not drift_transactions:
            raise ValueError("Paired drift contract violation: drift_transactions is empty")

        # 2. Validate merchant identity
        ctrl_merchants = {t.merchant_id for t in control_transactions}
        drift_merchants = {t.merchant_id for t in drift_transactions}
        if merchant_id not in ctrl_merchants:
            raise ValueError(f"Paired drift contract violation: merchant '{merchant_id}' not found in control stream")
        if merchant_id not in drift_merchants:
            raise ValueError(f"Paired drift contract violation: merchant '{merchant_id}' not found in drift stream")
        if ctrl_merchants != drift_merchants:
            raise ValueError(f"Paired drift contract violation: merchant sets differ (control: {ctrl_merchants}, drift: {drift_merchants})")

        # 3. Exact Time Bounds & Exact Duration Equality (without minute rounding)
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

        # 4. Validate GroundTruth event specifications
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

        # 5. Transaction-Level Uncontrolled-Attribute Isolation for Common Transactions
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

        # 6. Rigorous Distribution-Level Validation for Growth Transactions
        growth_txs = [t for t in drift_transactions if t.transaction_id not in ctrl_tx_map]
        if growth_txs:
            # Resolve canonical merchant profile
            prof = merchant_profile or create_merchant_profile(42, merchant_id, "stable")

            # A. Canonical Customer Pool & Distribution Validation
            valid_cust_ids = {f"CUST-{i}" for i in range(1, prof.legit_customer_pool_size + 1)}
            for gt in growth_txs:
                if gt.customer_id not in valid_cust_ids:
                    raise ValueError(
                        f"Paired drift contract violation: customer_id '{gt.customer_id}' is outside canonical pool "
                        f"for merchant '{merchant_id}' (expected CUST-1..CUST-{prof.legit_customer_pool_size})"
                    )

            if len(growth_txs) >= 10 and prof.legit_customer_pool_size > 1:
                cust_counts = Counter(t.customer_id for t in growth_txs)
                max_cust_share = max(cust_counts.values()) / float(len(growth_txs))
                if max_cust_share > 0.60:
                    raise ValueError(
                        f"Paired drift contract violation: customer distribution is strongly skewed "
                        f"(max customer share {max_cust_share:.2%} > 60%)"
                    )

            # B. Canonical Device Pool & Distribution Validation
            valid_dev_ids = {f"DEV-{i}" for i in range(1, prof.legit_device_pool_size + 1)}
            for gt in growth_txs:
                if gt.device_id not in valid_dev_ids:
                    raise ValueError(
                        f"Paired drift contract violation: device_id '{gt.device_id}' is outside canonical pool "
                        f"for merchant '{merchant_id}' (expected DEV-1..DEV-{prof.legit_device_pool_size})"
                    )

            if len(growth_txs) >= 10 and prof.legit_device_pool_size > 1:
                dev_counts = Counter(t.device_id for t in growth_txs)
                max_dev_share = max(dev_counts.values()) / float(len(growth_txs))
                if max_dev_share > 0.60:
                    raise ValueError(
                        f"Paired drift contract violation: device distribution is strongly skewed "
                        f"(max device share {max_dev_share:.2%} > 60%)"
                    )

            # C. Canonical Country Distribution Validation
            ctrl_countries = {t.country for t in control_transactions}
            for gt in growth_txs:
                if gt.country not in ctrl_countries:
                    raise ValueError(
                        f"Paired drift contract violation: uncontrolled country '{gt.country}' in growth stream "
                        f"(not present in canonical country set {ctrl_countries})"
                    )

            high_risk_ratio = len([t for t in growth_txs if t.country == "HIGH_RISK_GEO"]) / float(len(growth_txs))
            if abs(high_risk_ratio - prof.p_high_risk_country) > 0.10:
                raise ValueError(
                    f"Paired drift contract violation: country distribution deviates from canonical profile "
                    f"({high_risk_ratio:.2%} vs expected {prof.p_high_risk_country:.2%})"
                )

            # D. Canonical Payment Distribution Validation (Total Variation Distance)
            ctrl_payments = {t.payment_method for t in control_transactions}
            for gt in growth_txs:
                if gt.payment_method not in ctrl_payments:
                    raise ValueError(
                        f"Paired drift contract violation: uncontrolled payment method '{gt.payment_method}' "
                        f"(not present in canonical payment set {ctrl_payments})"
                    )

            pay_counts = Counter(t.payment_method for t in growth_txs)
            p_prep = pay_counts.get("PREPAID_CARD", 0) / float(len(growth_txs))
            p_deb = pay_counts.get("DEBIT_CARD", 0) / float(len(growth_txs))
            p_cred = pay_counts.get("CREDIT_CARD", 0) / float(len(growth_txs))

            q_prep = prof.p_prepaid_payment
            q_deb = prof.p_debit_payment
            q_cred = max(0.0, 1.0 - q_prep - q_deb)

            tvd_payment = 0.5 * (abs(p_prep - q_prep) + abs(p_deb - q_deb) + abs(p_cred - q_cred))
            if tvd_payment > 0.20:
                raise ValueError(
                    f"Paired drift contract violation: payment distribution deviates from canonical profile "
                    f"(Total Variation Distance {tvd_payment:.3f} > 0.20)"
                )

            # E. Moment-Based Amount Distribution Validation (Mean, Std, Median)
            ctrl_amts = [t.amount for t in control_transactions]
            growth_amts = [t.amount for t in growth_txs]

            mean_ctrl, mean_growth = float(np.mean(ctrl_amts)), float(np.mean(growth_amts))
            std_ctrl, std_growth = float(np.std(ctrl_amts)), float(np.std(growth_amts))
            med_ctrl, med_growth = float(np.median(ctrl_amts)), float(np.median(growth_amts))

            mean_shift = abs(mean_growth - mean_ctrl) / max(1.0, mean_ctrl)
            std_shift = abs(std_growth - std_ctrl) / max(1.0, std_ctrl)
            med_shift = abs(med_growth - med_ctrl) / max(1.0, med_ctrl)

            if mean_shift > 0.15 or std_shift > 0.30 or med_shift > 0.20:
                raise ValueError(
                    f"Paired drift contract violation: newly added growth transactions exhibit uncontrolled amount distribution shift "
                    f"(mean shift {mean_shift:.2%}, std shift {std_shift:.2%}, median shift {med_shift:.2%})"
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
        merchant_profile: Optional[MerchantProfile] = None,
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
            merchant_profile=merchant_profile,
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
