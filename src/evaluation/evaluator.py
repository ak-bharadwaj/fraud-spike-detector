"""AnomalyEvaluator module for deterministic measurement of detector performance against ground truth.

Key Invariants:
- Evaluates detector predictions (Alert list) against GroundTruthEvent list.
- Driven strictly by authoritative EvaluationConfig (from config/evaluation.yaml or injected config object).
- Exact configured anomaly-specific detection horizons (Section 13/14):
    velocity_burst     = 60s
    volume_spike       = 120s
    amount_shift       = 180s
    behavioral_anomaly = 180s
    attribute_shift    = 180s
    sustained_spike    = 300s
    compound_anomaly   = 300s
    evasive_patterns   = 300s
- Rejects unknown or unconfigured anomaly types with explicit ValueError (no substring guessing or silent fallbacks).
- Valid detection interval is:
    GT.start_time <= first valid alert <= GT.start_time + configured horizon
- Pre-onset alerts (alert.timestamp < GT.start_time) are strictly False Positives (FP).
- Alerts past horizon (alert.timestamp > GT.start_time + horizon) are False Positives (FP).
- Ground truth matching is strictly partitioned per merchant_id.
- One-to-one greedy matching rule: One Alert matches at most one GroundTruthEvent; one GroundTruthEvent matches at most one Alert.
- Unmatched alerts -> FP; Unmatched events -> FN.
- Latency calculation: (first_matching_alert.timestamp - event.start_time).total_seconds().
- Cost Model:
    fp_cost = fp * config.cost_model.fp_review_cost
    fn_exposure = sum(config.cost_model.fn_exposure_factor * event_exposure for unmatched GT)
    total_cost = fp_cost + fn_exposure
- Produces: TP, FP, FN, Precision, Recall, F1, Mean Latency, Median Latency, P95 Latency, FP Cost, FN Exposure, Total Cost.
- Zero-denominator semantics:
  - empty/empty (TP=0, FP=0, FN=0) -> P=1, R=1, F1=1.
  - events/no alerts (TP=0, FP=0, FN>0) -> P=0, R=0, F1=0.
  - alerts/no events (TP=0, FP>0, FN=0) -> P=0, R=1, F1=0.
  - P + R == 0 -> F1 = 0.
- No threshold tuning: Evaluator ONLY measures performance, does NOT modify detector parameters.
- GroundTruth isolation: Ground truth flows ONLY into evaluation, NEVER into detector.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from src.contracts.contracts import Alert, GroundTruthEvent, EvaluationMetrics
from src.contracts.config_schemas import EvaluationConfig, CostModelConfig
from src.contracts.config_loader import load_evaluation_config


ANOMALY_HORIZON_MAP: Dict[str, str] = {
    "velocity_burst": "velocity_burst",
    "velocity_spike": "velocity_burst",
    "velocity": "velocity_burst",
    "volume_spike": "volume_spike",
    "sudden_volume_spike": "volume_spike",
    "surge_volume": "volume_spike",
    "surge": "volume_spike",
    "volume": "volume_spike",
    "amount_shift": "amount_shift",
    "amount_spike": "amount_shift",
    "amount_distribution_shift": "amount_shift",
    "amount": "amount_shift",
    "behavioral_anomaly": "behavioral_anomaly",
    "behavioral_shift": "behavioral_anomaly",
    "device_behavior_anomaly": "behavioral_anomaly",
    "behavioral": "behavioral_anomaly",
    "attribute_shift": "attribute_shift",
    "attribute_anomaly": "attribute_shift",
    "attribute_geographic_shift": "attribute_shift",
    "attribute": "attribute_shift",
    "sustained_spike": "sustained_spike",
    "sustained_anomaly": "sustained_spike",
    "sustained": "sustained_spike",
    "compound_anomaly": "compound_anomaly",
    "compound": "compound_anomaly",
    "evasive_patterns": "evasive_patterns",
    "slow_burn_evasion": "evasive_patterns",
    "slow_burn": "evasive_patterns",
    "evasion": "evasive_patterns",
    "threshold_hugging_evasion": "evasive_patterns",
    "persistence_evasion": "evasive_patterns",
    "staircase_ramp": "evasive_patterns",
    "oscillating_sub_threshold": "evasive_patterns",
}


class AnomalyEvaluator:
    """Evaluates detector Alert outputs against GroundTruthEvent streams using strict one-to-one matching and authoritative EvaluationConfig."""

    def __init__(
        self,
        config: Optional[Union[EvaluationConfig, str, Path]] = None,
        temporal_tolerance_seconds: float = 0.0,
        fp_unit_cost: Optional[float] = None,
        fn_exposure_factor: Optional[float] = None,
    ):
        """Initialize evaluator with authoritative EvaluationConfig and optional overrides."""
        if temporal_tolerance_seconds < 0.0:
            raise ValueError(f"temporal_tolerance_seconds cannot be negative, got {temporal_tolerance_seconds}")

        if config is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "evaluation.yaml"
            self.config = load_evaluation_config(config_path)
        elif isinstance(config, (str, Path)):
            self.config = load_evaluation_config(config)
        elif isinstance(config, EvaluationConfig):
            self.config = config
        else:
            raise TypeError(f"Invalid config type {type(config).__name__}, expected EvaluationConfig, str, or Path")

        self.temporal_tolerance_seconds = float(temporal_tolerance_seconds)
        self.fp_review_cost = float(fp_unit_cost) if fp_unit_cost is not None else float(self.config.cost_model.fp_review_cost)
        self.fn_exposure_factor = float(fn_exposure_factor) if fn_exposure_factor is not None else float(self.config.cost_model.fn_exposure_factor)

    def resolve_horizon(self, anomaly_type: str) -> float:
        """Resolve detection horizon in seconds from configured horizons using exact canonical mapping.

        Raises ValueError for unknown anomaly types.
        """
        raw_key = anomaly_type.strip().lower()
        canon_key = ANOMALY_HORIZON_MAP.get(raw_key)

        if canon_key is None or canon_key not in self.config.horizons:
            # Check direct match in horizons dictionary
            if raw_key in self.config.horizons:
                return float(self.config.horizons[raw_key])
            raise ValueError(
                f"Unknown or unconfigured anomaly type: '{anomaly_type}'. "
                f"Configured evaluation horizons: {sorted(self.config.horizons.keys())}"
            )

        return float(self.config.horizons[canon_key])

    def evaluate(
        self,
        alerts: List[Alert],
        ground_truth_events: List[GroundTruthEvent],
    ) -> EvaluationMetrics:
        """Evaluate alerts against ground_truth_events and return complete EvaluationMetrics."""
        # 1. Group alerts and ground truth by merchant_id
        alerts_by_merchant: Dict[str, List[Alert]] = {}
        for alt in alerts:
            alerts_by_merchant.setdefault(alt.merchant_id, []).append(alt)

        gt_by_merchant: Dict[str, List[GroundTruthEvent]] = {}
        for gt in ground_truth_events:
            gt_by_merchant.setdefault(gt.merchant_id, []).append(gt)

        all_merchants = set(alerts_by_merchant.keys()).union(set(gt_by_merchant.keys()))

        tp = 0
        fp = 0
        fn = 0

        matched_events_details: List[Dict[str, Any]] = []
        unmatched_alerts: List[str] = []
        unmatched_events: List[str] = []
        unmatched_gt_objects: List[GroundTruthEvent] = []
        latencies: List[float] = []

        for m_id in sorted(all_merchants):
            m_alerts = sorted(alerts_by_merchant.get(m_id, []), key=lambda a: a.timestamp)
            m_events = sorted(gt_by_merchant.get(m_id, []), key=lambda e: e.start_time)

            # Track matched alert IDs to enforce strict ONE-TO-ONE matching
            used_alert_ids = set()

            for gt in m_events:
                horizon_sec = self.resolve_horizon(gt.anomaly_type)
                # Pre-onset alerts are strictly False Positives (no tolerance before start_time)
                valid_start = gt.start_time.timestamp()
                valid_end = (gt.start_time + timedelta(seconds=horizon_sec)).timestamp() + self.temporal_tolerance_seconds

                # Candidate alerts strictly within valid detection horizon [start_time, start_time + horizon + tol]
                candidate_alerts = [
                    alt for alt in m_alerts
                    if (alt.alert_id not in used_alert_ids and
                        valid_start <= alt.timestamp.timestamp() <= valid_end)
                ]

                if candidate_alerts:
                    tp += 1
                    first_alt = candidate_alerts[0]  # Earliest available valid alert
                    used_alert_ids.add(first_alt.alert_id)

                    lat_sec = (first_alt.timestamp - gt.start_time).total_seconds()
                    latencies.append(lat_sec)

                    matched_events_details.append({
                        "event_id": gt.event_id,
                        "merchant_id": m_id,
                        "alert_id": first_alt.alert_id,
                        "latency_seconds": lat_sec,
                        "horizon_seconds": horizon_sec,
                        "severity": gt.severity,
                        "severity_level": gt.severity_level,
                    })
                else:
                    fn += 1
                    unmatched_events.append(gt.event_id)
                    unmatched_gt_objects.append(gt)

            # Any alert for this merchant not used in one-to-one matching is a False Positive (FP)
            for alt in m_alerts:
                if alt.alert_id not in used_alert_ids:
                    fp += 1
                    unmatched_alerts.append(alt.alert_id)

        # 2. Calculate Precision, Recall, F1
        if tp + fp == 0:
            precision = 1.0 if fn == 0 else 0.0
        else:
            precision = float(tp / (tp + fp))

        if tp + fn == 0:
            recall = 1.0
        else:
            recall = float(tp / (tp + fn))

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))

        # 3. Latency statistics: Mean, Median, P95
        if latencies:
            lat_arr = np.array(latencies, dtype=np.float64)
            mean_latency = float(np.mean(lat_arr))
            median_latency = float(np.median(lat_arr))
            p95_latency = float(np.percentile(lat_arr, 95))
        else:
            mean_latency = None
            median_latency = None
            p95_latency = None

        # 4. Cost Model (Section 24):
        # FP cost = FP * review_cost
        # FN exposure = sum(excess_transaction_count * mean_transaction_amount * exposure_factor)
        fp_cost = float(fp * self.fp_review_cost)

        fn_exposure = 0.0
        for u_gt in unmatched_gt_objects:
            params = u_gt.parameters or {}
            if "excess_transaction_count" in params and "mean_transaction_amount" in params:
                excess_tx = float(params["excess_transaction_count"])
                mean_amt = float(params["mean_transaction_amount"])
                factor = float(params.get("exposure_factor", self.fn_exposure_factor))
                event_exposure = excess_tx * mean_amt * factor
            elif "excess_tx_count" in params and "mean_amount" in params:
                excess_tx = float(params["excess_tx_count"])
                mean_amt = float(params["mean_amount"])
                factor = float(params.get("exposure_factor", self.fn_exposure_factor))
                event_exposure = excess_tx * mean_amt * factor
            else:
                base_exp = float(params.get("amount_exposure", params.get("exposure", 100.0 * u_gt.severity)))
                event_exposure = float(self.fn_exposure_factor * base_exp)
            fn_exposure += float(event_exposure)

        total_cost = float(fp_cost + fn_exposure)

        return EvaluationMetrics(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=None,  # Event-based detection omits TN
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_latency_seconds=mean_latency,
            median_latency_seconds=median_latency,
            p95_latency_seconds=p95_latency,
            fp_cost=fp_cost,
            fn_exposure=fn_exposure,
            total_cost=total_cost,
            matched_events=matched_events_details,
            unmatched_alerts=unmatched_alerts,
            unmatched_events=unmatched_events,
        )
