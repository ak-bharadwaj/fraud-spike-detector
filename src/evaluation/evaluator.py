"""AnomalyEvaluator module for deterministic measurement of detector performance against ground truth.

Key Invariants:
- Evaluates detector predictions (Alert list) against GroundTruthEvent list.
- Exact configured anomaly-specific detection horizons (Section 13/14):
    velocity     = 60s
    volume       = 120s
    amount       = 180s
    behavioral   = 180s
    attribute    = 180s
    sustained    = 300s
    compound     = 300s
    evasive      = 300s
- Valid detection interval is:
    GT.start_time <= first valid alert <= GT.start_time + configured horizon
- Pre-onset alerts (alert.timestamp < GT.start_time) are strictly False Positives (FP).
- Alerts past horizon (alert.timestamp > GT.start_time + horizon) are False Positives (FP).
- Ground truth matching is strictly partitioned per merchant_id.
- One-to-one greedy matching rule: One Alert matches at most one GroundTruthEvent; one GroundTruthEvent matches at most one Alert.
- Unmatched alerts -> FP; Unmatched events -> FN.
- Latency calculation: (first_matching_alert.timestamp - event.start_time).total_seconds().
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
import numpy as np

from src.contracts.contracts import Alert, GroundTruthEvent, EvaluationMetrics


DEFAULT_DETECTION_HORIZONS: Dict[str, float] = {
    "velocity": 60.0,
    "volume": 120.0,
    "amount": 180.0,
    "behavioral": 180.0,
    "attribute": 180.0,
    "sustained": 300.0,
    "compound": 300.0,
    "evasive": 300.0,
    "evasion": 300.0,
    "slow_burn": 300.0,
}


def resolve_detection_horizon(
    anomaly_type: str,
    custom_horizons: Optional[Dict[str, float]] = None,
) -> float:
    """Resolve the detection horizon in seconds for a given anomaly type."""
    horizons = custom_horizons or DEFAULT_DETECTION_HORIZONS
    atype = anomaly_type.lower()
    for key, val in horizons.items():
        if key in atype:
            return float(val)
    # Default fallback to 120.0s (volume)
    return float(horizons.get("volume", 120.0))


class AnomalyEvaluator:
    """Evaluates detector Alert outputs against GroundTruthEvent streams using strict one-to-one matching and detection horizons."""

    def __init__(
        self,
        custom_horizons: Optional[Dict[str, float]] = None,
        temporal_tolerance_seconds: float = 0.0,
        fp_unit_cost: float = 50.0,
        fn_unit_exposure: float = 500.0,
    ):
        """Initialize evaluator with detection horizons, optional tolerance, and cost model weights."""
        if temporal_tolerance_seconds < 0.0:
            raise ValueError(f"temporal_tolerance_seconds cannot be negative, got {temporal_tolerance_seconds}")
        if fp_unit_cost < 0.0:
            raise ValueError(f"fp_unit_cost cannot be negative, got {fp_unit_cost}")
        if fn_unit_exposure < 0.0:
            raise ValueError(f"fn_unit_exposure cannot be negative, got {fn_unit_exposure}")

        self.custom_horizons = custom_horizons or dict(DEFAULT_DETECTION_HORIZONS)
        self.temporal_tolerance_seconds = float(temporal_tolerance_seconds)
        self.fp_unit_cost = float(fp_unit_cost)
        self.fn_unit_exposure = float(fn_unit_exposure)

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
        latencies: List[float] = []

        for m_id in sorted(all_merchants):
            m_alerts = sorted(alerts_by_merchant.get(m_id, []), key=lambda a: a.timestamp)
            m_events = sorted(gt_by_merchant.get(m_id, []), key=lambda e: e.start_time)

            # Track matched alert IDs to enforce strict ONE-TO-ONE matching
            used_alert_ids = set()

            for gt in m_events:
                horizon_sec = resolve_detection_horizon(gt.anomaly_type, self.custom_horizons)
                valid_start = gt.start_time.timestamp() - self.temporal_tolerance_seconds
                valid_end = (gt.start_time + timedelta(seconds=horizon_sec)).timestamp() + self.temporal_tolerance_seconds

                # Candidate alerts strictly within valid detection horizon [start_time - tol, start_time + horizon + tol]
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

        # 4. Cost Model
        fp_cost = float(fp * self.fp_unit_cost)
        fn_exposure = float(fn * self.fn_unit_exposure)
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
