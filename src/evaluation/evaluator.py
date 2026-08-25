"""AnomalyEvaluator module for deterministic measurement of detector performance against ground truth.

Key Invariants:
- Evaluates detector predictions (Alert list) against GroundTruthEvent list.
- Ground truth matching is strictly partitioned per merchant_id.
- Temporal overlap rule: alert matches event if event.start_time <= alert.timestamp <= event.end_time.
- Multiple alerts per event: First matching alert pairs with the GT event (TP=1).
- Unmatched alerts: Alert outside any active GT event interval for that merchant = FP.
- Unmatched events: GT event with zero matching alerts = FN.
- Latency calculation: (first_matching_alert.timestamp - event.start_time).total_seconds().
- Zero-denominator semantics:
  - TP + FP == 0 -> precision = 1.0.
  - TP + FN == 0 -> recall = 1.0.
  - P + R == 0 -> f1_score = 0.0.
- No threshold tuning: Evaluator ONLY measures performance, does NOT modify detector parameters.
- GroundTruth isolation: Ground truth flows ONLY into evaluation, NEVER into detector.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from src.contracts.contracts import Alert, GroundTruthEvent, EvaluationMetrics


class AnomalyEvaluator:
    """Evaluates detector Alert outputs against GroundTruthEvent streams."""

    def __init__(self, temporal_tolerance_seconds: float = 0.0):
        if temporal_tolerance_seconds < 0.0:
            raise ValueError(f"temporal_tolerance_seconds cannot be negative, got {temporal_tolerance_seconds}")
        self.temporal_tolerance_seconds = float(temporal_tolerance_seconds)

    def evaluate(
        self,
        alerts: List[Alert],
        ground_truth_events: List[GroundTruthEvent],
    ) -> EvaluationMetrics:
        """Evaluate alerts against ground_truth_events and return EvaluationMetrics."""
        # 1. Group alerts and ground truth by merchant_id
        alerts_by_merchant: Dict[str, List[Alert]] = {}
        for alt in alerts:
            alerts_by_merchant.setdefault(alt.merchant_id, []).append(alt)

        gt_by_merchant: Dict[str, List[GroundTruthEvent]] = {}
        for gt in ground_truth_events:
            gt_by_merchant.setdefault(gt.merchant_id, []).append(gt)

        # Sort GT events by start_time for deterministic processing
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

            # Track matched status
            matched_alert_ids = set()

            for gt in m_events:
                # Find all alerts that overlap with GT event interval [start_time - tol, end_time + tol]
                matching_alerts = [
                    alt for alt in m_alerts
                    if (gt.start_time.timestamp() - self.temporal_tolerance_seconds
                        <= alt.timestamp.timestamp()
                        <= gt.end_time.timestamp() + self.temporal_tolerance_seconds)
                ]

                if matching_alerts:
                    tp += 1
                    first_alt = matching_alerts[0]
                    matched_alert_ids.add(first_alt.alert_id)

                    # Latency from event start_time to first matching alert
                    lat_sec = (first_alt.timestamp - gt.start_time).total_seconds()
                    latencies.append(lat_sec)

                    matched_events_details.append({
                        "event_id": gt.event_id,
                        "merchant_id": m_id,
                        "alert_id": first_alt.alert_id,
                        "latency_seconds": lat_sec,
                        "severity": gt.severity,
                        "severity_level": gt.severity_level,
                    })
                else:
                    fn += 1
                    unmatched_events.append(gt.event_id)

            # Check for false positive alerts (alerts not falling in any GT event interval for that merchant)
            for alt in m_alerts:
                # Check if alt falls within ANY GT event interval for this merchant
                is_inside_any_gt = any(
                    (gt.start_time.timestamp() - self.temporal_tolerance_seconds
                     <= alt.timestamp.timestamp()
                     <= gt.end_time.timestamp() + self.temporal_tolerance_seconds)
                    for gt in m_events
                )
                if not is_inside_any_gt:
                    fp += 1
                    unmatched_alerts.append(alt.alert_id)

        # 2. Calculate Precision, Recall, F1
        if tp + fp == 0:
            precision = 1.0
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

        mean_latency = float(sum(latencies) / len(latencies)) if latencies else None

        return EvaluationMetrics(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=None,  # Event-based detection omits TN
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_latency_seconds=mean_latency,
            matched_events=matched_events_details,
            unmatched_alerts=unmatched_alerts,
            unmatched_events=unmatched_events,
        )
