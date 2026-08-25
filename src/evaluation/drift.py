"""DriftRunner module for executing data drift and regime shift characterization experiments.

Key Invariants:
- Evaluates detector performance under statistical data drift / regime shifts using a frozen detector.
- Paired evaluation design: Both Control and Drifted streams process IDENTICAL 120-minute time windows and IDENTICAL GroundTruthEvents (2 GT events).
- Single-factor drift isolation: The ONLY difference between Control and Drifted streams is the 2.5x volume step increase starting at minute 40 in Drifted stream.
- Dataset boundary: Consumes data/drift/ or development streams; rejects data/holdout/ with ValueError.
- Frozen detector configuration: Uses FrozenDetectorConfig (threshold=3.5, alpha=0.3, P=2, C=5); zero detector tuning.
- Baseline adaptation measurement: Tracks BaselineEngine adaptation convergence time (adaptation_window_count) until baseline expected volume converges within <= 20% of empirical drifted volume target.
- Schema compliance: Emits DriftResult validating strictly against Pydantic schema contract.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    Alert,
    EvaluationMetrics,
    DriftConditionConfig,
    DriftResult,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.holdout import FrozenDetectorConfig, HoldoutManifest


def load_drift_data(
    data_dir: Union[str, Path] = "data/drift"
) -> Tuple[HoldoutManifest, List[Transaction], List[Transaction], List[GroundTruthEvent]]:
    """Load paired drift characterization dataset from data/drift/. Raises ValueError if holdout path is supplied."""
    d_path = Path(data_dir)
    if "holdout" in str(d_path).lower():
        raise ValueError("Holdout contamination error: Drift framework cannot consume locked holdout data!")

    manifest_path = d_path / "manifest.json"
    control_path = d_path / "control_transactions.json"
    drifted_path = d_path / "drifted_transactions.json"
    gt_path = d_path / "ground_truth.json"

    if not manifest_path.exists() or not control_path.exists() or not drifted_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Paired drift dataset missing in {d_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = HoldoutManifest(**manifest_data)

    def parse_txs(raw_list):
        return [
            Transaction(
                transaction_id=t["id"],
                timestamp=datetime.fromisoformat(t["ts"]),
                merchant_id=t["m_id"],
                customer_id=t["c_id"],
                amount=t["amt"],
                payment_method=t["pm"],
                country=t["country"],
                device_id=t["d_id"],
            )
            for t in raw_list
        ]

    control_txs = parse_txs(json.loads(control_path.read_text(encoding="utf-8")))
    drifted_txs = parse_txs(json.loads(drifted_path.read_text(encoding="utf-8")))

    gt_raw = json.loads(gt_path.read_text(encoding="utf-8"))
    ground_truth_events = [
        GroundTruthEvent(
            event_id=e["id"],
            merchant_id=e["m_id"],
            anomaly_type=e["type"],
            start_time=datetime.fromisoformat(e["st"]),
            end_time=datetime.fromisoformat(e["et"]),
            severity=e["sev"],
        )
        for e in gt_raw
    ]

    return manifest, control_txs, drifted_txs, ground_truth_events


class DriftRunner:
    """Runner executing controlled data drift characterization experiments."""

    def __init__(
        self,
        config: Optional[FrozenDetectorConfig] = None,
        temporal_tolerance_seconds: float = 0.0,
    ):
        self.config = config if config is not None else FrozenDetectorConfig()
        self.temporal_tolerance_seconds = temporal_tolerance_seconds

    def run_drift_suite(
        self,
        control_transactions: List[Transaction],
        drifted_transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
        conditions: Optional[List[DriftConditionConfig]] = None,
    ) -> List[DriftResult]:
        """Run control vs drifted regime comparisons for paired streams and identical GT events."""
        if conditions is None:
            conditions = self.get_standard_drift_conditions()

        results: List[DriftResult] = []
        for cond in conditions:
            res = self.evaluate_drift_condition(
                cond, control_transactions, drifted_transactions, ground_truth_events
            )
            results.append(res)

        return results

    def evaluate_drift_condition(
        self,
        condition: DriftConditionConfig,
        control_transactions: List[Transaction],
        drifted_transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
    ) -> DriftResult:
        """Run paired control vs drifted execution for identical GT events and measure baseline adaptation."""
        # 1. Evaluate Control Stream (normal volume regime, 2 GT events)
        control_metrics, _, _ = self._run_detector_pipeline(control_transactions, ground_truth_events)

        # 2. Evaluate Drifted Stream (2.5x volume step increase starting at minute 40, identical 2 GT events)
        start_time = (
            drifted_transactions[0].timestamp
            if drifted_transactions
            else datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        )
        drift_onset_time = start_time + timedelta(minutes=condition.start_minute)

        drifted_metrics, false_alert_count, adaptation_windows = self._run_detector_pipeline(
            drifted_transactions, ground_truth_events, drift_onset_time=drift_onset_time, target_magnitude=condition.magnitude
        )

        delta_f1 = drifted_metrics.f1_score - control_metrics.f1_score
        delta_p = drifted_metrics.precision - control_metrics.precision
        delta_r = drifted_metrics.recall - control_metrics.recall

        delta_lat = None
        if drifted_metrics.mean_latency_seconds is not None and control_metrics.mean_latency_seconds is not None:
            delta_lat = drifted_metrics.mean_latency_seconds - control_metrics.mean_latency_seconds

        return DriftResult(
            condition_id=condition.condition_id,
            control_metrics=control_metrics,
            drifted_metrics=drifted_metrics,
            delta_f1=delta_f1,
            delta_precision=delta_p,
            delta_recall=delta_r,
            delta_latency_seconds=delta_lat,
            adaptation_window_count=adaptation_windows,
            false_alert_count=false_alert_count,
        )

    def _run_detector_pipeline(
        self,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
        drift_onset_time: Optional[datetime] = None,
        target_magnitude: float = 1.0,
    ) -> Tuple[EvaluationMetrics, int, int]:
        """Execute frozen detector pipeline and measure false alerts and baseline adaptation convergence."""
        feature_engine = FeatureEngine()
        baseline_engine = BaselineEngine(min_window_count=self.config.min_window_count)
        scorer = HybridEWMAScorer(alpha=self.config.ewma_alpha)
        state_machine = AlertStateMachine(
            persistence=self.config.persistence,
            cooldown_windows=self.config.cooldown_windows,
            static_threshold=self.config.static_threshold,
        )

        tx_by_merchant: Dict[str, List[Transaction]] = {}
        for tx in sorted(transactions, key=lambda x: x.timestamp):
            tx_by_merchant.setdefault(tx.merchant_id, []).append(tx)

        alerts: List[Alert] = []
        adaptation_windows = 0
        is_adapted = False
        baseline_volume_target = None

        for merchant_id in sorted(tx_by_merchant.keys()):
            m_txs = tx_by_merchant[merchant_id]
            if not m_txs:
                continue

            start_time = m_txs[0].timestamp
            end_time = m_txs[-1].timestamp
            curr_window_start = start_time

            while curr_window_start <= end_time:
                curr_window_end = curr_window_start + timedelta(minutes=feature_engine.window_duration_minutes)
                curr_txs = [t for t in m_txs if curr_window_start <= t.timestamp < curr_window_end]

                feat_snap = feature_engine.extract_snapshot(merchant_id, curr_txs, curr_window_start, curr_window_end)
                base_snap = baseline_engine.get_baseline(merchant_id, feat_snap)

                # Track pre-drift baseline volume target for merchant DRIFT_M1
                if drift_onset_time and merchant_id == "DRIFT_M1" and curr_window_start < drift_onset_time:
                    if "volume" in base_snap.expected_values:
                        baseline_volume_target = base_snap.expected_values["volume"] * target_magnitude

                # Measure baseline adaptation convergence after drift onset
                if drift_onset_time and merchant_id == "DRIFT_M1" and curr_window_start >= drift_onset_time and not is_adapted:
                    adaptation_windows += 1
                    if baseline_volume_target and "volume" in base_snap.expected_values:
                        curr_exp_vol = base_snap.expected_values["volume"]
                        # Convergence criterion: baseline expected volume adapts within <= 20% of true drifted regime target
                        if curr_exp_vol > 0 and abs(curr_exp_vol - baseline_volume_target) / baseline_volume_target <= 0.20:
                            is_adapted = True

                risk_score = scorer.calculate_score(feat_snap, base_snap)
                baseline_engine.update(feat_snap)

                _, alert = state_machine.process_score(merchant_id, curr_window_end, risk_score)
                if alert is not None:
                    alerts.append(alert)

                curr_window_start = curr_window_end

        evaluator = AnomalyEvaluator(temporal_tolerance_seconds=self.temporal_tolerance_seconds)
        metrics = evaluator.evaluate(alerts, ground_truth_events)

        false_alert_count = metrics.fp
        return metrics, false_alert_count, adaptation_windows

    @staticmethod
    def get_standard_drift_conditions() -> List[DriftConditionConfig]:
        """Return standard suite of data drift characterization conditions."""
        return [
            DriftConditionConfig(
                condition_id="VOLUME_DRIFT_PROMOTIONAL_REGIME",
                description="Legitimate promotional volume surge (2.5x volume step increase starting at minute 40)",
                changed_factor="volume_rate",
                magnitude=2.5,
                start_minute=40.0,
                duration_minutes=80.0,
            ),
        ]
