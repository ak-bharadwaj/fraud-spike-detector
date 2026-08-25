"""DriftRunner module for executing data drift and regime shift characterization experiments.

Key Invariants:
- Evaluates detector performance under statistical data drift / regime shifts using a frozen detector.
- Dataset boundary: Consumes data/drift/ or development streams; rejects data/holdout/ with ValueError.
- Single-factor drift: Isolates exact changed variable (e.g., volume rate step increase) across control vs drifted runs.
- Frozen detector configuration: Uses FrozenDetectorConfig (threshold=3.5, alpha=0.3, P=2, C=5); zero detector tuning.
- Baseline adaptation measurement: Tracks BaselineEngine adaptation convergence time (adaptation_window_count) under regime shifts.
- Schema compliance: Emits DriftResult validating strictly against Pydantic schema contract.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
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
from src.evaluation.holdout import FrozenDetectorConfig, HoldoutManifest, load_locked_holdout_data


def load_drift_data(data_dir: Union[str, Path] = "data/drift") -> Tuple[HoldoutManifest, List[Transaction], List[GroundTruthEvent]]:
    """Load drift characterization dataset from data/drift/. Raises ValueError if holdout path is supplied."""
    d_path = Path(data_dir)
    if "holdout" in str(d_path).lower():
        raise ValueError("Holdout contamination error: Drift framework cannot consume locked holdout data!")

    return load_locked_holdout_data(d_path)


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
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
        conditions: Optional[List[DriftConditionConfig]] = None,
    ) -> List[DriftResult]:
        """Run control vs drifted regime comparisons for a suite of drift conditions."""
        if conditions is None:
            conditions = self.get_standard_drift_conditions()

        results: List[DriftResult] = []
        for cond in conditions:
            res = self.evaluate_drift_condition(cond, transactions, ground_truth_events)
            results.append(res)

        return results

    def evaluate_drift_condition(
        self,
        condition: DriftConditionConfig,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
    ) -> DriftResult:
        """Run control vs drifted execution for a single drift condition and measure baseline adaptation."""
        # 1. Evaluate Control Stream (pre-drift baseline stream)
        # Partition transactions prior to drift start time for control
        start_time = transactions[0].timestamp if transactions else datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

        drift_onset_time = start_time + timedelta(minutes=condition.start_minute)

        control_txs = [t for t in transactions if t.timestamp < drift_onset_time]
        control_events = [e for e in ground_truth_events if e.start_time < drift_onset_time]

        control_metrics, _, _ = self._run_detector_pipeline(control_txs, control_events)

        # 2. Evaluate Drifted Stream (full stream containing regime shift)
        drifted_metrics, false_alert_count, adaptation_windows = self._run_detector_pipeline(
            transactions, ground_truth_events, drift_onset_time=drift_onset_time
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

                # Measure baseline adaptation after drift onset
                if drift_onset_time and curr_window_start >= drift_onset_time and not is_adapted:
                    adaptation_windows += 1
                    # Convergence check: baseline expected volume adapts to empirical drifted volume rate
                    if base_snap.history_count >= 10 and "volume" in base_snap.expected_values:
                        exp_vol = base_snap.expected_values["volume"]
                        if exp_vol > 0 and abs(feat_snap.volume - exp_vol) / exp_vol < 0.5:
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
