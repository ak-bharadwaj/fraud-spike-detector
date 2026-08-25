"""DetectorCalibrator module for calibrating static threshold operating values on development/validation data.

Key Invariants:
- Calibrates static threshold T on calibration/validation dataset (NOT holdout!).
- Structural Holdout Protection: Accepts CalibrationDataset or scores/events; rejects is_holdout=True datasets with HoldoutAccessError.
- Candidate threshold search grid: T in [1.0, 10.0] with step 0.5.
- Calibration mechanism: Replays candidate threshold T through AlertStateMachine using frozen persistence P=2 and cooldown C=5.
- Selection objective: Maximize F1-score evaluated using AnomalyEvaluator.
- Tie-breaking rule: On equal F1-score, choose HIGHER threshold (more conservative, lower FP).
- Minimum evidence rule: Minimum min_samples=10 scores required; if sample_count < 10, retains default_threshold=3.5 with status="INSUFFICIENT_EVIDENCE".
- Upstream component immutability: Calibrator does NOT alter FeatureEngine, BaselineEngine, Scorer, or StateMachine.
- Deterministic: Identical calibration dataset produces 100% identical CalibrationResult.
"""

from typing import List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

from src.contracts.contracts import RiskScore, GroundTruthEvent, Alert, CalibrationResult
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.holdout import HoldoutAccessError
from src.state.alert_state_machine import AlertStateMachine


class CalibrationDataset:
    """Container for calibration dataset streams enforcing holdout protection."""

    def __init__(
        self,
        scores_with_timestamps: List[Tuple[str, datetime, RiskScore]],
        ground_truth_events: List[GroundTruthEvent],
        is_holdout: bool = False,
    ):
        if is_holdout:
            raise HoldoutAccessError("Holdout access denied: Locked holdout datasets cannot be passed for calibration.")
        self.scores_with_timestamps = scores_with_timestamps
        self.ground_truth_events = ground_truth_events
        self.is_holdout = is_holdout


class DetectorCalibrator:
    """Calibrates static risk score threshold using training/validation observations."""

    def __init__(
        self,
        min_samples: int = 10,
        default_threshold: float = 3.5,
        persistence: int = 2,
        cooldown_windows: int = 5,
    ):
        if min_samples < 1:
            raise ValueError(f"min_samples must be positive, got {min_samples}")
        if default_threshold <= 0.0:
            raise ValueError(f"default_threshold must be positive, got {default_threshold}")

        self.min_samples = min_samples
        self.default_threshold = float(default_threshold)
        self.persistence = persistence
        self.cooldown_windows = cooldown_windows

    def calibrate(
        self,
        scores_with_timestamps: List[Tuple[str, datetime, RiskScore]],
        ground_truth_events: List[GroundTruthEvent],
        candidate_thresholds: Optional[List[float]] = None,
        is_holdout: bool = False,
    ) -> CalibrationResult:
        """Calibrate static threshold over candidate_thresholds and return CalibrationResult.

        Raises HoldoutAccessError if is_holdout is True.
        """
        # Structural Holdout Isolation Enforcement
        if is_holdout:
            raise HoldoutAccessError("Holdout access denied: DetectorCalibrator cannot consume locked holdout data.")

        if candidate_thresholds is None:
            candidate_thresholds = [float(t) for t in np.arange(1.0, 10.5, 0.5)]

        sample_count = len(scores_with_timestamps)

        # Handle insufficient evidence edge case
        if sample_count < self.min_samples:
            metrics = self._evaluate_threshold(self.default_threshold, scores_with_timestamps, ground_truth_events)
            return CalibrationResult(
                selected_threshold=self.default_threshold,
                calibrated_f1=metrics.f1_score,
                calibrated_precision=metrics.precision,
                calibrated_recall=metrics.recall,
                sample_count=sample_count,
                status="INSUFFICIENT_EVIDENCE",
            )

        best_threshold = self.default_threshold
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0

        # Sweep candidate thresholds in ascending order
        for th in sorted(candidate_thresholds):
            metrics = self._evaluate_threshold(th, scores_with_timestamps, ground_truth_events)

            # Maximize F1-score. Tie-breaking: choose HIGHER threshold (ge threshold preferred)
            if metrics.f1_score > best_f1 or (metrics.f1_score == best_f1 and th >= best_threshold):
                best_f1 = metrics.f1_score
                best_threshold = th
                best_precision = metrics.precision
                best_recall = metrics.recall

        return CalibrationResult(
            selected_threshold=best_threshold,
            calibrated_f1=best_f1,
            calibrated_precision=best_precision,
            calibrated_recall=best_recall,
            sample_count=sample_count,
            status="SUCCESS",
        )

    def calibrate_dataset(
        self,
        dataset: CalibrationDataset,
        candidate_thresholds: Optional[List[float]] = None,
    ) -> CalibrationResult:
        """Calibrate threshold using a CalibrationDataset object."""
        return self.calibrate(
            scores_with_timestamps=dataset.scores_with_timestamps,
            ground_truth_events=dataset.ground_truth_events,
            candidate_thresholds=candidate_thresholds,
            is_holdout=dataset.is_holdout,
        )

    def _evaluate_threshold(
        self,
        threshold: float,
        scores_with_timestamps: List[Tuple[str, datetime, RiskScore]],
        ground_truth_events: List[GroundTruthEvent],
    ) -> Any:
        """Helper to run AlertStateMachine and AnomalyEvaluator for a specific candidate threshold."""
        sm = AlertStateMachine(
            persistence=self.persistence,
            cooldown_windows=self.cooldown_windows,
            static_threshold=threshold,
        )

        alerts: List[Alert] = []
        for merchant_id, ts, risk_score in scores_with_timestamps:
            _, alert = sm.process_score(merchant_id, ts, risk_score)
            if alert is not None:
                alerts.append(alert)

        evaluator = AnomalyEvaluator()
        return evaluator.evaluate(alerts, ground_truth_events)
