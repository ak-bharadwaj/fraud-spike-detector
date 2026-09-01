"""Calibration module for development-time threshold tuning and Day-8 descriptive final-holdout calibration.

Key Invariants:
- Day-8 Descriptive Holdout Calibration:
  - Generates empirical statistics across score buckets [0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0].
  - For populated buckets: reports empirical mean_score, observed_positive_rate, and sample count N.
  - For empty buckets: explicitly reports N=0, mean_score=None, observed_positive_rate=None (no pseudo-values!).
  - Computes Expected Calibration Error (ECE) strictly over populated samples.
  - Generates reliability diagram visualization data.
  - NO threshold search, fitting, or optimization on holdout data.
- Development-only DetectorCalibrator:
  - Calibrates static threshold T on development/validation data only (rejects holdout data with HoldoutAccessError).
"""

from typing import List, Optional, Tuple, Sequence, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from pydantic import BaseModel, Field

from src.contracts.contracts import RiskScore, GroundTruthEvent, Alert, CalibrationResult
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.holdout import HoldoutAccessError
from src.state.alert_state_machine import AlertStateMachine


class CalibrationBucket(BaseModel):
    """Container for a single calibration bucket."""
    bucket: str
    range: List[float]
    n: int
    mean_score: Optional[float] = None
    observed_positive_rate: Optional[float] = None


class DescriptiveCalibrationResult(BaseModel):
    """Container for descriptive calibration results on holdout or validation data."""
    buckets: List[CalibrationBucket]
    expected_calibration_error: Optional[float] = None
    total_samples: int
    populated_samples: int
    reliability_diagram_data: Dict[str, Any] = Field(default_factory=dict)


def compute_descriptive_calibration(
    scores_with_timestamps: Sequence[Tuple[str, datetime, RiskScore]],
    ground_truth_events: Sequence[GroundTruthEvent],
    buckets: Sequence[Tuple[float, float]] = (
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.0),
    ),
    threshold: float = 3.5,
) -> DescriptiveCalibrationResult:
    """Compute empirical descriptive calibration statistics across score/confidence buckets.

    - Populated buckets: empirical mean score, empirical positive rate, sample count N.
    - Empty buckets: N=0, mean_score=None, observed_positive_rate=None.
    - ECE: Expected Calibration Error weighted by populated sample proportion.
    """
    gt_intervals = [(e.merchant_id, e.start_time, e.end_time) for e in ground_truth_events]

    valid_samples: List[Tuple[float, int]] = []
    for m_id, ts, rs in scores_with_timestamps:
        if rs.score is None:
            continue

        raw_score = float(rs.score)
        # Normalized score on [0, 1] probability scale
        prob = float(np.clip(raw_score / max(threshold * 2.0, 1.0), 0.0, 1.0))
        is_gt_positive = 1 if any(m_id == gm and st <= ts <= et for gm, st, et in gt_intervals) else 0
        valid_samples.append((prob, is_gt_positive))

    bucket_results: List[CalibrationBucket] = []
    populated_samples = 0
    weighted_error_sum = 0.0
    rel_x = []
    rel_y = []

    for low, high in buckets:
        # Match samples falling into bucket [low, high) (or [low, high] for last bucket)
        in_b = [
            s for s in valid_samples
            if (low <= s[0] < high) or (high >= 1.0 and low <= s[0] <= 1.0)
        ]
        n_b = len(in_b)

        if n_b > 0:
            mean_s = float(np.mean([s[0] for s in in_b]))
            obs_pos = float(np.mean([s[1] for s in in_b]))
            populated_samples += n_b
            weighted_error_sum += n_b * abs(obs_pos - mean_s)
            rel_x.append(round(mean_s, 4))
            rel_y.append(round(obs_pos, 4))

            bucket_results.append(
                CalibrationBucket(
                    bucket=f"{low:.1f}–{high:.1f}",
                    range=[float(low), float(high)],
                    n=n_b,
                    mean_score=round(mean_s, 4),
                    observed_positive_rate=round(obs_pos, 4),
                )
            )
        else:
            # Empty bucket: Explicitly report N=0, None for mean_score and observed_positive_rate
            bucket_results.append(
                CalibrationBucket(
                    bucket=f"{low:.1f}–{high:.1f}",
                    range=[float(low), float(high)],
                    n=0,
                    mean_score=None,
                    observed_positive_rate=None,
                )
            )

    ece = round(weighted_error_sum / populated_samples, 4) if populated_samples > 0 else 0.0

    return DescriptiveCalibrationResult(
        buckets=bucket_results,
        expected_calibration_error=ece,
        total_samples=len(valid_samples),
        populated_samples=populated_samples,
        reliability_diagram_data={
            "mean_predicted_probabilities": rel_x,
            "fraction_of_positives": rel_y,
        },
    )


class CalibrationDataset:
    """Container for calibration dataset streams enforcing structural holdout protection."""

    def __init__(
        self,
        scores_with_timestamps: List[Tuple[str, datetime, RiskScore]],
        ground_truth_events: List[GroundTruthEvent],
        is_holdout: bool = False,
    ):
        if is_holdout:
            raise HoldoutAccessError(
                "Holdout access denied: Locked holdout dataset cannot be used for calibration."
            )
        self.scores_with_timestamps = scores_with_timestamps
        self.ground_truth_events = ground_truth_events
        self.is_holdout = is_holdout

    @classmethod
    def from_development_stream(
        cls,
        scores_with_timestamps: List[Tuple[str, datetime, RiskScore]],
        ground_truth_events: List[GroundTruthEvent],
    ) -> "CalibrationDataset":
        """Factory creating a verified development CalibrationDataset."""
        return cls(
            scores_with_timestamps=scores_with_timestamps,
            ground_truth_events=ground_truth_events,
            is_holdout=False,
        )


class DetectorCalibrator:
    """Development-only calibrator for searching static threshold operating values on training/validation data."""

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
        dataset: CalibrationDataset,
        candidate_thresholds: Optional[List[float]] = None,
    ) -> CalibrationResult:
        """Calibrate static threshold over candidate_thresholds using CalibrationDataset."""
        if not isinstance(dataset, CalibrationDataset):
            raise TypeError(f"calibrate() requires CalibrationDataset input, got {type(dataset).__name__}")

        if dataset.is_holdout:
            raise HoldoutAccessError("Holdout access denied: DetectorCalibrator cannot consume locked holdout data.")

        scores_with_timestamps = dataset.scores_with_timestamps
        ground_truth_events = dataset.ground_truth_events

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

        for th in sorted(candidate_thresholds):
            metrics = self._evaluate_threshold(th, scores_with_timestamps, ground_truth_events)
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
