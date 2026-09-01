"""Anomaly injectors and GroundTruthEvent generation.

Eleven canonical anomaly classes:
1. sudden_volume_spike: sharp volume spike over a window.
2. velocity_burst: intense short-duration burst of transactions.
3. sustained_spike: multi-window sustained volume elevation.
4. amount_distribution_shift: sudden upward shift in transaction amounts.
5. device_behavior_anomaly: high concentration in compromised devices/customers.
6. attribute_geographic_shift: sudden surge in high-risk country or prepaid payment methods.
7. compound_anomaly: concurrent multi-signal anomaly (volume + amount + device + attribute).
8. threshold_hugging_evasion: crafted anomaly hovering right below or at decision threshold.
9. persistence_evasion: alternating short bursts intentionally failing the multi-window persistence requirement.
10. staircase_ramp: multi-window step-wise progressive regime increase in transaction rate.
11. oscillating_sub_threshold: periodic oscillatory waveform remaining within the sub-threshold envelope.

Ground Truth Realized Magnitude Architecture:
Target magnitude (spec.target_magnitude) represents the generator injection control intent.
Realized magnitude (realized_magnitude) is computed from actual generated transactions:
M = |observed - expected| / robust_scale.
GroundTruthEvent.severity receives realized_magnitude, and severity_level is derived automatically.

Compound Severity Rule (Section 14):
Compound severity = mean absolute standardized deviation across active signals.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, List
import numpy as np

from src.contracts.contracts import GroundTruthEvent

# Canonical 11 Anomaly Type Identifiers
CANONICAL_ANOMALY_TYPES: List[str] = [
    "sudden_volume_spike",
    "velocity_burst",
    "sustained_spike",
    "amount_distribution_shift",
    "device_behavior_anomaly",
    "attribute_geographic_shift",
    "compound_anomaly",
    "threshold_hugging_evasion",
    "persistence_evasion",
    "staircase_ramp",
    "oscillating_sub_threshold",
]


@dataclass
class AnomalySpec:
    """Injection control specification for generating anomalies."""

    anomaly_type: str
    start_time: datetime
    duration_seconds: float
    target_magnitude: float  # Injection control intent
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> datetime:
        st = self.start_time if self.start_time.tzinfo else self.start_time.replace(tzinfo=timezone.utc)
        return st + timedelta(seconds=self.duration_seconds)


def compute_standardized_magnitude(observed: float, expected: float, robust_scale: float) -> float:
    """Compute standardized deviation magnitude M = |observed - expected| / robust_scale."""
    scale = max(0.001, robust_scale)
    return abs(observed - expected) / scale


def compute_compound_severity(signal_magnitudes: list[float]) -> float:
    """Compute compound severity per Section 14 rule:

    Compound severity = mean absolute standardized deviation across active signals.
    """
    if not signal_magnitudes:
        return 0.0
    return float(np.mean([abs(m) for m in signal_magnitudes]))


def create_ground_truth_event(
    event_id: str,
    merchant_id: str,
    spec: AnomalySpec,
    realized_magnitude: float,
) -> GroundTruthEvent:
    """Create a valid GroundTruthEvent with measured realized magnitude."""
    st = spec.start_time if spec.start_time.tzinfo else spec.start_time.replace(tzinfo=timezone.utc)
    et = spec.end_time if spec.end_time.tzinfo else spec.end_time.replace(tzinfo=timezone.utc)

    if st >= et:
        raise ValueError(f"GroundTruthEvent temporal error: start_time ({st}) must be < end_time ({et})")

    params = dict(spec.parameters)
    params["target_magnitude"] = float(spec.target_magnitude)
    params.setdefault("excess_transaction_count", max(1.0, float(round(10.0 * realized_magnitude))))
    params.setdefault("mean_transaction_amount", 50.0)
    params.setdefault("exposure_factor", 1.0)

    return GroundTruthEvent(
        event_id=event_id,
        merchant_id=merchant_id,
        anomaly_type=spec.anomaly_type,
        start_time=st,
        end_time=et,
        severity=float(realized_magnitude),
        parameters=params,
    )

