"""Anomaly injectors and GroundTruthEvent generation.

Seven required anomaly classes:
1. velocity_spike: burst of transactions in a short window.
2. volume_spike: elevated transaction volume over a window.
3. amount_spike: sudden shift to high transaction amounts.
4. behavioral_shift: spike in unique device_ids / customer_ids.
5. attribute_anomaly: shift in payment_method or country.
6. sustained_anomaly: multi-window sustained volume/velocity elevation.
7. compound_anomaly: simultaneous multi-signal anomaly (volume + amount + device).

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
from typing import Any, Optional
import numpy as np

from src.contracts.contracts import GroundTruthEvent


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

    return GroundTruthEvent(
        event_id=event_id,
        merchant_id=merchant_id,
        anomaly_type=spec.anomaly_type,
        start_time=st,
        end_time=et,
        severity=float(realized_magnitude),
        parameters=params,
    )
