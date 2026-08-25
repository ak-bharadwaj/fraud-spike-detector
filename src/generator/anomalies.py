"""Anomaly injectors and GroundTruthEvent generation.

Seven required anomaly classes:
1. velocity_spike: burst of transactions in a short window.
2. volume_spike: elevated transaction volume over a window.
3. amount_spike: sudden shift to high transaction amounts.
4. behavioral_shift: spike in unique device_ids / customer_ids.
5. attribute_anomaly: shift in payment_method or country.
6. sustained_anomaly: multi-window sustained volume/velocity elevation.
7. compound_anomaly: simultaneous multi-signal anomaly (volume + amount + device).

GroundTruthEvent creation enforces:
- start_time < end_time
- standardized deviation magnitude M = |observed - expected| / robust_scale
- severity_level derived automatically ("LOW", "MEDIUM", "HIGH")
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import numpy as np

from src.contracts.contracts import GroundTruthEvent
from src.generator.archetypes import MerchantProfile


@dataclass
class AnomalySpec:
    anomaly_type: str
    start_time: datetime
    duration_seconds: float
    target_magnitude: float  # Intended standardized deviation M
    parameters: dict[str, Any]

    @property
    def end_time(self) -> datetime:
        st = self.start_time if self.start_time.tzinfo else self.start_time.replace(tzinfo=timezone.utc)
        return st + timedelta(seconds=self.duration_seconds)


def create_ground_truth_event(
    event_id: str,
    merchant_id: str,
    spec: AnomalySpec,
    actual_magnitude: Optional[float] = None,
) -> GroundTruthEvent:
    """Create a valid GroundTruthEvent from an AnomalySpec."""
    st = spec.start_time if spec.start_time.tzinfo else spec.start_time.replace(tzinfo=timezone.utc)
    et = spec.end_time if spec.end_time.tzinfo else spec.end_time.replace(tzinfo=timezone.utc)

    if st >= et:
        raise ValueError(f"GroundTruthEvent temporal error: start_time ({st}) must be < end_time ({et})")

    magnitude = actual_magnitude if actual_magnitude is not None else spec.target_magnitude

    return GroundTruthEvent(
        event_id=event_id,
        merchant_id=merchant_id,
        anomaly_type=spec.anomaly_type,
        start_time=st,
        end_time=et,
        severity=float(magnitude),
        parameters=spec.parameters,
    )


def compute_standardized_magnitude(observed: float, expected: float, robust_scale: float) -> float:
    """Compute standardized deviation magnitude M = |observed - expected| / robust_scale."""
    scale = max(0.001, robust_scale)
    return abs(observed - expected) / scale


def compute_compound_severity(signal_magnitudes: list[float]) -> float:
    """Compute compound severity as mean absolute standardized deviation across active signals."""
    if not signal_magnitudes:
        return 0.0
    return float(np.mean([abs(m) for m in signal_magnitudes]))
