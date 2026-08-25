"""BaselineEngine module for establishing reference baseline expectations and scale dispersion.

Consumes historical FeatureSnapshot objects to compute BaselineSnapshot objects.

Key Invariants:
- Baseline evidence eligibility: EMPTY snapshots (volume == 0) are EXCLUDED from median and MAD calculations.
- Evidence state ownership: INSUFFICIENT, DEGRADED, SUFFICIENT.
- Provenance: min_history_count (50) and min_window_count (5) are config-driven from config/detector.yaml & Section 15.
- Historical-only updates: current window baseline depends strictly on past eligible snapshots (t_past < t_current).
- Zero future leakage: adding future snapshots does not affect past/current baseline state.
- GroundTruth & Holdout isolation: NO imports of GroundTruthEvent, AnomalySpec, ground truth code, or holdout code.
- Merchant isolation: history strictly partitioned per merchant_id.
- Robust statistics: sample median for expected_values, MAD with robust floor for robust_scale.
- Schema compliance: all emitted snapshots validate against BaselineSnapshot contract.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot


class BaselineEngine:
    """Establishes reference baseline expectations and robust scale dispersion per merchant."""

    def __init__(
        self,
        min_history_count: int = 50,
        min_window_count: int = 5,
        max_history_window: Optional[int] = 500,
    ):
        if min_history_count <= 0:
            raise ValueError(f"min_history_count must be positive, got {min_history_count}")
        if min_window_count < 0:
            raise ValueError(f"min_window_count must be non-negative, got {min_window_count}")

        self.min_history_count = min_history_count
        self.min_window_count = min_window_count
        self.max_history_window = max_history_window

        # Per-merchant history storage: merchant_id -> list of FeatureSnapshots
        self.histories: Dict[str, List[FeatureSnapshot]] = {}

    def get_baseline(
        self,
        merchant_id: str,
        current_snapshot: FeatureSnapshot,
    ) -> BaselineSnapshot:
        """Compute the BaselineSnapshot for current_snapshot using eligible past historical snapshots (t_past < t_current)."""
        ts = current_snapshot.timestamp
        if ts.tzinfo is None:
            raise TypeError(f"current_snapshot timestamp must be timezone-aware (got naive datetime {ts})")

        merchant_history = self.histories.get(merchant_id, [])

        # Filter past snapshots strictly before current timestamp (t_past < t_current)
        past_history = [
            snap for snap in merchant_history
            if snap.timestamp < ts
        ]

        # Filter baseline evidence eligibility: exclude EMPTY windows (volume == 0) from median/MAD calculations
        eligible_history = [
            snap for snap in past_history
            if snap.data_quality != "EMPTY" and snap.volume > 0.0
        ]

        if self.max_history_window and len(eligible_history) > self.max_history_window:
            eligible_history = eligible_history[-self.max_history_window:]

        history_count = len(eligible_history)
        current_volume = int(round(current_snapshot.volume))

        # Determine evidence_state based on eligible history count and current window count
        if history_count < self.min_history_count:
            evidence_state = "INSUFFICIENT"
        elif current_snapshot.data_quality == "EMPTY" or current_volume < self.min_window_count:
            evidence_state = "DEGRADED"
        else:
            evidence_state = "SUFFICIENT"

        if history_count == 0:
            # First snapshot or no eligible historical evidence: empty baseline
            return BaselineSnapshot(
                merchant_id=merchant_id,
                timestamp=ts,
                expected_values={},
                robust_scale={},
                history_count=0,
                current_window_count=current_volume,
                evidence_state=evidence_state,
            )

        # Compute robust expected_values and robust_scale strictly across eligible_history
        expected_values: Dict[str, float] = {}
        robust_scale: Dict[str, float] = {}

        # 1. Scalar features from FeatureSnapshot
        scalar_keys = ["volume", "velocity", "unique_customers", "unique_devices"]
        for key in scalar_keys:
            vals = np.array([getattr(snap, key) for snap in eligible_history], dtype=np.float64)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))

            if key in ("volume", "velocity"):
                floor = max(0.5, 0.2 * med)
            else:
                floor = max(1.0, 0.2 * med)

            expected_values[key] = med
            robust_scale[key] = max(floor, mad)

        # 2. Nested features from amount_statistics
        all_amount_keys = set()
        for snap in eligible_history:
            all_amount_keys.update(snap.amount_statistics.keys())

        for key in sorted(all_amount_keys):
            vals = np.array([snap.amount_statistics.get(key, 0.0) for snap in eligible_history], dtype=np.float64)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            floor = max(1.0, 0.2 * med)

            expected_values[f"amount_{key}"] = med
            robust_scale[f"amount_{key}"] = max(floor, mad)

        return BaselineSnapshot(
            merchant_id=merchant_id,
            timestamp=ts,
            expected_values=expected_values,
            robust_scale=robust_scale,
            history_count=history_count,
            current_window_count=current_volume,
            evidence_state=evidence_state,
        )

    def update(self, snapshot: FeatureSnapshot) -> None:
        """Update merchant history by appending snapshot for future baseline computations."""
        if snapshot.timestamp.tzinfo is None:
            raise TypeError(f"snapshot timestamp must be timezone-aware (got naive datetime {snapshot.timestamp})")

        m_id = snapshot.merchant_id
        if m_id not in self.histories:
            self.histories[m_id] = []

        self.histories[m_id].append(snapshot)
        self.histories[m_id].sort(key=lambda s: s.timestamp)
