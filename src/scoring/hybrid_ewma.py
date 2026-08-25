"""HybridEWMAScorer module for computing anomaly risk scores from feature and baseline snapshots.

Key Invariants:
- Input mapping: (FeatureSnapshot, BaselineSnapshot) -> RiskScore.
- Standardized magnitude M_k = |f_k - expected_k| / robust_scale_k.
- Raw score S_raw = max_k M_k.
- EWMA smoothing: S_ewma,t = alpha * S_raw,t + (1 - alpha) * S_ewma,t-1.
- Evidence state mapping:
  - INSUFFICIENT: score = None, confidence = 0.0, triggered_signals = [].
  - DEGRADED: score = float(S_ewma), confidence = 0.5, data_quality = "DEGRADED".
  - SUFFICIENT: score = float(S_ewma), confidence = 1.0, data_quality = "GOOD".
- Triggered signals: list of feature names where M_k >= static_threshold.
- Merchant EWMA isolation: EWMA state is strictly maintained per merchant_id.
- GroundTruth & Holdout isolation: NO imports of ground truth or holdout code.
"""

from typing import Dict, List, Optional
import numpy as np

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.contracts.config_schemas import DetectorConfig, ScorerConfig


class HybridEWMAScorer:
    """Computes robust standardized deviation scores smoothed via Exponential Weighted Moving Average (EWMA)."""

    def __init__(
        self,
        alpha: float = 0.3,
        static_threshold: float = 3.5,
        persistence: int = 2,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")
        if static_threshold <= 0.0:
            raise ValueError(f"static_threshold must be positive, got {static_threshold}")

        self.alpha = float(alpha)
        self.static_threshold = float(static_threshold)
        self.persistence = persistence

        # Per-merchant EWMA state: merchant_id -> last_ewma_score (float)
        self._ewma_states: Dict[str, float] = {}

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "HybridEWMAScorer":
        """Construct HybridEWMAScorer from DetectorConfig."""
        scorer_cfg: ScorerConfig = config.scorer
        return cls(
            alpha=scorer_cfg.alpha,
            static_threshold=scorer_cfg.static_threshold,
            persistence=scorer_cfg.persistence,
        )

    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
    ) -> RiskScore:
        """Calculate RiskScore from feature_snapshot and baseline_snapshot."""
        m_id = feature_snapshot.merchant_id

        # 1. Check evidence_state
        if baseline_snapshot.evidence_state == "INSUFFICIENT":
            return RiskScore(
                score=None,
                confidence=0.0,
                triggered_signals=[],
                data_quality=feature_snapshot.data_quality,
            )

        # 2. Compute standardized magnitudes M_k across all monitored features
        m_magnitudes: Dict[str, float] = {}

        # Scalar features
        scalar_keys = ["volume", "velocity", "unique_customers", "unique_devices"]
        for key in scalar_keys:
            if key in baseline_snapshot.expected_values and key in baseline_snapshot.robust_scale:
                f_val = float(getattr(feature_snapshot, key))
                exp_val = baseline_snapshot.expected_values[key]
                scale_val = baseline_snapshot.robust_scale[key]
                if scale_val > 0.0:
                    m_magnitudes[key] = abs(f_val - exp_val) / scale_val

        # Amount features
        for amt_key, f_val in feature_snapshot.amount_statistics.items():
            b_key = f"amount_{amt_key}"
            if b_key in baseline_snapshot.expected_values and b_key in baseline_snapshot.robust_scale:
                exp_val = baseline_snapshot.expected_values[b_key]
                scale_val = baseline_snapshot.robust_scale[b_key]
                if scale_val > 0.0:
                    m_magnitudes[b_key] = abs(f_val - exp_val) / scale_val

        if not m_magnitudes:
            s_raw = 0.0
            triggered_signals = []
        else:
            s_raw = float(max(m_magnitudes.values()))
            triggered_signals = sorted([
                k for k, mag in m_magnitudes.items()
                if mag >= self.static_threshold
            ])

        # 3. Update EWMA state per merchant
        if m_id not in self._ewma_states:
            s_ewma = s_raw
        else:
            s_ewma = self.alpha * s_raw + (1.0 - self.alpha) * self._ewma_states[m_id]

        self._ewma_states[m_id] = s_ewma

        # 4. Map evidence state confidence and data_quality
        if baseline_snapshot.evidence_state == "DEGRADED":
            confidence = 0.5
            dq = "DEGRADED"
        else:
            confidence = 1.0
            dq = "GOOD"

        return RiskScore(
            score=s_ewma,
            confidence=confidence,
            triggered_signals=triggered_signals,
            data_quality=dq,
        )

    def reset(self, merchant_id: Optional[str] = None) -> None:
        """Reset EWMA state for a specific merchant or all merchants."""
        if merchant_id is not None:
            self._ewma_states.pop(merchant_id, None)
        else:
            self._ewma_states.clear()
