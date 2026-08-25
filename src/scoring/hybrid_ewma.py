"""HybridEWMAScorer module for computing anomaly risk scores from feature and baseline snapshots.

Key Invariants:
- Input mapping: (FeatureSnapshot, BaselineSnapshot) -> RiskScore.
- Explicit feature mapping for all 11 monitored features (4 scalar + 7 amount statistics).
- Standardized magnitude M_k = |f_k - expected_k| / robust_scale_k.
- Zero-scale protection: raises ValueError if robust_scale <= 0.0.
- Raw score S_raw = max_k M_k.
- EWMA smoothing: S_ewma,t = alpha * S_raw,t + (1 - alpha) * S_ewma,t-1.
- State reset on INSUFFICIENT evidence: resets merchant EWMA state on evidence gap.
- Evidence state mapping:
  - INSUFFICIENT: score = None, confidence = 0.0, triggered_signals = [].
  - DEGRADED: score = float(S_ewma), confidence = 0.5, data_quality = "DEGRADED".
  - SUFFICIENT: score = float(S_ewma), confidence = 1.0, data_quality = "GOOD".
- Triggered signals: list of feature names where M_k >= static_threshold.
- Persistence gating is owned by AlertStateMachine in Day 6.
- Merchant EWMA isolation: EWMA state is strictly maintained per merchant_id.
- GroundTruth & Holdout isolation: NO imports of ground truth or holdout code.
"""

from typing import Dict, Optional

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.contracts.config_schemas import DetectorConfig, ScorerConfig

# Explicit feature contract mapping: (FeatureSnapshot field/attribute) -> (BaselineSnapshot expected/scale key)
FEATURE_BASELINE_MAP: Dict[str, str] = {
    "volume": "volume",
    "velocity": "velocity",
    "unique_customers": "unique_customers",
    "unique_devices": "unique_devices",
    "total_amount": "amount_total_amount",
    "mean_amount": "amount_mean_amount",
    "std_amount": "amount_std_amount",
    "median_amount": "amount_median_amount",
    "mad_amount": "amount_mad_amount",
    "min_amount": "amount_min_amount",
    "max_amount": "amount_max_amount",
}


class HybridEWMAScorer:
    """Computes robust standardized deviation scores smoothed via Exponential Weighted Moving Average (EWMA)."""

    def __init__(
        self,
        alpha: float = 0.3,
        static_threshold: float = 3.5,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")
        if static_threshold <= 0.0:
            raise ValueError(f"static_threshold must be positive, got {static_threshold}")

        self.alpha = float(alpha)
        self.static_threshold = float(static_threshold)

        # Per-merchant EWMA state: merchant_id -> last_ewma_score (float)
        self._ewma_states: Dict[str, float] = {}

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "HybridEWMAScorer":
        """Construct HybridEWMAScorer from DetectorConfig."""
        scorer_cfg: ScorerConfig = config.scorer
        return cls(
            alpha=scorer_cfg.alpha,
            static_threshold=scorer_cfg.static_threshold,
        )

    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
    ) -> RiskScore:
        """Calculate RiskScore from feature_snapshot and baseline_snapshot."""
        m_id = feature_snapshot.merchant_id

        # 1. Handle INSUFFICIENT evidence state
        if baseline_snapshot.evidence_state == "INSUFFICIENT":
            # Reset merchant EWMA state on evidence gap to prevent stale state leakage
            self._ewma_states.pop(m_id, None)

            dq = "EMPTY" if feature_snapshot.data_quality == "EMPTY" else "INSUFFICIENT"
            return RiskScore(
                score=None,
                confidence=0.0,
                triggered_signals=[],
                data_quality=dq,
            )

        # 2. Compute standardized magnitudes M_k for all 11 required features
        m_magnitudes: Dict[str, float] = {}

        for feat_name, base_key in FEATURE_BASELINE_MAP.items():
            if base_key not in baseline_snapshot.expected_values or base_key not in baseline_snapshot.robust_scale:
                raise KeyError(f"Missing required baseline feature expectation/scale for '{base_key}' (feature '{feat_name}')")

            # Extract feature value
            if feat_name in ("volume", "velocity", "unique_customers", "unique_devices"):
                f_val = float(getattr(feature_snapshot, feat_name))
            else:
                if feat_name not in feature_snapshot.amount_statistics:
                    raise KeyError(f"Missing required amount statistic '{feat_name}' in FeatureSnapshot")
                f_val = float(feature_snapshot.amount_statistics[feat_name])

            exp_val = float(baseline_snapshot.expected_values[base_key])
            scale_val = float(baseline_snapshot.robust_scale[base_key])

            if scale_val <= 0.0:
                raise ValueError(f"Invalid non-positive robust scale for feature '{base_key}': {scale_val}")

            m_magnitudes[base_key] = abs(f_val - exp_val) / scale_val

        # Maximum standardized magnitude
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
