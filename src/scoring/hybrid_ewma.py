"""Hybrid EWMA scorer implementation conforming to AnomalyScorer ABC (Section 17).

Key Invariants:
- Implements AnomalyScorer ABC interface.
- Monitored feature mapping: maps all required features from FeatureSnapshot to BaselineSnapshot expected values and scales.
- Scorer-level signal masking and weighting: filters and weights candidate deviations inside the scorer interface.
- Standardized deviation: M_k = (|observed_k - expected_k| / robust_scale_k) * weight_k.
- Score aggregation: raw score S_raw = max_k M_k over active features.
- Dynamic EWMA update: S_ewma(t) = alpha * S_raw + (1 - alpha) * S_ewma(t - 1) per merchant.
- Reset on INSUFFICIENT evidence state to avoid carrying stale score across data gaps.
- Confidence mapping:
    INSUFFICIENT -> confidence = 0.0, score = None
    DEGRADED     -> confidence = 0.5, score = S_ewma
    SUFFICIENT   -> confidence = 1.0, score = S_ewma
- Triggered signals: list of feature names where M_k >= static_threshold.
"""

from typing import Dict, Optional, Sequence

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.contracts.config_schemas import DetectorConfig, ScorerConfig
from src.scoring.base import AnomalyScorer

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

FEATURE_GROUP_MAP: Dict[str, str] = {
    "volume": "volume",
    "velocity": "velocity",
    "unique_customers": "behavioral",
    "unique_devices": "behavioral",
    "total_amount": "amount",
    "mean_amount": "amount",
    "std_amount": "amount",
    "median_amount": "amount",
    "mad_amount": "amount",
    "min_amount": "amount",
    "max_amount": "amount",
}


class HybridEWMAScorer(AnomalyScorer):
    """Computes robust standardized deviation scores smoothed via Exponential Weighted Moving Average (EWMA) conforming to AnomalyScorer ABC."""

    def __init__(
        self,
        alpha: float = 0.3,
        static_threshold: float = 3.5,
        signal_weights: Optional[Dict[str, float]] = None,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}")
        if static_threshold <= 0.0:
            raise ValueError(f"static_threshold must be positive, got {static_threshold}")

        self.alpha = float(alpha)
        self.static_threshold = float(static_threshold)
        self.signal_weights = dict(signal_weights) if signal_weights is not None else None

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
        signal_mask: Optional[Sequence[str]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> RiskScore:
        """Calculate RiskScore from feature_snapshot and baseline_snapshot, applying optional signal_mask and weights."""
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

        active_weights = signal_weights if signal_weights is not None else self.signal_weights

        # 2. Compute standardized magnitudes M_k for features
        m_magnitudes: Dict[str, float] = {}

        for feat_name, base_key in FEATURE_BASELINE_MAP.items():
            grp = FEATURE_GROUP_MAP.get(feat_name, feat_name)

            # Apply scorer-level signal mask if specified
            if signal_mask is not None:
                if feat_name not in signal_mask and grp not in signal_mask and base_key not in signal_mask:
                    continue

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

            m_k = abs(f_val - exp_val) / scale_val

            # Apply signal weight if configured
            if active_weights is not None:
                w = float(active_weights.get(feat_name, active_weights.get(grp, 1.0)))
                m_k *= w

            m_magnitudes[base_key] = m_k

        # Maximum standardized magnitude
        s_raw = float(max(m_magnitudes.values())) if m_magnitudes else 0.0
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
        if baseline_snapshot.evidence_state == "DEGRADED" or feature_snapshot.data_quality == "DEGRADED":
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
