"""StaticThresholdScorer module implementing fixed threshold-based scoring conforming to AnomalyScorer ABC.

Key Invariants:
- Fixed static threshold limits without adaptive robust scaling or EWMA smoothing.
- Implements AnomalyScorer ABC interface.
- Input mapping: (FeatureSnapshot, BaselineSnapshot, Optional[signal_mask], Optional[signal_weights]) -> RiskScore.
- Scorer-level signal masking and weighting: allows evaluating feature subsets and weight vectors.
- Produces valid RiskScore with explicit confidence and data quality mapping:
  - INSUFFICIENT: score = None, confidence = 0.0, triggered_signals = [].
  - DEGRADED: score = float(S_static), confidence = 0.5, data_quality = "DEGRADED".
  - SUFFICIENT: score = float(S_static), confidence = 1.0, data_quality = "GOOD".
- Deterministic, stateless, merchant-isolated, zero GroundTruth / Holdout dependency.
"""

from typing import Dict, Optional, Sequence

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore
from src.scoring.base import AnomalyScorer

DEFAULT_STATIC_LIMITS: Dict[str, float] = {
    "volume": 30.0,
    "velocity": 30.0,
    "unique_customers": 20.0,
    "unique_devices": 20.0,
    "total_amount": 1500.0,
    "mean_amount": 150.0,
    "std_amount": 50.0,
    "median_amount": 150.0,
    "mad_amount": 30.0,
    "min_amount": 100.0,
    "max_amount": 300.0,
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


class StaticThresholdScorer(AnomalyScorer):
    """Computes static threshold risk scores using fixed limits rather than adaptive statistical deviations."""

    def __init__(
        self,
        static_threshold: float = 3.5,
        static_limits: Optional[Dict[str, float]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ):
        if static_threshold <= 0.0:
            raise ValueError(f"static_threshold must be positive, got {static_threshold}")
        self.static_threshold = float(static_threshold)
        self.static_limits = dict(static_limits if static_limits is not None else DEFAULT_STATIC_LIMITS)
        self.signal_weights = dict(signal_weights) if signal_weights is not None else None

    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
        signal_mask: Optional[Sequence[str]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> RiskScore:
        """Calculate RiskScore from feature_snapshot against fixed static limits."""
        # 1. Handle INSUFFICIENT evidence state
        if baseline_snapshot.evidence_state == "INSUFFICIENT":
            dq = "EMPTY" if feature_snapshot.data_quality == "EMPTY" else "INSUFFICIENT"
            return RiskScore(
                score=None,
                confidence=0.0,
                triggered_signals=[],
                data_quality=dq,
            )

        active_weights = signal_weights if signal_weights is not None else self.signal_weights

        # 2. Compute static ratio magnitudes for all active features
        m_magnitudes: Dict[str, float] = {}

        for feat_name, limit_val in self.static_limits.items():
            grp = FEATURE_GROUP_MAP.get(feat_name, feat_name)

            # Apply scorer-level signal mask if specified
            if signal_mask is not None:
                if feat_name not in signal_mask and grp not in signal_mask:
                    continue

            if limit_val <= 0.0:
                raise ValueError(f"Static limit for '{feat_name}' must be positive, got {limit_val}")

            # Extract feature value
            if feat_name in ("volume", "velocity", "unique_customers", "unique_devices"):
                f_val = float(getattr(feature_snapshot, feat_name))
            else:
                amt_stats = feature_snapshot.amount_statistics
                if feat_name not in amt_stats:
                    raise KeyError(f"Missing amount statistic '{feat_name}' in feature snapshot")
                f_val = float(amt_stats[feat_name])

            # Static magnitude normalized to static_threshold scale
            m_k = (f_val / limit_val) * self.static_threshold

            # Apply signal weight if configured
            if active_weights is not None:
                w = float(active_weights.get(feat_name, active_weights.get(grp, 1.0)))
                m_k *= w

            m_magnitudes[feat_name] = m_k

        # 3. Maximum deviation aggregation: S = max_k M_k
        raw_score = float(max(m_magnitudes.values())) if m_magnitudes else 0.0

        # 4. Triggered signals
        triggered = sorted([
            feat_name for feat_name, mag in m_magnitudes.items()
            if mag >= self.static_threshold
        ])

        # 5. Evidence state and composite confidence calculation
        from src.scoring.confidence import compute_composite_confidence
        confidence = compute_composite_confidence(
            feature_snapshot=feature_snapshot,
            baseline_snapshot=baseline_snapshot,
            magnitudes=m_magnitudes,
            static_threshold=self.static_threshold,
            signal_mask=signal_mask,
        )
        data_quality = "DEGRADED" if (baseline_snapshot.evidence_state == "DEGRADED" or feature_snapshot.data_quality == "DEGRADED") else "GOOD"

        return RiskScore(
            score=raw_score,
            confidence=confidence,
            triggered_signals=triggered,
            data_quality=data_quality,
        )
