"""StatisticalDeviationScorer module for Day 3-6 vertical slice and ablation scoring.

Key Invariants:
- Input mapping: (FeatureSnapshot, BaselineSnapshot, Optional[signal_mask]) -> RiskScore.
- Scorer-level signal masking: allows evaluating FULL, -VOLUME, -VELOCITY, -AMOUNT, -BEHAVIORAL without perturbing baseline history.
- Standardized deviation magnitude M_k = |f_k - expected_k| / robust_scale_k.
- Zero-scale protection: raises ValueError if robust_scale <= 0.0.
- Statistical raw score S = max_k M_k (pure statistical standardized deviation).
- Evidence state mapping:
  - INSUFFICIENT: score = None, confidence = 0.0, triggered_signals = [].
  - DEGRADED: score = float(S), confidence = 0.5, data_quality = "DEGRADED".
  - SUFFICIENT: score = float(S), confidence = 1.0, data_quality = "GOOD".
- Triggered signals: list of feature names where M_k >= static_threshold.
- Deterministic and pure statistical scorer.
- GroundTruth & Holdout isolation: NO imports of ground truth or holdout code.
"""

from typing import Dict, Optional, Sequence

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore

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


class StatisticalDeviationScorer:
    """Computes standardized statistical deviation magnitude without EWMA smoothing."""

    def __init__(self, static_threshold: float = 3.5):
        if static_threshold <= 0.0:
            raise ValueError(f"static_threshold must be positive, got {static_threshold}")
        self.static_threshold = float(static_threshold)

    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
        signal_mask: Optional[Sequence[str]] = None,
    ) -> RiskScore:
        """Calculate RiskScore from feature_snapshot and baseline_snapshot, applying optional signal_mask."""
        # 1. Handle INSUFFICIENT evidence state
        if baseline_snapshot.evidence_state == "INSUFFICIENT":
            dq = "EMPTY" if feature_snapshot.data_quality == "EMPTY" else "INSUFFICIENT"
            return RiskScore(
                score=None,
                confidence=0.0,
                triggered_signals=[],
                data_quality=dq,
            )

        # 2. Compute standardized magnitudes M_k for all features (respecting signal_mask)
        m_magnitudes: Dict[str, float] = {}

        for feat_name, base_key in FEATURE_BASELINE_MAP.items():
            # Apply scorer-level signal mask if specified
            if signal_mask is not None:
                grp = FEATURE_GROUP_MAP.get(feat_name, feat_name)
                # If neither the feature name nor its group is in signal_mask, skip feature
                if feat_name not in signal_mask and grp not in signal_mask and base_key not in signal_mask:
                    continue

            if base_key not in baseline_snapshot.expected_values or base_key not in baseline_snapshot.robust_scale:
                raise KeyError(f"Missing required baseline feature expectation/scale for '{base_key}' (feature '{feat_name}')")

            # Extract feature value
            if feat_name in ("volume", "velocity", "unique_customers", "unique_devices"):
                f_val = float(getattr(feature_snapshot, feat_name))
            else:
                amt_stats = feature_snapshot.amount_statistics
                if feat_name not in amt_stats:
                    raise KeyError(f"Missing amount statistic '{feat_name}' in feature snapshot")
                f_val = float(amt_stats[feat_name])

            exp_val = float(baseline_snapshot.expected_values[base_key])
            r_scale = float(baseline_snapshot.robust_scale[base_key])

            if r_scale <= 0.0:
                raise ValueError(f"Robust scale for '{base_key}' must be strictly positive, got {r_scale}")

            m_k = abs(f_val - exp_val) / r_scale
            m_magnitudes[feat_name] = m_k

        # 3. Maximum deviation aggregation: S = max_k M_k
        raw_score = float(max(m_magnitudes.values())) if m_magnitudes else 0.0

        # 4. Triggered signals
        triggered = [
            feat_name for feat_name, mag in m_magnitudes.items()
            if mag >= self.static_threshold
        ]

        # 5. Evidence state and confidence mapping
        if baseline_snapshot.evidence_state == "DEGRADED":
            confidence = 0.5
            data_quality = "DEGRADED"
        else:
            confidence = 1.0
            data_quality = "GOOD"

        return RiskScore(
            score=raw_score,
            confidence=confidence,
            triggered_signals=triggered,
            data_quality=data_quality,
        )
