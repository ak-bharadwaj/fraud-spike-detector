"""Composite confidence calculation module conforming to Master Plan Section 17/19.

Calculates normalized confidence in [0.0, 1.0] as a composite of:
1. Evidence Quality (E): Based on BaselineEngine evidence state and FeatureSnapshot data quality.
2. Feature Availability (F): Proportion of unmasked, valid feature groups present in the snapshot.
3. Signal Agreement (S): Multi-signal corroboration among elevated feature deviations.

Formula:
  C = 0.0 if E == 0.0 else round(E * F * S, 4)

Key Invariants:
- RiskScore.score semantics remain strictly unaltered.
- Confidence is bounded in [0.0, 1.0].
- Evidence state remains owned solely by BaselineEngine.
- Scorer-level ablation reduces feature availability (F) without starving baseline history.
- Enables independent variation: High Risk with Lower Confidence under degraded quality (E=0.5), partial availability (F<1.0), or isolated single-signal anomalies (S<1.0).
"""

from typing import Dict, Optional, Sequence
from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot

FEATURE_GROUPS = {
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
    "amount_total_amount": "amount",
    "amount_mean_amount": "amount",
    "amount_std_amount": "amount",
    "amount_median_amount": "amount",
    "amount_mad_amount": "amount",
    "amount_min_amount": "amount",
    "amount_max_amount": "amount",
}

ALL_FEATURE_GROUP_NAMES = {"volume", "velocity", "behavioral", "amount"}


def compute_composite_confidence(
    feature_snapshot: FeatureSnapshot,
    baseline_snapshot: BaselineSnapshot,
    magnitudes: Dict[str, float],
    static_threshold: float = 3.5,
    signal_mask: Optional[Sequence[str]] = None,
) -> float:
    """Compute deterministic composite confidence bounded in [0.0, 1.0]."""
    # 1. Evidence Quality Factor (E)
    if baseline_snapshot.evidence_state == "INSUFFICIENT" or feature_snapshot.data_quality in ("INSUFFICIENT", "EMPTY"):
        return 0.0

    if baseline_snapshot.evidence_state == "DEGRADED" or feature_snapshot.data_quality == "DEGRADED":
        e_factor = 0.5
    else:
        e_factor = 1.0

    # 2. Feature Availability Factor (F)
    if signal_mask is None:
        f_factor = 1.0
    else:
        mask_set = set(signal_mask)
        active_groups = set()
        for feat_name, grp in FEATURE_GROUPS.items():
            if feat_name in mask_set or grp in mask_set:
                active_groups.add(grp)
        f_factor = len(active_groups) / len(ALL_FEATURE_GROUP_NAMES) if ALL_FEATURE_GROUP_NAMES else 1.0
        f_factor = max(0.25, min(1.0, f_factor))

    # 3. Signal Agreement Factor (S)
    if not magnitudes:
        s_factor = 1.0
    else:
        max_mag = max(magnitudes.values()) if magnitudes else 0.0
        if max_mag < static_threshold:
            # Sub-threshold nominal regime: all signals agree on nominal behavior
            s_factor = 1.0
        else:
            # Above-threshold anomaly regime: check multi-group corroboration
            group_max_mags: Dict[str, float] = {}
            for feat_name, mag in magnitudes.items():
                grp = FEATURE_GROUPS.get(feat_name, feat_name)
                group_max_mags[grp] = max(group_max_mags.get(grp, 0.0), mag)

            elevated_groups = sum(
                1 for g_mag in group_max_mags.values()
                if g_mag >= max(1.0, 0.3 * max_mag)
            )
            total_active_groups = len(group_max_mags) if group_max_mags else 1

            if total_active_groups <= 1 or elevated_groups >= 2:
                # Corroborated anomaly across 2 or more feature groups (e.g. volume + velocity)
                s_factor = 1.0
            else:
                # Isolated single-signal spike without corroboration from any other feature group
                s_factor = 0.75

    # 4. Multiplicative Composite Confidence
    composite = e_factor * f_factor * s_factor
    return round(float(max(0.0, min(1.0, composite))), 4)
