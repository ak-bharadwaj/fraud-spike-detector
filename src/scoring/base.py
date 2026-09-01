"""Base abstract class for anomaly scorers (Section 17).

Key Invariants:
- Common AnomalyScorer ABC interface for all concrete scoring strategies:
  - StaticThresholdScorer
  - StatisticalDeviationScorer
  - HybridEWMAScorer
- Strategy polymorphism: StreamingDetectorPipeline interacts uniformly via AnomalyScorer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

from src.contracts.contracts import FeatureSnapshot, BaselineSnapshot, RiskScore


class AnomalyScorer(ABC):
    """Abstract base class defining the common anomaly scoring strategy contract (Section 17)."""

    @abstractmethod
    def calculate_score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
        signal_mask: Optional[Sequence[str]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> RiskScore:
        """Calculate RiskScore from FeatureSnapshot and BaselineSnapshot."""
        pass

    def score(
        self,
        feature_snapshot: FeatureSnapshot,
        baseline_snapshot: BaselineSnapshot,
        signal_mask: Optional[Sequence[str]] = None,
        signal_weights: Optional[Dict[str, float]] = None,
    ) -> RiskScore:
        """Convenience alias for calculate_score conforming to Section 17 contract."""
        return self.calculate_score(
            feature_snapshot=feature_snapshot,
            baseline_snapshot=baseline_snapshot,
            signal_mask=signal_mask,
            signal_weights=signal_weights,
        )
