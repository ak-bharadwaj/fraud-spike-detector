"""Scoring package for calculating standardized deviation risk scores.

Provides both statistical and ML-based anomaly scoring strategies:
- StatisticalDeviationScorer: Robust Z-score deviation (frozen canonical scorer)
- HybridEWMAScorer: EWMA-smoothed statistical deviation
- StaticThresholdScorer: Fixed threshold scoring
- IsolationForestScorer: Unsupervised anomaly detection (ML)
- XGBoostFraudScorer: Supervised fraud classification with Platt calibration (ML)
- EnsembleFraudScorer: Weighted IF + XGBoost combination (ML)
"""

from src.scoring.base import AnomalyScorer
from src.scoring.static import StaticThresholdScorer
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.scoring.ml_scorer import (
    IsolationForestScorer,
    XGBoostFraudScorer,
    EnsembleFraudScorer,
)

__all__ = [
    "AnomalyScorer",
    "StaticThresholdScorer",
    "StatisticalDeviationScorer",
    "HybridEWMAScorer",
    "IsolationForestScorer",
    "XGBoostFraudScorer",
    "EnsembleFraudScorer",
]
