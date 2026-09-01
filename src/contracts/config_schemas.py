"""Configuration schemas for YAML validation using Pydantic.

Structural schemas define configuration shapes and requirements. Operating values
must be explicitly supplied via external YAML configuration files and selected during
development sweeps (Days 4-7). No research parameters are hardcoded into Python schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ScorerConfig(BaseModel):
    """Configuration shape for anomaly scoring strategy."""
    type: str
    alpha: Optional[float] = None
    persistence: Optional[int] = None
    static_threshold: Optional[float] = None


class EvidenceConfig(BaseModel):
    """Configuration shape for evidence requirements."""
    min_history_count: int
    min_window_count: int


class StateMachineConfig(BaseModel):
    """Configuration shape for alert state machine transitions."""
    cooldown_windows: int


class DetectorConfig(BaseModel):
    """Root configuration structure for detector pipeline."""
    version: str
    scorer: ScorerConfig
    evidence: EvidenceConfig
    state_machine: StateMachineConfig
    signal_weights: Optional[dict[str, float]] = None


class MerchantConfig(BaseModel):
    """Configuration shape for synthetic merchant benchmark archetype."""
    id: str
    archetype: str


class GeneratorConfig(BaseModel):
    """Root configuration structure for synthetic generator."""
    seed: int
    merchants: list[MerchantConfig] = Field(default_factory=list)


class CostModelConfig(BaseModel):
    """Configuration shape for illustrative business risk evaluation cost model."""
    fp_review_cost: float
    fn_exposure_factor: float


class EvaluationConfig(BaseModel):
    """Root configuration structure for evaluation metrics and horizons."""
    horizons: dict[str, int] = Field(default_factory=dict)
    cost_model: CostModelConfig
