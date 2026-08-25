"""Configuration schemas for YAML validation using Pydantic."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ScorerConfig(BaseModel):
    type: str = "HybridEWMAScorer"
    alpha: float = 0.3
    persistence: int = 2
    static_threshold: float = 3.5


class EvidenceConfig(BaseModel):
    min_history_count: int = 50
    min_window_count: int = 5


class StateMachineConfig(BaseModel):
    cooldown_windows: int = 5


class DetectorConfig(BaseModel):
    version: str = "1.0.0"
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    state_machine: StateMachineConfig = Field(default_factory=StateMachineConfig)


class MerchantConfig(BaseModel):
    id: str
    archetype: str


class GeneratorConfig(BaseModel):
    seed: int = 42
    merchants: list[MerchantConfig] = Field(default_factory=list)


class CostModelConfig(BaseModel):
    fp_review_cost: float = 50.0
    fn_exposure_factor: float = 1.0


class EvaluationConfig(BaseModel):
    horizons: dict[str, int] = Field(default_factory=dict)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
