"""Core data contracts for Fraud-Spike Detector using Pydantic.

Preserves exact frozen nullability requirements:
- RiskScore.score: float | None
- AuditRecord.risk_score: float | None
- Alert.risk_score: float (non-nullable)
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str
    timestamp: datetime
    merchant_id: str
    customer_id: str
    amount: float
    payment_method: str
    country: str
    device_id: str


class GroundTruthEvent(BaseModel):
    event_id: str
    merchant_id: str
    anomaly_type: str
    start_time: datetime
    end_time: datetime
    severity: float
    parameters: dict[str, Any] = Field(default_factory=dict)


class FeatureSnapshot(BaseModel):
    merchant_id: str
    timestamp: datetime
    volume: float
    velocity: float
    amount_statistics: dict[str, float] = Field(default_factory=dict)
    unique_customers: int
    unique_devices: int
    data_quality: str


class BaselineSnapshot(BaseModel):
    merchant_id: str
    timestamp: datetime
    expected_values: dict[str, float] = Field(default_factory=dict)
    robust_scale: dict[str, float] = Field(default_factory=dict)
    history_count: int
    current_window_count: int
    evidence_state: str  # SUFFICIENT, DEGRADED, INSUFFICIENT


class RiskScore(BaseModel):
    score: Optional[float] = None  # Explicit float | None requirement
    confidence: float
    triggered_signals: list[str] = Field(default_factory=list)
    data_quality: str = "GOOD"


class Alert(BaseModel):
    alert_id: str
    merchant_id: str
    timestamp: datetime
    risk_score: float  # Explicit non-nullable float requirement
    confidence: float
    reason: str
    triggered_signals: list[str] = Field(default_factory=list)
    detector_version: str


class AuditRecord(BaseModel):
    audit_id: str
    alert_id: Optional[str] = None
    merchant_id: str
    timestamp: datetime
    risk_score: Optional[float] = None  # Explicit float | None requirement
    confidence: float
    features: dict[str, Any] = Field(default_factory=dict)
    baseline: dict[str, Any] = Field(default_factory=dict)
    triggered_signals: list[str] = Field(default_factory=list)
    detector_version: str
    data_quality_status: str
