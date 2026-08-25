"""Tests for core Pydantic data contracts and nullability invariants."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    FeatureSnapshot,
    BaselineSnapshot,
    RiskScore,
    Alert,
    AuditRecord,
)


def test_risk_score_nullability():
    """Verify RiskScore.score allows None (nullable float)."""
    rs_none = RiskScore(score=None, confidence=0.8, triggered_signals=["SIG1"], data_quality="GOOD")
    assert rs_none.score is None

    rs_val = RiskScore(score=0.75, confidence=0.9, triggered_signals=["SIG1"], data_quality="GOOD")
    assert rs_val.score == 0.75


def test_alert_risk_score_non_nullable():
    """Verify Alert.risk_score requires non-nullable float and fails if None."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    alert = Alert(
        alert_id="ALT-001",
        merchant_id="M1",
        timestamp=now,
        risk_score=0.85,
        confidence=0.95,
        reason="Volume spike detected",
        triggered_signals=["VOLUME_SPIKE"],
        detector_version="1.0.0",
    )
    assert alert.risk_score == 0.85

    with pytest.raises(ValidationError):
        Alert(
            alert_id="ALT-002",
            merchant_id="M1",
            timestamp=now,
            risk_score=None,  # Should fail
            confidence=0.95,
            reason="Volume spike detected",
            triggered_signals=["VOLUME_SPIKE"],
            detector_version="1.0.0",
        )


def test_audit_record_nullability():
    """Verify AuditRecord.risk_score allows None (nullable float)."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    record_none = AuditRecord(
        audit_id="AUD-001",
        merchant_id="M1",
        timestamp=now,
        risk_score=None,
        confidence=0.0,
        detector_version="1.0.0",
        data_quality_status="INSUFFICIENT_DATA",
    )
    assert record_none.risk_score is None

    record_val = AuditRecord(
        audit_id="AUD-002",
        merchant_id="M1",
        timestamp=now,
        risk_score=0.92,
        confidence=0.95,
        detector_version="1.0.0",
        data_quality_status="GOOD",
    )
    assert record_val.risk_score == 0.92


def test_transaction_contract():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tx = Transaction(
        transaction_id="TX-100",
        timestamp=now,
        merchant_id="M1",
        customer_id="C-1",
        amount=150.50,
        payment_method="CREDIT_CARD",
        country="US",
        device_id="DEV-99",
    )
    assert tx.amount == 150.50


def test_ground_truth_event_contract():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="EVT-1",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=now,
        end_time=now,
        severity=4.5,
        parameters={"multiplier": 3.0},
    )
    assert gt.severity == 4.5
