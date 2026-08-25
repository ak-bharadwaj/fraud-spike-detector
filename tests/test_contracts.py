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
    derive_severity_level,
)


def test_ground_truth_event_severity_derivation():
    """Verify severity level derivation (Section 14: LOW < 2, MEDIUM 2..4, HIGH >= 4)."""
    assert derive_severity_level(1.5) == "LOW"
    assert derive_severity_level(2.0) == "MEDIUM"
    assert derive_severity_level(3.9) == "MEDIUM"
    assert derive_severity_level(4.0) == "HIGH"
    assert derive_severity_level(5.5) == "HIGH"

    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Event with magnitude M = 1.2 -> LOW
    gt_low = GroundTruthEvent(
        event_id="EVT-1",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=now,
        end_time=now,
        severity=1.2,
    )
    assert gt_low.severity == 1.2
    assert gt_low.severity_level == "LOW"

    # Event with magnitude M = 4.5 -> HIGH
    gt_high = GroundTruthEvent(
        event_id="EVT-2",
        merchant_id="M1",
        anomaly_type="velocity_burst",
        start_time=now,
        end_time=now,
        severity=4.5,
    )
    assert gt_high.severity == 4.5
    assert gt_high.severity_level == "HIGH"


def test_ground_truth_event_severity_validation_rejection():
    """Verify accepted and rejected GroundTruthEvent severity levels."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Valid combinations (accepted)
    gt_1 = GroundTruthEvent(
        event_id="E1", merchant_id="M1", anomaly_type="spike", start_time=now, end_time=now,
        severity=1.5, severity_level="LOW"
    )
    assert gt_1.severity_level == "LOW"

    gt_2 = GroundTruthEvent(
        event_id="E2", merchant_id="M1", anomaly_type="spike", start_time=now, end_time=now,
        severity=2.0, severity_level="MEDIUM"
    )
    assert gt_2.severity_level == "MEDIUM"

    gt_3 = GroundTruthEvent(
        event_id="E3", merchant_id="M1", anomaly_type="spike", start_time=now, end_time=now,
        severity=4.0, severity_level="HIGH"
    )
    assert gt_3.severity_level == "HIGH"

    # Mismatched combinations (rejected with ValidationError)
    with pytest.raises(ValidationError):
        GroundTruthEvent(
            event_id="E4", merchant_id="M1", anomaly_type="spike", start_time=now, end_time=now,
            severity=1.5, severity_level="HIGH"
        )

    with pytest.raises(ValidationError):
        GroundTruthEvent(
            event_id="E5", merchant_id="M1", anomaly_type="spike", start_time=now, end_time=now,
            severity=2.0, severity_level="LOW"
        )

    with pytest.raises(ValidationError):
        GroundTruthEvent(
            event_id="E6", merchant_id="M1", anomaly_type="spike", start_time=now, end_time=now,
            severity=4.0, severity_level="MEDIUM"
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
