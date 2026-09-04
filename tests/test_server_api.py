"""Unit tests for FastAPI web server endpoints and frontend-backend connectivity."""

import pytest
from fastapi.testclient import TestClient
from src.web.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_server_status_endpoint(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FROZEN_DEMO"
    assert data["detector_version"] == "1.1.0"


def test_server_static_frontend_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Fraud-Spike Detector" in response.text


def test_server_audit_endpoint_empty_and_after_steps(client):
    # Initial audit fetch
    response = client.get("/api/audit?merchant_id=M1")
    assert response.status_code == 200
    data = response.json()
    assert data["merchant_id"] == "M1"
    assert "audit_records" in data
    assert "alerts" in data

    # Reset demo
    reset_res = client.post("/api/demo/reset?merchant_id=M1")
    assert reset_res.status_code == 200

    # Step demo through multiple windows to generate audit records and alerts
    for _ in range(7):
        step_res = client.post("/api/demo/step")
        assert step_res.status_code == 200
        step_data = step_res.json()
        assert "audit" in step_data

    # Query audit trail again (now populated with records and alerts)
    audit_res = client.get("/api/audit?merchant_id=M1")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["audit_record_count"] >= 7
    assert isinstance(audit_data["alerts"], list)


def test_server_artifact_endpoints(client):
    categories = [
        "report",
        "realworld",
        "metrics",
        "signal_ablation",
        "ewma_tradeoff",
        "drift",
        "evasion",
        "uncertainty",
        "portfolio",
        "calibration",
        "robustness",
    ]
    for cat in categories:
        res = client.get(f"/api/artifacts/{cat}")
        assert res.status_code == 200, f"Failed fetching artifact category {cat}"

    # Invalid artifact category
    res_404 = client.get("/api/artifacts/unknown_category")
    assert res_404.status_code == 404
