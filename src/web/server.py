"""FastAPI web server for Fraud-Spike Detector web UI & interactive demo.

Provides REST APIs for:
1. Canonical artifact inspection & provenance metadata
2. Interactive real-time detector stream demo (Transactions -> Features -> Baseline -> Scorer -> Confidence -> State Machine -> Alert)
3. SQLite audit trail exploration
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from src.contracts.contracts import Transaction, FrozenDetectorConfig
from src.generator.anomalies import AnomalySpec
from src.generator.stream_generator import SyntheticStreamGenerator
from src.detector.pipeline import StreamingDetectorPipeline
from src.stream.clock import VirtualClock
from src.evaluation.freeze import load_freeze_record


ROOT_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Fraud-Spike Detector Risk Operations Console", version="1.1.0")


class DemoSession:
    """Stateful demo simulation runner."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.reset()

    def reset(self, merchant_id: str = "M1"):
        self.merchant_id = merchant_id
        self.start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        self.clock = VirtualClock(initial_time=self.start_time)
        
        # Configure generator with standard merchant profile
        merchant_specs = [
            {"id": "M1", "archetype": "stable"},
            {"id": "M2", "archetype": "growing"},
            {"id": "M3", "archetype": "volatile"},
            {"id": "DEV_M1", "archetype": "stable"},
            {"id": "HOLDOUT_M1", "archetype": "growing"},
        ]
        self.generator = SyntheticStreamGenerator(self.seed, merchant_specs, self.clock)
        
        # Schedule a sudden volume spike anomaly at minute 5 to demonstrate alert emission & cooldown
        anomaly_spec = AnomalySpec(
            anomaly_type="sudden_volume_spike",
            start_time=self.start_time + timedelta(minutes=5),
            duration_seconds=120.0,
            target_magnitude=6.5,
            parameters={"rate_multiplier": 5.0},
        )
        self.generator.schedule_anomaly("M1", anomaly_spec, event_id="EVT-DEMO-SPIKE-001")

        # Load frozen detector configuration
        self.freeze_record = load_freeze_record(ROOT_DIR / "config" / "freeze_record.json")
        params = self.freeze_record.all_selected_parameters
        self.config = FrozenDetectorConfig(
            scorer=params["scorer"],
            static_threshold=float(params["static_threshold"]),
            persistence=int(params["persistence"]),
            cooldown_windows=int(params["cooldown_windows"]),
            min_history_count=int(params.get("min_history_count", 1)),
            min_window_count=int(params.get("min_window_count", 1)),
            signal_weights=params.get("signal_weights"),
            detector_version=self.freeze_record.detector_version,
        )

        self.pipeline = StreamingDetectorPipeline(
            config=self.config,
            db_path=":memory:",
        )
        self.current_window_index = 0
        self.history: List[Dict[str, Any]] = []


demo_session = DemoSession()


@app.get("/api/status")
def get_status():
    """Return immutable detector version, status, and provenance metadata."""
    fr = demo_session.freeze_record
    return {
        "status": "FROZEN_DEMO",
        "detector_version": fr.detector_version,
        "config_hash": fr.config_hash,
        "development_dataset_hash": fr.development_dataset_hash,
        "seed": fr.seed,
        "selected_scorer": fr.selected_scorer,
        "parameters": fr.all_selected_parameters,
        "defense_only_notice": "Produces risk signals and alerts for human review — never auto-blocks transactions.",
    }


@app.get("/api/artifacts/{category}")
def get_artifact(category: str):
    """Retrieve committed research artifact JSONs."""
    mapping = {
        "report": ROOT_DIR / "artifacts" / "final" / "report.json",
        "realworld": ROOT_DIR / "artifacts" / "realworld" / "report.json",
        "metrics": ROOT_DIR / "artifacts" / "final" / "metrics.json",
        "signal_ablation": ROOT_DIR / "artifacts" / "ablation" / "signal_ablation.json",
        "ewma_tradeoff": ROOT_DIR / "artifacts" / "ablation" / "ewma_tradeoff.json",
        "drift": ROOT_DIR / "artifacts" / "drift" / "holdout_drift.json",
        "evasion": ROOT_DIR / "artifacts" / "evasion" / "holdout_evasion.json",
        "uncertainty": ROOT_DIR / "artifacts" / "uncertainty" / "bootstrap_uncertainty.json",
        "portfolio": ROOT_DIR / "artifacts" / "portfolio" / "portfolio_comparison.json",
        "calibration": ROOT_DIR / "artifacts" / "calibration" / "holdout_calibration.json",
        "robustness": ROOT_DIR / "artifacts" / "robustness" / "data_quality_characterization.json",
    }
    path = mapping.get(category.lower())
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{category}' not found.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/audit")
def get_audit(merchant_id: Optional[str] = None):
    """Query SQLite audit trail records and state transitions."""
    audits = demo_session.pipeline.audit_store.get_audit_records(merchant_id=merchant_id or "M1")
    alerts = demo_session.pipeline.audit_store.get_alerts(merchant_id=merchant_id or "M1")
    return {
        "merchant_id": merchant_id or "M1",
        "audit_record_count": len(audits),
        "alert_count": len(alerts),
        "audit_records": audits,
        "alerts": [a.model_dump(mode="json") for a in alerts],
    }


@app.post("/api/demo/reset")
def reset_demo(merchant_id: str = Query("M1")):
    """Reset live simulation demo runner to initial state."""
    demo_session.reset(merchant_id=merchant_id)
    return {"status": "SUCCESS", "message": f"Demo reset for merchant {merchant_id}."}


@app.post("/api/demo/step")
def step_demo():
    """Advance simulation by 1 minute window through StreamingDetectorPipeline."""
    w_idx = demo_session.current_window_index
    win_start = demo_session.start_time + timedelta(minutes=w_idx)
    win_end = win_start + timedelta(minutes=1)

    # 1. Generate window transactions
    txs, events = demo_session.generator.generate_window(1.0)
    m_txs = [t for t in txs if t.merchant_id == demo_session.merchant_id]

    # 2. Run pipeline
    alerts = demo_session.pipeline.process_transactions(m_txs)

    # 3. Retrieve latest feature, baseline, audit record from pipeline
    audits = demo_session.pipeline.audit_store.get_audit_records(merchant_id=demo_session.merchant_id)
    latest_audit = audits[-1] if audits else None

    # Derive state machine status
    state_machine_status = demo_session.pipeline.state_machine.get_merchant_state(demo_session.merchant_id)

    # Synthesize plain-English explanation ("What happened?")
    volume = len(m_txs)
    explanation = ""
    if state_machine_status == "ALERT":
        explanation = f"CRITICAL: Merchant {demo_session.merchant_id} volume ({volume} txs/min) breached static threshold 5.00σ with 1-window persistence (P=1). Alert emitted to SQLite audit trail!"
    elif state_machine_status == "COOLDOWN":
        explanation = f"COOLDOWN ACTIVE: Alert recently emitted for {demo_session.merchant_id}. Suppressing redundant alerts for 5 consecutive normal windows to prevent operational fatigue."
    else:
        explanation = f"NORMAL OPERATION: Merchant {demo_session.merchant_id} operating near learned statistical baseline ({volume} txs/min)."

    step_data = {
        "window_index": w_idx,
        "timestamp": win_start.isoformat(),
        "merchant_id": demo_session.merchant_id,
        "transaction_count": volume,
        "transactions": [t.model_dump(mode="json") for t in m_txs[:20]],
        "audit": latest_audit,
        "alerts_emitted": [a.model_dump(mode="json") for a in alerts],
        "state_machine_status": state_machine_status,
        "explanation": explanation,
        "pipeline_stages": {
            "transactions": len(m_txs),
            "features_extracted": True,
            "baseline_computed": True,
            "scorer_executed": True,
            "confidence": latest_audit.get("confidence") if latest_audit else 1.0,
            "state": state_machine_status,
            "alert_fired": len(alerts) > 0,
        },
    }

    demo_session.history.append(step_data)
    demo_session.current_window_index += 1

    return step_data


# Serve static web frontend files
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run_server(port: int = 8000):
    """Start uvicorn server for local demonstration."""
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    run_server()
