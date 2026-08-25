"""Script to generate and serialize the drift characterization dataset artifact in data/drift/."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock
from src.evaluation.holdout import HoldoutManifest, compute_holdout_dataset_hash


def generate_and_save_drift_artifact():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(
        global_seed=2002,
        merchant_configs=[
            {"id": "DRIFT_M1", "archetype": "stable"},
            {"id": "DRIFT_M2", "archetype": "growing"},
        ],
        clock=VirtualClock(initial_time=st),
    )

    # Fraud anomaly during non-drifted period
    spec1 = AnomalySpec("volume_spike", st + timedelta(minutes=10), 300.0, 4.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("DRIFT_M1", spec1, "EVT-DRIFT-001")

    # Fraud anomaly during volume surge drift period
    spec2 = AnomalySpec("velocity_spike", st + timedelta(minutes=60), 300.0, 4.0, {"rate_multiplier": 4.0})
    gen.schedule_anomaly("DRIFT_M1", spec2, "EVT-DRIFT-002")

    txs1, gt_events1 = gen.generate_window(40.0)
    txs2, gt_events2 = gen.generate_window(80.0, is_surge_active={"DRIFT_M1": True})

    txs = txs1 + txs2
    gt_events = gt_events1 + gt_events2

    dataset_hash = compute_holdout_dataset_hash(txs, gt_events)
    manifest = HoldoutManifest(
        dataset_hash=dataset_hash,
        generator_version="1.0.0",
        seed=2002,
        schema_version="1.0.0",
        created_at="2026-08-25T00:00:00Z",
    )

    data_dir = Path("data/drift")
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = data_dir / "manifest.json"
    tx_file = data_dir / "transactions.json"
    gt_file = data_dir / "ground_truth.json"

    manifest_file.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")

    tx_raw = [
        {
            "id": t.transaction_id,
            "ts": t.timestamp.isoformat(),
            "m_id": t.merchant_id,
            "c_id": t.customer_id,
            "amt": float(t.amount),
            "pm": t.payment_method,
            "country": t.country,
            "d_id": t.device_id,
        }
        for t in txs
    ]
    tx_file.write_text(json.dumps(tx_raw, indent=2), encoding="utf-8")

    gt_raw = [
        {
            "id": e.event_id,
            "m_id": e.merchant_id,
            "type": e.anomaly_type,
            "st": e.start_time.isoformat(),
            "et": e.end_time.isoformat(),
            "sev": float(e.severity),
        }
        for e in gt_events

    ]
    gt_file.write_text(json.dumps(gt_raw, indent=2), encoding="utf-8")

    print(f"Drift dataset generated: {len(txs)} txs, {len(gt_events)} events. Hash: {dataset_hash}")


if __name__ == "__main__":
    generate_and_save_drift_artifact()
