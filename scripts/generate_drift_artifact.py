"""Script to generate and serialize paired drift characterization dataset artifacts in data/drift/."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock
from src.evaluation.holdout import HoldoutManifest, compute_holdout_dataset_hash


def generate_and_save_drift_artifact():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # 1. Generate Control Stream (No drift)
    gen_control = SyntheticStreamGenerator(
        global_seed=2002,
        merchant_configs=[
            {"id": "DRIFT_M1", "archetype": "stable"},
            {"id": "DRIFT_M2", "archetype": "growing"},
        ],
        clock=VirtualClock(initial_time=st),
    )

    spec1 = AnomalySpec("volume_spike", st + timedelta(minutes=10), 300.0, 4.0, {"rate_multiplier": 3.0})
    gen_control.schedule_anomaly("DRIFT_M1", spec1, "EVT-DRIFT-001")

    spec2 = AnomalySpec("velocity_spike", st + timedelta(minutes=60), 300.0, 4.0, {"rate_multiplier": 4.0})
    gen_control.schedule_anomaly("DRIFT_M1", spec2, "EVT-DRIFT-002")

    control_txs, gt_events_control = gen_control.generate_window(120.0)

    # 2. Generate Drifted Stream (2.5x volume surge starting at minute 40)
    gen_drift = SyntheticStreamGenerator(
        global_seed=2002,
        merchant_configs=[
            {"id": "DRIFT_M1", "archetype": "stable"},
            {"id": "DRIFT_M2", "archetype": "growing"},
        ],
        clock=VirtualClock(initial_time=st),
    )

    spec1_d = AnomalySpec("volume_spike", st + timedelta(minutes=10), 300.0, 4.0, {"rate_multiplier": 3.0})
    gen_drift.schedule_anomaly("DRIFT_M1", spec1_d, "EVT-DRIFT-001")

    spec2_d = AnomalySpec("velocity_spike", st + timedelta(minutes=60), 300.0, 4.0, {"rate_multiplier": 4.0})
    gen_drift.schedule_anomaly("DRIFT_M1", spec2_d, "EVT-DRIFT-002")

    txs1_d, gt1_d = gen_drift.generate_window(40.0)
    txs2_d, gt2_d = gen_drift.generate_window(80.0, is_surge_active={"DRIFT_M1": True})

    drifted_txs = txs1_d + txs2_d

    # Compute combined hash over control, drifted, and ground truth
    dataset_hash = compute_holdout_dataset_hash(control_txs + drifted_txs, gt_events_control)

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
    control_file = data_dir / "control_transactions.json"
    drifted_file = data_dir / "drifted_transactions.json"
    tx_file = data_dir / "transactions.json"
    gt_file = data_dir / "ground_truth.json"

    manifest_file.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")

    def serialize_txs(tx_list):
        return [
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
            for t in tx_list
        ]

    control_file.write_text(json.dumps(serialize_txs(control_txs), indent=2), encoding="utf-8")
    drifted_file.write_text(json.dumps(serialize_txs(drifted_txs), indent=2), encoding="utf-8")
    tx_file.write_text(json.dumps(serialize_txs(drifted_txs), indent=2), encoding="utf-8")

    gt_raw = [
        {
            "id": e.event_id,
            "m_id": e.merchant_id,
            "type": e.anomaly_type,
            "st": e.start_time.isoformat(),
            "et": e.end_time.isoformat(),
            "sev": float(e.severity),
        }
        for e in gt_events_control
    ]
    gt_file.write_text(json.dumps(gt_raw, indent=2), encoding="utf-8")

    print(
        f"Paired drift dataset generated: control {len(control_txs)} txs, drifted {len(drifted_txs)} txs, "
        f"gt {len(gt_events_control)} events. Hash: {dataset_hash}"
    )


if __name__ == "__main__":
    generate_and_save_drift_artifact()
