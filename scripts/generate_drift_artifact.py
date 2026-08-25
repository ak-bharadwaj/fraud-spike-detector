"""Script to generate and serialize paired drift characterization dataset artifacts in data/drift/."""

from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path

from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock


def compute_tx_hash(transactions) -> str:
    tx_list = [
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
        for t in sorted(transactions, key=lambda x: (x.timestamp, x.transaction_id))
    ]
    serialized = json.dumps(tx_list, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_gt_hash(gt_events) -> str:
    gt_list = [
        {
            "id": e.event_id,
            "m_id": e.merchant_id,
            "type": e.anomaly_type,
            "st": e.start_time.isoformat(),
            "et": e.end_time.isoformat(),
            "sev": float(e.severity),
        }
        for e in sorted(gt_events, key=lambda x: (x.start_time, x.event_id))
    ]
    serialized = json.dumps(gt_list, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def generate_and_save_drift_artifact():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    onset_time = st + timedelta(minutes=70.0)

    # 1. Generate Control Base Stream (No volume drift)
    gen_control = SyntheticStreamGenerator(
        global_seed=2002,
        merchant_configs=[
            {"id": "DRIFT_M1", "archetype": "stable"},
            {"id": "DRIFT_M2", "archetype": "growing"},
        ],
        clock=VirtualClock(initial_time=st),
    )

    # Anomaly 1 at minute 52 (post 50-window warm-up)
    spec1 = AnomalySpec("volume_spike", st + timedelta(minutes=52), 600.0, 8.0, {"rate_multiplier": 8.0})
    gen_control.schedule_anomaly("DRIFT_M1", spec1, "EVT-DRIFT-001")

    # Anomaly 2 at minute 100 (during drifted regime)
    spec2 = AnomalySpec("velocity_spike", st + timedelta(minutes=100), 600.0, 8.0, {"rate_multiplier": 8.0})
    gen_control.schedule_anomaly("DRIFT_M1", spec2, "EVT-DRIFT-002")

    control_txs, gt_events = gen_control.generate_window(150.0)

    # 2. Derive Drifted Stream by keeping pre-onset txs & unaffected merchant txs IDENTICAL
    pre_onset_txs = [t for t in control_txs if t.timestamp < onset_time]
    post_onset_m2_txs = [t for t in control_txs if t.timestamp >= onset_time and t.merchant_id == "DRIFT_M2"]

    # Generate additional surge transactions for DRIFT_M1 post-onset (starting strictly at onset_time = minute 70)
    gen_surge = SyntheticStreamGenerator(
        global_seed=3003,
        merchant_configs=[
            {"id": "DRIFT_M1", "archetype": "stable"},
        ],
        clock=VirtualClock(initial_time=onset_time),
    )

    # Schedule identical post-onset anomaly for surge generator
    spec2_s = AnomalySpec("velocity_spike", st + timedelta(minutes=100), 600.0, 8.0, {"rate_multiplier": 8.0})
    gen_surge.schedule_anomaly("DRIFT_M1", spec2_s, "EVT-DRIFT-002")

    surge_txs_all, _ = gen_surge.generate_window(80.0, is_surge_active={"DRIFT_M1": True})
    post_onset_m1_surge_txs = [t for t in surge_txs_all if t.merchant_id == "DRIFT_M1" and t.timestamp >= onset_time]

    drifted_txs = sorted(pre_onset_txs + post_onset_m2_txs + post_onset_m1_surge_txs, key=lambda x: x.timestamp)

    # Compute independent hashes for control, drifted, ground truth, and combined experiment hash
    control_hash = compute_tx_hash(control_txs)
    drifted_hash = compute_tx_hash(drifted_txs)
    gt_hash = compute_gt_hash(gt_events)
    exp_payload = f"{control_hash}:{drifted_hash}:{gt_hash}"
    experiment_hash = hashlib.sha256(exp_payload.encode("utf-8")).hexdigest()

    manifest_dict = {
        "control_dataset_hash": control_hash,
        "drifted_dataset_hash": drifted_hash,
        "ground_truth_hash": gt_hash,
        "experiment_hash": experiment_hash,
        "generator_version": "1.0.0",
        "seed": 2002,
        "schema_version": "1.0.0",
        "created_at": "2026-08-25T00:00:00Z",
    }

    data_dir = Path("data/drift")
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = data_dir / "manifest.json"
    control_file = data_dir / "control_transactions.json"
    drifted_file = data_dir / "drifted_transactions.json"
    gt_file = data_dir / "ground_truth.json"
    alias_file = data_dir / "transactions.json"

    if alias_file.exists():
        alias_file.unlink()

    manifest_file.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")

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

    print(
        f"Paired drift dataset generated: control {len(control_txs)} txs, drifted {len(drifted_txs)} txs, "
        f"gt {len(gt_events)} events.\nControl Hash: {control_hash}\nDrifted Hash: {drifted_hash}\nExperiment Hash: {experiment_hash}"
    )


if __name__ == "__main__":
    generate_and_save_drift_artifact()
