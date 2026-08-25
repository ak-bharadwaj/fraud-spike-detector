"""Script to generate and serialize paired evasion characterization dataset artifacts in data/evasion/."""

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


def generate_and_save_evasion_artifact():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # 1. Generate Control Stream (Concentrated high-rate volume spike at minute 60)
    gen_control = SyntheticStreamGenerator(
        global_seed=3003,
        merchant_configs=[
            {"id": "EVASION_M1", "archetype": "stable"},
            {"id": "EVASION_M2", "archetype": "growing"},
        ],
        clock=VirtualClock(initial_time=st),
    )

    spec_control = AnomalySpec("volume_spike", st + timedelta(minutes=60), 600.0, 8.0, {"rate_multiplier": 8.0})
    gen_control.schedule_anomaly("EVASION_M1", spec_control, "EVT-EVASION-001")

    control_txs, gt_events = gen_control.generate_window(120.0)

    # 2. Generate Evasion Stream (Temporal Dilution: Sub-threshold rate multiplier 1.3x over 30 mins)
    gen_evasion = SyntheticStreamGenerator(
        global_seed=3003,
        merchant_configs=[
            {"id": "EVASION_M1", "archetype": "stable"},
            {"id": "EVASION_M2", "archetype": "growing"},
        ],
        clock=VirtualClock(initial_time=st),
    )

    spec_evasion = AnomalySpec("volume_spike", st + timedelta(minutes=50), 1800.0, 1.3, {"rate_multiplier": 1.3})
    gen_evasion.schedule_anomaly("EVASION_M1", spec_evasion, "EVT-EVASION-001")

    evasion_txs, _ = gen_evasion.generate_window(120.0)

    # Compute independent hashes for control, evasion, ground truth, and combined experiment hash
    control_hash = compute_tx_hash(control_txs)
    evasion_hash = compute_tx_hash(evasion_txs)
    gt_hash = compute_gt_hash(gt_events)
    exp_payload = f"{control_hash}:{evasion_hash}:{gt_hash}"
    experiment_hash = hashlib.sha256(exp_payload.encode("utf-8")).hexdigest()

    manifest_dict = {
        "control_dataset_hash": control_hash,
        "evasion_dataset_hash": evasion_hash,
        "ground_truth_hash": gt_hash,
        "experiment_hash": experiment_hash,
        "generator_version": "1.0.0",
        "seed": 3003,
        "schema_version": "1.0.0",
        "created_at": "2026-08-25T00:00:00Z",
    }

    data_dir = Path("data/evasion")
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = data_dir / "manifest.json"
    control_file = data_dir / "control_transactions.json"
    evasion_file = data_dir / "evasion_transactions.json"
    gt_file = data_dir / "ground_truth.json"

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
    evasion_file.write_text(json.dumps(serialize_txs(evasion_txs), indent=2), encoding="utf-8")

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
        f"Paired evasion dataset generated: control {len(control_txs)} txs, evasion {len(evasion_txs)} txs, "
        f"gt {len(gt_events)} events.\nControl Hash: {control_hash}\nEvasion Hash: {evasion_hash}\nExperiment Hash: {experiment_hash}"
    )


if __name__ == "__main__":
    generate_and_save_evasion_artifact()
