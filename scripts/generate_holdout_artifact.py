"""Script to generate and serialize the locked holdout dataset artifact in data/holdout/."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock
from src.evaluation.holdout import HoldoutManifest, compute_holdout_dataset_hash


def generate_and_save_holdout_artifact():
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(
        global_seed=4242,
        merchant_configs=[{"id": "HOLDOUT_M1", "archetype": "stable"}],
        clock=VirtualClock(initial_time=st),
    )

    # 1. Primary volume spike
    spec1 = AnomalySpec("volume_spike", st + timedelta(minutes=10), 300.0, 4.0, {"rate_multiplier": 3.0})
    gen.schedule_anomaly("HOLDOUT_M1", spec1, "EVT-HOLDOUT-001")

    # 2. Threshold hugging evasion
    spec2 = AnomalySpec("threshold_hugging_evasion", st + timedelta(minutes=20), 240.0, 4.8, {"rate_multiplier": 1.75})
    gen.schedule_anomaly("HOLDOUT_M1", spec2, "EVT-HOLDOUT-002")

    # 3. Persistence evasion
    spec3 = AnomalySpec("persistence_evasion", st + timedelta(minutes=30), 240.0, 5.6, {"rate_multiplier": 2.10})
    gen.schedule_anomaly("HOLDOUT_M1", spec3, "EVT-HOLDOUT-003")

    # 4. Staircase ramp
    spec4 = AnomalySpec("staircase_ramp", st + timedelta(minutes=40), 240.0, 6.5, {"rate_multiplier": 7.5})
    gen.schedule_anomaly("HOLDOUT_M1", spec4, "EVT-HOLDOUT-004")

    # 5. Oscillating sub-threshold
    spec5 = AnomalySpec("oscillating_sub_threshold", st + timedelta(minutes=50), 240.0, 4.2, {"amplitude": 0.8, "rate_multiplier": 1.2})
    gen.schedule_anomaly("HOLDOUT_M1", spec5, "EVT-HOLDOUT-005")

    txs, gt_events = gen.generate_window(120.0)

    dataset_hash = compute_holdout_dataset_hash(txs, gt_events)
    manifest = HoldoutManifest(
        dataset_hash=dataset_hash,
        generator_version="1.0.0",
        seed=4242,
        schema_version="1.0.0",
        created_at="2026-08-25T00:00:00Z",
    )

    data_dir = Path("data/holdout")
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

    print(f"Locked holdout generated: {len(txs)} txs, {len(gt_events)} events. Hash: {dataset_hash}")


if __name__ == "__main__":
    generate_and_save_holdout_artifact()
