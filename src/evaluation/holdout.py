"""Locked holdout dataset protection guard, canonical hashing, and evaluation runner.

Key Invariants:
- Locked holdout evaluation runs ONLY when explicit_evaluation_mode=True.
- Canonical dataset hashing: compute_holdout_dataset_hash SHA-256 over deterministic JSON payload.
- Checksum verification: verifies computed actual dataset hash matches HoldoutManifest.dataset_hash; aborts with ChecksumMismatchError on mismatch.
- Historical-only baseline: current window transactions curr_txs are extracted for FeatureSnapshot; baseline_engine.get_baseline() is called BEFORE updating baseline_engine with curr_txs (NO current-window baseline leakage!).
- Structurally frozen detector config: FrozenDetectorConfig encapsulates detector parameters; evaluate_holdout accepts ZERO parameter overrides.
- Emits EvaluationMetrics compliant with Pydantic schema contract.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel, Field

from src.contracts.contracts import Transaction, GroundTruthEvent, Alert, EvaluationMetrics, FrozenDetectorConfig
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.evaluation.evaluator import AnomalyEvaluator


class HoldoutManifest(BaseModel):
    dataset_hash: str
    generator_version: str
    seed: int
    schema_version: str
    created_at: str



class HoldoutAccessError(PermissionError):
    """Raised when holdout is accessed without explicit evaluation mode enabled."""

    pass


class ChecksumMismatchError(ValueError):
    """Raised when holdout dataset hash fails validation against the manifest."""

    pass


def compute_holdout_dataset_hash(
    transactions: List[Transaction],
    ground_truth_events: List[GroundTruthEvent],
) -> str:
    """Compute canonical SHA-256 dataset hash over deterministic JSON serialization."""
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

    gt_list = [
        {
            "id": e.event_id,
            "m_id": e.merchant_id,
            "type": e.anomaly_type,
            "st": e.start_time.isoformat(),
            "et": e.end_time.isoformat(),
            "sev": float(e.severity),
        }
        for e in sorted(ground_truth_events, key=lambda x: (x.start_time, x.event_id))
    ]

    payload = {
        "transactions": tx_list,
        "ground_truth_events": gt_list,
    }

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class HoldoutProtection:
    """Protection guard for holdout dataset access."""

    @staticmethod
    def verify_access(
        manifest: HoldoutManifest,
        actual_dataset_hash: Optional[str] = None,
        explicit_evaluation_mode: bool = False,
        transactions: Optional[List[Transaction]] = None,
        ground_truth_events: Optional[List[GroundTruthEvent]] = None,
    ) -> bool:
        """Verify explicit evaluation mode and dataset hash checksum.

        Raises HoldoutAccessError if explicit_evaluation_mode is False.
        Raises ChecksumMismatchError if dataset hash fails validation.
        """
        if not explicit_evaluation_mode:
            raise HoldoutAccessError(
                "Holdout access denied: Normal development mode cannot access holdout data. "
                "Explicit evaluation mode flag is required."
            )

        if actual_dataset_hash is None and transactions is not None and ground_truth_events is not None:
            actual_dataset_hash = compute_holdout_dataset_hash(transactions, ground_truth_events)

        if actual_dataset_hash != manifest.dataset_hash:
            raise ChecksumMismatchError(
                f"Holdout dataset checksum mismatch! Expected: {manifest.dataset_hash}, "
                f"Actual: {actual_dataset_hash}. Evaluation aborted."
            )

        return True


class HoldoutEvaluator:
    """Single-pass evaluator for running frozen detector pipeline against locked holdout dataset."""

    def __init__(
        self,
        manifest: HoldoutManifest,
        config: Optional[FrozenDetectorConfig] = None,
        explicit_evaluation_mode: bool = False,
    ):
        self.manifest = manifest
        self.config = config if config is not None else FrozenDetectorConfig()
        self.explicit_evaluation_mode = explicit_evaluation_mode

    def evaluate_holdout(
        self,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
    ) -> EvaluationMetrics:
        """Execute single-pass evaluation of frozen detector against locked holdout dataset.

        Requires ZERO parameter overrides at invocation time to guarantee frozen config integrity.
        """
        # 1. Verify access & checksum integrity using canonical hashing
        HoldoutProtection.verify_access(
            manifest=self.manifest,
            transactions=transactions,
            ground_truth_events=ground_truth_events,
            explicit_evaluation_mode=self.explicit_evaluation_mode,
        )

        # 2. Instantiate frozen pipeline components
        feature_engine = FeatureEngine()
        baseline_engine = BaselineEngine(min_window_count=self.config.min_window_count)
        scorer = HybridEWMAScorer(alpha=self.config.ewma_alpha)
        state_machine = AlertStateMachine(
            persistence=self.config.persistence,
            cooldown_windows=self.config.cooldown_windows,
            static_threshold=self.config.static_threshold,
        )

        # 3. Group transactions by merchant
        tx_by_merchant: Dict[str, List[Transaction]] = {}
        for tx in sorted(transactions, key=lambda x: x.timestamp):
            tx_by_merchant.setdefault(tx.merchant_id, []).append(tx)

        alerts: List[Alert] = []

        # Process each merchant stream independently
        for merchant_id in sorted(tx_by_merchant.keys()):
            m_txs = tx_by_merchant[merchant_id]
            if not m_txs:
                continue

            start_time = m_txs[0].timestamp
            end_time = m_txs[-1].timestamp

            # 1-minute window steps
            curr_window_start = start_time

            while curr_window_start <= end_time:
                curr_window_end = curr_window_start + timedelta(minutes=feature_engine.window_duration_minutes)

                # Extract ONLY current window transactions for current FeatureSnapshot
                curr_txs = [t for t in m_txs if curr_window_start <= t.timestamp < curr_window_end]

                # Extract feature snapshot for current window
                feat_snap = feature_engine.extract_snapshot(merchant_id, curr_txs, curr_window_start, curr_window_end)

                # Compute baseline snapshot strictly BEFORE updating baseline with current window!
                base_snap = baseline_engine.get_baseline(merchant_id, feat_snap)

                # Score features against pre-current baseline
                risk_score = scorer.calculate_score(feat_snap, base_snap)

                # Update baseline engine with current snapshot AFTER baseline computation
                baseline_engine.update(feat_snap)

                # Evaluate risk score through alert state machine
                _, alert = state_machine.process_score(merchant_id, curr_window_end, risk_score)
                if alert is not None:
                    alerts.append(alert)

                curr_window_start = curr_window_end

        # 4. Run AnomalyEvaluator against ground truth events
        evaluator = AnomalyEvaluator(temporal_tolerance_seconds=self.config.temporal_tolerance_seconds)
        return evaluator.evaluate(alerts, ground_truth_events)


def load_locked_holdout_data(data_dir: Union[str, Path]) -> Tuple[HoldoutManifest, List[Transaction], List[GroundTruthEvent]]:
    """Load locked holdout manifest, transactions, and ground truth events from stored directory."""
    d_path = Path(data_dir)
    manifest_path = d_path / "manifest.json"
    tx_path = d_path / "transactions.json"
    gt_path = d_path / "ground_truth.json"

    if not manifest_path.exists() or not tx_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Holdout artifact missing in {d_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = HoldoutManifest(**manifest_data)

    tx_raw = json.loads(tx_path.read_text(encoding="utf-8"))
    transactions = [
        Transaction(
            transaction_id=t["id"],
            timestamp=datetime.fromisoformat(t["ts"]),
            merchant_id=t["m_id"],
            customer_id=t["c_id"],
            amount=t["amt"],
            payment_method=t["pm"],
            country=t["country"],
            device_id=t["d_id"],
        )
        for t in tx_raw
    ]

    gt_raw = json.loads(gt_path.read_text(encoding="utf-8"))
    ground_truth_events = [
        GroundTruthEvent(
            event_id=e["id"],
            merchant_id=e["m_id"],
            anomaly_type=e["type"],
            start_time=datetime.fromisoformat(e["st"]),
            end_time=datetime.fromisoformat(e["et"]),
            severity=e["sev"],
        )
        for e in gt_raw
    ]

    return manifest, transactions, ground_truth_events
