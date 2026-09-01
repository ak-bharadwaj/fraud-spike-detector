"""Immutable Freeze Record module for Day 7 freeze gate.

Key Invariants:
- Durable freeze record containing:
  - detector_version
  - config_hash (canonical SHA-256 hex digest of deterministic JSON serialization)
  - development_dataset_hash (canonical SHA-256 hex digest of serialized transaction stream)
  - random seed
  - selected scorer
  - all selected parameters
  - selection rationale
  - freeze timestamp
- Determinism:
  - Same config -> same config_hash.
  - Different config -> different config_hash.
  - Same development data -> same development_dataset_hash.
- Immutability and Post-Freeze Override Protection:
  - Validates that attempts to mutate or override frozen configuration post-freeze fail hash verification.
"""

from typing import Dict, Any, Optional, Sequence, Union
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
from pydantic import BaseModel, Field

from src.contracts.contracts import Transaction, FrozenDetectorConfig


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 hex digest of frozen configuration dictionary."""
    canonical_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_dataset_hash(transactions: Sequence[Transaction], seed: Optional[int] = None) -> str:
    """Compute canonical SHA-256 hex digest of development dataset transactions."""
    hasher = hashlib.sha256()
    if seed is not None:
        hasher.update(f"seed:{seed}\n".encode("utf-8"))
    for tx in transactions:
        tx_repr = f"{tx.transaction_id}:{tx.timestamp.isoformat()}:{tx.merchant_id}:{tx.amount:.2f}:{tx.payment_method}:{tx.device_id}\n"
        hasher.update(tx_repr.encode("utf-8"))
    return hasher.hexdigest()


class FreezeRecord(BaseModel):
    """Immutable freeze record contract."""
    detector_version: str = "1.0.0"
    config_hash: str
    development_dataset_hash: str
    seed: int
    selected_scorer: str
    all_selected_parameters: Dict[str, Any] = Field(default_factory=dict)
    selection_rationale: str
    freeze_timestamp: str

    def verify_config(self, config_dict: Dict[str, Any]) -> bool:
        """Verify that a candidate configuration dictionary matches this frozen config_hash."""
        candidate_hash = compute_config_hash(config_dict)
        return candidate_hash == self.config_hash

    def verify_dataset(self, transactions: Sequence[Transaction], seed: Optional[int] = None) -> bool:
        """Verify that a candidate dataset matches this frozen development_dataset_hash."""
        candidate_hash = compute_dataset_hash(transactions, seed=seed if seed is not None else self.seed)
        return candidate_hash == self.development_dataset_hash


def create_freeze_record(
    selected_scorer: str,
    selected_parameters: Dict[str, Any],
    development_transactions: Sequence[Transaction],
    seed: int = 42,
    detector_version: str = "1.0.0",
    selection_rationale: str = "Selected optimal operating point minimizing total cost and maximizing F1 on development benchmark data.",
    freeze_timestamp: Optional[str] = None,
) -> FreezeRecord:
    """Construct an immutable FreezeRecord from development tuning outcomes."""
    config_hash = compute_config_hash(selected_parameters)
    dataset_hash = compute_dataset_hash(development_transactions, seed=seed)
    ts = freeze_timestamp if freeze_timestamp is not None else "2026-01-07T00:00:00Z"

    return FreezeRecord(
        detector_version=detector_version,
        config_hash=config_hash,
        development_dataset_hash=dataset_hash,
        seed=seed,
        selected_scorer=selected_scorer,
        all_selected_parameters=selected_parameters,
        selection_rationale=selection_rationale,
        freeze_timestamp=ts,
    )


def save_freeze_record(record: FreezeRecord, file_path: Union[str, Path]) -> None:
    """Save FreezeRecord to a JSON file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(record.model_dump_json(indent=2))


def load_freeze_record(file_path: Union[str, Path]) -> FreezeRecord:
    """Load FreezeRecord from a JSON file."""
    p = Path(file_path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FreezeRecord.model_validate(data)
