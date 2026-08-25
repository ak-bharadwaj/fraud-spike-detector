"""SQLite persistence and audit store implementation (Section 9).

Key Invariants:
- Implements tables for alerts, audit_records, state_transitions, experiments.
- Nested Pydantic list/dict fields (features, baseline, triggered_signals, metrics, costs) JSON-serialized to SQLite TEXT columns.
- Supports in-memory database (:memory:) and persistent file-backed SQLite database.
- Alert risk_score non-nullable; AuditRecord risk_score float | None.
- Scorer error audit records persisted with risk_score = None and data_quality_status = "SCORER_ERROR".
- Audit trail queries supported for live review and pitch rehearsal.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import sqlite3
from pathlib import Path

from src.contracts.contracts import Alert, AuditRecord


class SQLiteAuditStore:
    """SQLite database manager for alerts, audit records, state transitions, and experiment metadata."""

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """Initialize SQLite connection and execute table schema creation."""
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create required SQLite tables if they do not exist (Section 9)."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    triggered_signals TEXT NOT NULL,
                    detector_version TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_records (
                    alert_id TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    risk_score REAL,
                    confidence REAL NOT NULL,
                    features TEXT NOT NULL,
                    baseline TEXT NOT NULL,
                    triggered_signals TEXT NOT NULL,
                    detector_version TEXT NOT NULL,
                    data_quality_status TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    merchant_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    risk_score REAL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    config_hash TEXT NOT NULL,
                    detector_version TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    costs TEXT NOT NULL
                )
                """
            )

    def save_alert(self, alert: Alert) -> None:
        """Save Alert record to SQLite database."""
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    alert_id, merchant_id, timestamp, risk_score, confidence,
                    reason, triggered_signals, detector_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.merchant_id,
                    alert.timestamp.isoformat(),
                    float(alert.risk_score),
                    float(alert.confidence),
                    alert.reason,
                    json.dumps(alert.triggered_signals),
                    alert.detector_version,
                ),
            )

    def save_audit_record(self, audit: AuditRecord) -> None:
        """Save AuditRecord to SQLite database."""
        r_score = float(audit.risk_score) if audit.risk_score is not None else None
        
        # Serialize features and baseline cleanly supporting datetime fields
        if hasattr(audit.features, "model_dump"):
            feat_dict = audit.features.model_dump(mode="json")
        elif isinstance(audit.features, dict):
            feat_dict = audit.features
        else:
            feat_dict = dict(audit.features)

        if hasattr(audit.baseline, "model_dump"):
            base_dict = audit.baseline.model_dump(mode="json")
        elif isinstance(audit.baseline, dict):
            base_dict = audit.baseline
        else:
            base_dict = dict(audit.baseline)

        features_json = json.dumps(feat_dict, default=str)
        baseline_json = json.dumps(base_dict, default=str)
        alert_id_val = audit.audit_id

        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO audit_records (
                    alert_id, merchant_id, timestamp, risk_score, confidence,
                    features, baseline, triggered_signals, detector_version, data_quality_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id_val,
                    audit.merchant_id,
                    audit.timestamp.isoformat(),
                    r_score,
                    float(audit.confidence),
                    features_json,
                    baseline_json,
                    json.dumps(audit.triggered_signals),
                    audit.detector_version,
                    audit.data_quality_status,
                ),
            )

    def save_state_transition(
        self,
        merchant_id: str,
        timestamp: datetime,
        previous_state: str,
        new_state: str,
        reason: str,
        risk_score: Optional[float] = None,
    ) -> None:
        """Save AlertStateMachine state transition to SQLite database."""
        r_score = float(risk_score) if risk_score is not None else None
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO state_transitions (
                    merchant_id, timestamp, previous_state, new_state, reason, risk_score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    merchant_id,
                    timestamp.isoformat(),
                    previous_state,
                    new_state,
                    reason,
                    r_score,
                ),
            )

    def save_experiment(
        self,
        experiment_id: str,
        dataset_id: str,
        dataset_hash: str,
        seed: int,
        config_hash: str,
        detector_version: str,
        metrics: Dict[str, Any],
        costs: Dict[str, Any],
    ) -> None:
        """Save Experiment metadata record to SQLite database."""
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, dataset_id, dataset_hash, seed, config_hash,
                    detector_version, metrics, costs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    dataset_id,
                    dataset_hash,
                    int(seed),
                    config_hash,
                    detector_version,
                    json.dumps(metrics, default=str),
                    json.dumps(costs, default=str),
                ),
            )

    def get_alerts(self, merchant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query alert records from SQLite database."""
        query = "SELECT * FROM alerts"
        params: List[Any] = []
        if merchant_id:
            query += " WHERE merchant_id = ?"
            params.append(merchant_id)
        query += " ORDER BY timestamp ASC"

        cur = self.conn.execute(query, params)
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["triggered_signals"] = json.loads(d["triggered_signals"])
            result.append(d)
        return result

    def get_audit_records(self, merchant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query audit records from SQLite database."""
        query = "SELECT * FROM audit_records"
        params: List[Any] = []
        if merchant_id:
            query += " WHERE merchant_id = ?"
            params.append(merchant_id)
        query += " ORDER BY timestamp ASC"

        cur = self.conn.execute(query, params)
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["features"] = json.loads(d["features"])
            d["baseline"] = json.loads(d["baseline"])
            d["triggered_signals"] = json.loads(d["triggered_signals"])
            result.append(d)
        return result

    def get_state_transitions(self, merchant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query state transitions from SQLite database."""
        query = "SELECT * FROM state_transitions"
        params: List[Any] = []
        if merchant_id:
            query += " WHERE merchant_id = ?"
            params.append(merchant_id)
        query += " ORDER BY transition_id ASC"

        cur = self.conn.execute(query, params)
        return [dict(r) for r in cur.fetchall()]

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Query experiment metadata records from SQLite database."""
        cur = self.conn.execute("SELECT * FROM experiments ORDER BY experiment_id ASC")
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"])
            d["costs"] = json.loads(d["costs"])
            result.append(d)
        return result

    def close(self) -> None:
        """Close SQLite database connection."""
        self.conn.close()
