"""StreamingDetectorPipeline module for orchestrating end-to-end streaming anomaly detection.

Key Invariants:
- Orchestrates: Transaction -> TimeOrderedEventBus -> FeatureEngine -> BaselineEngine -> HybridEWMAScorer -> AlertStateMachine -> SQLite.
- Historical-only baseline: BaselineEngine.get_baseline() is invoked BEFORE updating BaselineEngine with current window features.
- Section 20 Scorer Exception Path:
  - On scorer success: RiskScore -> StateMachine -> Alert + AuditRecord saved to SQLite.
  - On scorer exception: Error AuditRecord saved to SQLite (risk_score=None, data_quality_status="SCORER_ERROR"); NO Alert emitted; NO ALERT state transition; stream continues.
- Zero ground-truth imports across src/detector/ package (strictly enforced by architecture AST tests).
- VirtualClock remains sole time authority.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

from src.contracts.contracts import (
    Transaction,
    FeatureSnapshot,
    BaselineSnapshot,
    RiskScore,
    Alert,
    AuditRecord,
    FrozenDetectorConfig,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.audit.database import SQLiteAuditStore
from src.stream.clock import VirtualClock
from src.stream.bus import TimeOrderedEventBus


class StreamingDetectorPipeline:
    """End-to-end streaming detector pipeline orchestrator."""

    def __init__(
        self,
        config: Optional[FrozenDetectorConfig] = None,
        db_path: Union[str, Path] = ":memory:",
        clock: Optional[VirtualClock] = None,
    ):
        """Initialize pipeline components with frozen configuration and SQLite audit store."""
        self.config = config if config is not None else FrozenDetectorConfig()
        self.clock = clock or VirtualClock()
        self.bus = TimeOrderedEventBus(clock=self.clock)
        self.audit_store = SQLiteAuditStore(db_path=db_path)

        self.feature_engine = FeatureEngine()
        self.baseline_engine = BaselineEngine(
            min_history_count=self.config.min_window_count,
            min_window_count=self.config.min_window_count,
        )
        self.scorer = HybridEWMAScorer(alpha=self.config.ewma_alpha)
        self.state_machine = AlertStateMachine(
            persistence=self.config.persistence,
            cooldown_windows=self.config.cooldown_windows,
            static_threshold=self.config.static_threshold,
        )

    def process_transactions(self, transactions: List[Transaction]) -> List[Alert]:
        """Publish transaction stream onto bus and execute window-by-window streaming detection."""
        self.bus.clear()
        self.bus.publish_batch(transactions)

        ordered_txs = self.bus.get_ordered_events()
        if not ordered_txs:
            return []

        # Group transactions by merchant_id
        tx_by_merchant: Dict[str, List[Transaction]] = {}
        for tx in ordered_txs:
            tx_by_merchant.setdefault(tx.merchant_id, []).append(tx)

        emitted_alerts: List[Alert] = []

        for merchant_id in sorted(tx_by_merchant.keys()):
            m_txs = tx_by_merchant[merchant_id]
            if not m_txs:
                continue

            start_time = m_txs[0].timestamp
            end_time = m_txs[-1].timestamp
            curr_window_start = start_time

            while curr_window_start <= end_time:
                curr_window_end = curr_window_start + timedelta(minutes=self.feature_engine.window_duration_minutes)
                curr_txs = [t for t in m_txs if curr_window_start <= t.timestamp < curr_window_end]

                # 1. Feature extraction
                feat_snap = self.feature_engine.extract_snapshot(
                    merchant_id, curr_txs, curr_window_start, curr_window_end
                )

                # 2. Historical-only baseline extraction (BEFORE updating baseline!)
                base_snap = self.baseline_engine.get_baseline(merchant_id, feat_snap)

                # 3. Scorer calculation with Section 20 exception handling
                try:
                    risk_score = self.scorer.calculate_score(feat_snap, base_snap)

                    # Update baseline after scoring
                    self.baseline_engine.update(feat_snap)

                    prev_state = self.state_machine.get_merchant_state(merchant_id)
                    new_state, alert = self.state_machine.process_score(merchant_id, curr_window_end, risk_score)

                    # Track state transitions
                    if new_state != prev_state:
                        self.audit_store.save_state_transition(
                            merchant_id=merchant_id,
                            timestamp=curr_window_end,
                            previous_state=prev_state,
                            new_state=new_state,
                            reason=f"Score {risk_score.score} evaluated against threshold {self.config.static_threshold}",
                            risk_score=risk_score.score,
                        )

                    # Save Alert if generated
                    if alert is not None:
                        emitted_alerts.append(alert)
                        self.audit_store.save_alert(alert)

                    # Save standard AuditRecord
                    audit_record = AuditRecord(
                        audit_id=alert.alert_id if alert else f"AUDIT-{uuid.uuid4().hex[:16]}",
                        alert_id=alert.alert_id if alert else None,
                        merchant_id=merchant_id,
                        timestamp=curr_window_end,
                        risk_score=risk_score.score,
                        confidence=risk_score.confidence,
                        features=feat_snap,
                        baseline=base_snap,
                        triggered_signals=risk_score.triggered_signals,
                        detector_version=self.config.detector_version,
                        data_quality_status=risk_score.data_quality,
                    )
                    self.audit_store.save_audit_record(audit_record)

                except Exception as err:
                    # Section 20 Scorer Exception Path:
                    # Error AuditRecord saved to SQLite; NO Alert; NO ALERT state transition; stream continues.
                    err_audit = AuditRecord(
                        audit_id=f"ERR-{uuid.uuid4().hex[:16]}",
                        alert_id=None,
                        merchant_id=merchant_id,
                        timestamp=curr_window_end,
                        risk_score=None,
                        confidence=0.0,
                        features=feat_snap,
                        baseline=base_snap,
                        triggered_signals=[f"EXCEPT:{type(err).__name__}:{str(err)}"],
                        detector_version=self.config.detector_version,
                        data_quality_status="SCORER_ERROR",
                    )
                    self.audit_store.save_audit_record(err_audit)

                curr_window_start = curr_window_end

        return emitted_alerts
