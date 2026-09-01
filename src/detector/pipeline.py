"""StreamingDetectorPipeline module for orchestrating end-to-end streaming anomaly detection.

Key Invariants:
- EventBus-driven execution: process_transactions publishes transactions onto TimeOrderedEventBus and calls bus.drain(handler=pipeline._on_transaction_dispatched).
- Sole time authority: VirtualClock advances monotonically as transactions are drained by the bus.
- Half-open window boundaries: Aligned to minute boundaries [HH:MM:00 .. HH:MM:00 + duration) matching whole-minute generator specifications.
- Historical-only baseline: BaselineEngine.get_baseline() is invoked BEFORE updating BaselineEngine with current window features.
- Scorer-level signal masking: supports ablation studies by passing signal_mask to scorer.calculate_score without perturbing baseline history.
- Section 20 Scorer Exception Path:
  - Scoped STRICTLY to scorer invocation try...except block.
  - On scorer success: RiskScore -> StateMachine -> Alert + AuditRecord saved to SQLite.
  - On scorer exception: Error AuditRecord saved to SQLite (risk_score=None, data_quality_status="SCORER_ERROR"); NO Alert emitted; NO ALERT state transition; stream continues.
- Deterministic IDs: SHA-256 deterministic IDs derived from (merchant_id, timestamp, score/exception) with zero wall-clock or random entropy (no uuid.uuid4()).
- Zero ground-truth imports across src/detector/ package (strictly enforced by architecture AST tests).
"""

from typing import List, Tuple, Dict, Any, Optional, Union, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib

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
from src.scoring.statistical import StatisticalDeviationScorer
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
        scorer: Optional[Any] = None,
        signal_mask: Optional[Sequence[str]] = None,
        db_path: Union[str, Path] = ":memory:",
        clock: Optional[VirtualClock] = None,
    ):
        """Initialize pipeline components with frozen configuration, scorer, optional signal_mask, and SQLite audit store."""
        self.config = config if config is not None else FrozenDetectorConfig()
        self.clock = clock or VirtualClock()
        self.bus = TimeOrderedEventBus(clock=self.clock)
        self.audit_store = SQLiteAuditStore(db_path=db_path)
        self.signal_mask = list(signal_mask) if signal_mask is not None else None

        self.feature_engine = FeatureEngine()
        self.baseline_engine = BaselineEngine(
            min_history_count=self.config.min_window_count,
            min_window_count=self.config.min_window_count,
        )
        self.scorer = scorer if scorer is not None else HybridEWMAScorer(
            alpha=self.config.ewma_alpha,
            static_threshold=self.config.static_threshold,
        )
        self.state_machine = AlertStateMachine(
            persistence=self.config.persistence,
            cooldown_windows=self.config.cooldown_windows,
            static_threshold=self.config.static_threshold,
        )

        self._merchant_buffers: Dict[str, List[Transaction]] = {}
        self._merchant_window_starts: Dict[str, datetime] = {}
        self._emitted_alerts: List[Alert] = []

    def process_transactions(self, transactions: Sequence[Transaction]) -> List[Alert]:
        """Publish transaction stream onto bus and execute EventBus-driven streaming detection via bus.drain()."""
        self._merchant_buffers.clear()
        self._merchant_window_starts.clear()
        self._emitted_alerts.clear()

        self.bus.clear()
        self.bus.publish_batch(transactions)

        # EventBus-driven execution: bus.drain dispatches events sequentially via handler
        self.bus.drain(handler=self._on_transaction_dispatched)

        # Flush any remaining un-evaluated window buffers at stream completion
        for merchant_id in sorted(self._merchant_buffers.keys()):
            self._flush_merchant_window(merchant_id, force=True)

        return list(self._emitted_alerts)

    def _on_transaction_dispatched(self, tx: Transaction) -> None:
        """Handler invoked by TimeOrderedEventBus.drain() as VirtualClock advances."""
        merchant_id = tx.merchant_id

        if merchant_id not in self._merchant_window_starts:
            # Align window start to whole minute boundary of first transaction
            w_start_aligned = tx.timestamp.replace(second=0, microsecond=0)
            self._merchant_window_starts[merchant_id] = w_start_aligned
            self._merchant_buffers[merchant_id] = [tx]
            return

        w_start = self._merchant_window_starts[merchant_id]
        w_duration = timedelta(minutes=self.feature_engine.window_duration_minutes)
        w_end = w_start + w_duration

        if tx.timestamp < w_end:
            self._merchant_buffers[merchant_id].append(tx)
        else:
            # Current window completed -> evaluate window and slide start time
            while tx.timestamp >= w_end:
                self._flush_merchant_window(merchant_id, force=False)
                self._merchant_window_starts[merchant_id] = w_end
                w_start = w_end
                w_end = w_start + w_duration

            self._merchant_buffers[merchant_id].append(tx)

    def _flush_merchant_window(self, merchant_id: str, force: bool = False) -> None:
        """Extract features, evaluate baseline, score, process state machine, and persist audit records for a window."""
        w_start = self._merchant_window_starts.get(merchant_id)
        if w_start is None:
            return

        w_duration = timedelta(minutes=self.feature_engine.window_duration_minutes)
        w_end = w_start + w_duration
        txs = self._merchant_buffers.get(merchant_id, [])

        if not txs and not force:
            return

        # 1. Feature extraction over half-open window [w_start, w_end)
        feat_snap = self.feature_engine.extract_snapshot(merchant_id, txs, w_start, w_end)
        self._merchant_buffers[merchant_id] = []

        # 2. Historical-only baseline extraction (BEFORE updating baseline!)
        base_snap = self.baseline_engine.get_baseline(merchant_id, feat_snap)

        # 3. Scorer calculation SCOPED STRICTLY to scorer invocation (passing signal_mask for scorer-level ablation)
        scorer_exception: Optional[Exception] = None
        risk_score: Optional[RiskScore] = None

        try:
            if hasattr(self.scorer, "calculate_score"):
                import inspect
                sig = inspect.signature(self.scorer.calculate_score)
                if "signal_mask" in sig.parameters:
                    risk_score = self.scorer.calculate_score(feat_snap, base_snap, signal_mask=self.signal_mask)
                else:
                    risk_score = self.scorer.calculate_score(feat_snap, base_snap)
            else:
                risk_score = self.scorer(feat_snap, base_snap)
        except Exception as err:
            scorer_exception = err

        if scorer_exception is not None:
            # Section 20 Scorer Exception Path:
            # Error AuditRecord saved to SQLite (risk_score=None, data_quality_status="SCORER_ERROR");
            # NO Alert emitted; NO ALERT state transition; stream continues.
            spec_str = f"{merchant_id}:{w_end.isoformat()}:{type(scorer_exception).__name__}"
            err_audit_id = f"ERR-{hashlib.sha256(spec_str.encode('utf-8')).hexdigest()[:16]}"
            err_audit = AuditRecord(
                audit_id=err_audit_id,
                alert_id=None,
                merchant_id=merchant_id,
                timestamp=w_end,
                risk_score=None,
                confidence=0.0,
                features=feat_snap.model_dump(mode="json"),
                baseline=base_snap.model_dump(mode="json"),
                triggered_signals=[f"EXCEPT:{type(scorer_exception).__name__}:{str(scorer_exception)}"],
                detector_version=self.config.detector_version,
                data_quality_status="SCORER_ERROR",
            )
            self.audit_store.save_audit_record(err_audit)
            return

        # Scorer succeeded: update baseline, process state machine, persist alert & audit
        self.baseline_engine.update(feat_snap)

        prev_state = self.state_machine.get_merchant_state(merchant_id)
        new_state, alert = self.state_machine.process_score(merchant_id, w_end, risk_score)

        if new_state != prev_state:
            self.audit_store.save_state_transition(
                merchant_id=merchant_id,
                timestamp=w_end,
                previous_state=prev_state,
                new_state=new_state,
                reason=f"Score {risk_score.score} evaluated against threshold {self.config.static_threshold}",
                risk_score=risk_score.score,
            )

        if alert is not None:
            self._emitted_alerts.append(alert)
            self.audit_store.save_alert(alert)

        # Deterministic audit record ID
        score_repr = f"{risk_score.score:.4f}" if risk_score.score is not None else "NONE"
        spec_str = f"{merchant_id}:{w_end.isoformat()}:{score_repr}"
        audit_id = f"AUD-{hashlib.sha256(spec_str.encode('utf-8')).hexdigest()[:16]}"
        audit_record = AuditRecord(
            audit_id=audit_id,
            alert_id=alert.alert_id if alert else None,
            merchant_id=merchant_id,
            timestamp=w_end,
            risk_score=risk_score.score,
            confidence=risk_score.confidence,
            features=feat_snap.model_dump(mode="json"),
            baseline=base_snap.model_dump(mode="json"),
            triggered_signals=risk_score.triggered_signals,
            detector_version=self.config.detector_version,
            data_quality_status=risk_score.data_quality,
        )
        self.audit_store.save_audit_record(audit_record)
