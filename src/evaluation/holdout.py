"""Locked holdout dataset protection guard and evaluation runner.

Key Invariants:
- Locked holdout evaluation runs ONLY when explicit_evaluation_mode=True.
- Verifies dataset checksum hash against HoldoutManifest; aborts with ChecksumMismatchError if hash differs.
- Runs single-pass evaluation of frozen detector pipeline against locked holdout data.
- Zero post-hoc tuning permitted: detector configuration parameters are frozen prior to holdout evaluation.
- Emits EvaluationMetrics compliant with Pydantic schema contract.
"""

from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from src.contracts.contracts import Transaction, GroundTruthEvent, Alert, EvaluationMetrics
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


class HoldoutProtection:
    """Protection guard for holdout dataset access."""

    @staticmethod
    def verify_access(
        manifest: HoldoutManifest,
        actual_dataset_hash: str,
        explicit_evaluation_mode: bool = False,
    ) -> bool:
        """Verify explicit evaluation mode and dataset hash checksum.

        Raises HoldoutAccessError if explicit_evaluation_mode is False.
        Raises ChecksumMismatchError if actual_dataset_hash != manifest.dataset_hash.
        """
        if not explicit_evaluation_mode:
            raise HoldoutAccessError(
                "Holdout access denied: Normal development mode cannot access holdout data. "
                "Explicit evaluation mode flag is required."
            )

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
        explicit_evaluation_mode: bool = False,
    ):
        self.manifest = manifest
        self.explicit_evaluation_mode = explicit_evaluation_mode

    def evaluate_holdout(
        self,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
        actual_dataset_hash: str,
        static_threshold: float = 3.5,
        ewma_alpha: float = 0.3,
        persistence: int = 2,
        cooldown_windows: int = 5,
        min_window_count: int = 5,
        temporal_tolerance_seconds: float = 0.0,
    ) -> EvaluationMetrics:
        """Execute single-pass evaluation of frozen detector against locked holdout dataset."""
        # 1. Verify access & checksum integrity
        HoldoutProtection.verify_access(
            manifest=self.manifest,
            actual_dataset_hash=actual_dataset_hash,
            explicit_evaluation_mode=self.explicit_evaluation_mode,
        )

        # 2. Instantiate frozen pipeline components
        feature_engine = FeatureEngine()
        baseline_engine = BaselineEngine(min_window_count=min_window_count)
        scorer = HybridEWMAScorer(alpha=ewma_alpha)
        state_machine = AlertStateMachine(
            persistence=persistence,
            cooldown_windows=cooldown_windows,
            static_threshold=static_threshold,
        )

        # 3. Group transactions by window & merchant to construct feature/baseline snapshots
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
            history_txs: List[Transaction] = []

            while curr_window_start <= end_time:
                curr_window_end = curr_window_start + timedelta(minutes=feature_engine.window_duration_minutes)
                curr_txs = [t for t in m_txs if curr_window_start <= t.timestamp < curr_window_end]
                history_txs.extend(curr_txs)

                # Extract feature snapshot for current window
                feat_snap = feature_engine.extract_snapshot(merchant_id, history_txs, curr_window_start, curr_window_end)

                # Compute baseline snapshot from history
                base_snap = baseline_engine.get_baseline(merchant_id, feat_snap)
                baseline_engine.update(feat_snap)

                # Score features against baseline
                risk_score = scorer.calculate_score(feat_snap, base_snap)

                # Evaluate risk score through alert state machine
                _, alert = state_machine.process_score(merchant_id, curr_window_end, risk_score)
                if alert is not None:
                    alerts.append(alert)

                curr_window_start = curr_window_end

        # 4. Run AnomalyEvaluator against ground truth events
        evaluator = AnomalyEvaluator(temporal_tolerance_seconds=temporal_tolerance_seconds)
        return evaluator.evaluate(alerts, ground_truth_events)
