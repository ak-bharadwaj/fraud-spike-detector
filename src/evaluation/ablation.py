"""AblationRunner module for executing isolated component ablation studies.

Key Invariants:
- Single-factor causation: Each ablation variant modifies EXACTLY ONE pipeline component or mechanism relative to FULL_PIPELINE control.
- Single-factor validation: Enforces that diff_count == 1 for every ablation variant; multi-factor variants raise ValueError.
- Control baseline configuration: FULL_PIPELINE control uses FrozenDetectorConfig (threshold=3.5, alpha=0.3, P=2, C=5).
- Characterization dataset: Ablation runs EXCLUSIVELY against development/characterization datasets (data/development/). Zero holdout contamination!
- Holdout rejection: Passing locked holdout dataset or data/holdout/ to ablation runner raises ValueError.
- Zero upstream mutation: Does NOT alter FeatureEngine, BaselineEngine, Scorer, or StateMachine implementations.
- Schema compliance: All emitted results validate strictly against AblationResult Pydantic contract.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    Alert,
    EvaluationMetrics,
    AblationVariantConfig,
    AblationResult,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.holdout import FrozenDetectorConfig, HoldoutManifest


def load_characterization_data(data_dir: Union[str, Path] = "data/development") -> Tuple[HoldoutManifest, List[Transaction], List[GroundTruthEvent]]:
    """Load characterization dataset from data/development/. Raises ValueError if holdout path is supplied."""
    d_path = Path(data_dir)
    if "holdout" in str(d_path).lower():
        raise ValueError("Holdout contamination error: Ablation framework cannot consume locked holdout data!")

    manifest_path = d_path / "manifest.json"
    tx_path = d_path / "transactions.json"
    gt_path = d_path / "ground_truth.json"

    if not manifest_path.exists() or not tx_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Characterization dataset missing in {d_path}")

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


class AblationRunner:
    """Runner executing controlled ablation studies against detector baseline."""

    def __init__(
        self,
        config: Optional[FrozenDetectorConfig] = None,
        temporal_tolerance_seconds: float = 0.0,
    ):
        self.config = config if config is not None else FrozenDetectorConfig()
        self.temporal_tolerance_seconds = temporal_tolerance_seconds

    def validate_single_factor_variant(self, variant: AblationVariantConfig) -> None:
        """Validate that an ablation variant modifies EXACTLY ONE causal mechanism relative to control config."""
        if variant.variant_id == "FULL_PIPELINE":
            return

        diff_count = 0
        if variant.disable_ewma is True:
            diff_count += 1
        if variant.persistence != self.config.persistence:
            diff_count += 1
        if variant.cooldown_windows != self.config.cooldown_windows:
            diff_count += 1
        if variant.feature_subset is not None:
            diff_count += 1
        if variant.static_threshold != self.config.static_threshold:
            diff_count += 1

        if diff_count > 1:
            raise ValueError(
                f"Invalid multi-factor ablation variant '{variant.variant_id}': modifies {diff_count} factors. "
                "Ablation variants must modify EXACTLY ONE causal factor."
            )
        if diff_count == 0:
            raise ValueError(
                f"Invalid ablation variant '{variant.variant_id}': No causal factor modified relative to control baseline."
            )

    def run_ablation_suite(
        self,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
        variants: Optional[List[AblationVariantConfig]] = None,
    ) -> List[AblationResult]:
        """Run full control pipeline and a suite of ablation variants, returning delta comparison results."""
        if variants is None:
            variants = self.get_standard_ablation_variants()

        # Enforce single-factor causation validation on all supplied variants
        for var in variants:
            self.validate_single_factor_variant(var)

        # 1. Run Control Baseline (FULL_PIPELINE)
        control_variant = AblationVariantConfig(
            variant_id="FULL_PIPELINE",
            description="Control full pipeline with EWMA, persistence P=2, cooldown C=5, and all 11 features",
            disable_ewma=False,
            persistence=self.config.persistence,
            cooldown_windows=self.config.cooldown_windows,
            feature_subset=None,
            static_threshold=self.config.static_threshold,
        )

        control_metrics = self.evaluate_variant(control_variant, transactions, ground_truth_events)

        results: List[AblationResult] = []

        # Control result comparison (delta = 0)
        results.append(
            AblationResult(
                variant_id=control_variant.variant_id,
                metrics=control_metrics,
                delta_f1=0.0,
                delta_precision=0.0,
                delta_recall=0.0,
                delta_latency_seconds=0.0,
            )
        )

        # 2. Run each isolated ablation variant
        for var in variants:
            if var.variant_id == "FULL_PIPELINE":
                continue

            var_metrics = self.evaluate_variant(var, transactions, ground_truth_events)

            d_f1 = var_metrics.f1_score - control_metrics.f1_score
            d_p = var_metrics.precision - control_metrics.precision
            d_r = var_metrics.recall - control_metrics.recall

            d_lat = None
            if var_metrics.mean_latency_seconds is not None and control_metrics.mean_latency_seconds is not None:
                d_lat = var_metrics.mean_latency_seconds - control_metrics.mean_latency_seconds

            results.append(
                AblationResult(
                    variant_id=var.variant_id,
                    metrics=var_metrics,
                    delta_f1=d_f1,
                    delta_precision=d_p,
                    delta_recall=d_r,
                    delta_latency_seconds=d_lat,
                )
            )

        return results

    def evaluate_variant(
        self,
        variant: AblationVariantConfig,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
    ) -> EvaluationMetrics:
        """Run detector evaluation pipeline for a single ablation variant."""
        feature_engine = FeatureEngine()
        baseline_engine = BaselineEngine(min_window_count=self.config.min_window_count)

        alpha = 1.0 if variant.disable_ewma else self.config.ewma_alpha
        scorer = HybridEWMAScorer(alpha=alpha)

        state_machine = AlertStateMachine(
            persistence=variant.persistence,
            cooldown_windows=variant.cooldown_windows,
            static_threshold=variant.static_threshold,
        )

        tx_by_merchant: Dict[str, List[Transaction]] = {}
        for tx in sorted(transactions, key=lambda x: x.timestamp):
            tx_by_merchant.setdefault(tx.merchant_id, []).append(tx)

        alerts: List[Alert] = []

        for merchant_id in sorted(tx_by_merchant.keys()):
            m_txs = tx_by_merchant[merchant_id]
            if not m_txs:
                continue

            start_time = m_txs[0].timestamp
            end_time = m_txs[-1].timestamp
            curr_window_start = start_time

            while curr_window_start <= end_time:
                curr_window_end = curr_window_start + timedelta(minutes=feature_engine.window_duration_minutes)
                curr_txs = [t for t in m_txs if curr_window_start <= t.timestamp < curr_window_end]

                feat_snap = feature_engine.extract_snapshot(merchant_id, curr_txs, curr_window_start, curr_window_end)

                # Feature subset ablation filtering if specified
                if variant.feature_subset is not None:
                    if "volume" in variant.feature_subset:
                        feat_snap = feat_snap.model_copy(
                            update={
                                "velocity": 0.0,
                                "unique_customers": 0,
                                "unique_devices": 0,
                                "amount_statistics": {k: 0.0 for k in feat_snap.amount_statistics},
                            }
                        )

                base_snap = baseline_engine.get_baseline(merchant_id, feat_snap)
                risk_score = scorer.calculate_score(feat_snap, base_snap)
                baseline_engine.update(feat_snap)

                _, alert = state_machine.process_score(merchant_id, curr_window_end, risk_score)
                if alert is not None:
                    alerts.append(alert)

                curr_window_start = curr_window_end

        evaluator = AnomalyEvaluator(temporal_tolerance_seconds=self.temporal_tolerance_seconds)
        return evaluator.evaluate(alerts, ground_truth_events)

    def get_standard_ablation_variants(self) -> List[AblationVariantConfig]:
        """Return standard suite of single-factor ablation study variants derived from frozen control config."""
        return [
            AblationVariantConfig(
                variant_id="NO_EWMA",
                description="EWMA smoothing disabled (alpha=1.0, raw z-scores scored directly)",
                disable_ewma=True,
                persistence=self.config.persistence,
                cooldown_windows=self.config.cooldown_windows,
                static_threshold=self.config.static_threshold,
            ),
            AblationVariantConfig(
                variant_id="NO_PERSISTENCE",
                description="Persistence disabled (P=1, single breaching window triggers alert)",
                disable_ewma=False,
                persistence=1,
                cooldown_windows=self.config.cooldown_windows,
                static_threshold=self.config.static_threshold,
            ),
            AblationVariantConfig(
                variant_id="NO_COOLDOWN",
                description="Cooldown suppression disabled (C=0, no alert suppression windows)",
                disable_ewma=False,
                persistence=self.config.persistence,
                cooldown_windows=0,
                static_threshold=self.config.static_threshold,
            ),
            AblationVariantConfig(
                variant_id="SINGLE_FEATURE_VOLUME_ONLY",
                description="Feature set reduced to volume feature only",
                disable_ewma=False,
                persistence=self.config.persistence,
                cooldown_windows=self.config.cooldown_windows,
                static_threshold=self.config.static_threshold,
                feature_subset=["volume"],
            ),
        ]
