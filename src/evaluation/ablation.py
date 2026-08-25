"""AblationRunner module for executing isolated component ablation studies.

Key Invariants:
- Evaluates isolated ablation variants against control full pipeline.
- Single-factor causation: Each ablation variant modifies EXACTLY ONE pipeline component or mechanism.
- Identical evaluation streams: All variants process identical transaction streams and GroundTruthEvents.
- Zero holdout tuning: Ablation runs on development/validation dataset or benchmark evaluation dataset.
- Zero upstream mutation: Does NOT alter FeatureEngine, BaselineEngine, Scorer, or StateMachine implementations.
- Schema compliance: All emitted results validate strictly against AblationResult Pydantic contract.
"""

from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta

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


class AblationRunner:
    """Runner executing controlled ablation studies against detector baseline."""

    def __init__(self, temporal_tolerance_seconds: float = 0.0):
        self.temporal_tolerance_seconds = temporal_tolerance_seconds

    def run_ablation_suite(
        self,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
        variants: Optional[List[AblationVariantConfig]] = None,
    ) -> List[AblationResult]:
        """Run full control pipeline and a suite of ablation variants, returning delta comparison results."""
        if variants is None:
            variants = self.get_standard_ablation_variants()

        # 1. Run Control Baseline (FULL_PIPELINE)
        control_variant = AblationVariantConfig(
            variant_id="FULL_PIPELINE",
            description="Control full pipeline with EWMA, persistence P=2, cooldown C=5, and all 11 features",
            disable_ewma=False,
            persistence=2,
            cooldown_windows=5,
            feature_subset=None,
            static_threshold=3.5,
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
        baseline_engine = BaselineEngine(min_window_count=5)

        alpha = 1.0 if variant.disable_ewma else 0.3
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
                    # Retain only volume for VOLUME_ONLY variant, zero out others
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

    @staticmethod
    def get_standard_ablation_variants() -> List[AblationVariantConfig]:
        """Return standard suite of single-factor ablation study variants."""
        return [
            AblationVariantConfig(
                variant_id="NO_EWMA",
                description="EWMA smoothing disabled (alpha=1.0, raw z-scores scored directly)",
                disable_ewma=True,
                persistence=2,
                cooldown_windows=5,
            ),
            AblationVariantConfig(
                variant_id="NO_PERSISTENCE",
                description="Persistence disabled (P=1, single breaching window triggers alert)",
                disable_ewma=False,
                persistence=1,
                cooldown_windows=5,
            ),
            AblationVariantConfig(
                variant_id="NO_COOLDOWN",
                description="Cooldown suppression disabled (C=0, no alert suppression windows)",
                disable_ewma=False,
                persistence=2,
                cooldown_windows=0,
            ),
            AblationVariantConfig(
                variant_id="SINGLE_FEATURE_VOLUME_ONLY",
                description="Feature set reduced to volume feature only",
                disable_ewma=False,
                persistence=2,
                cooldown_windows=5,
                feature_subset=["volume"],
            ),
        ]
