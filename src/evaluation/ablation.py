"""AblationRunner module for executing isolated component and signal ablation studies.

Key Invariants:
- Canonical signal ablation variants (scorer-level signal masking):
    FULL        : All 4 feature groups enabled (volume, velocity, amount, behavioral).
    -VOLUME     : Excludes volume signal.
    -VELOCITY   : Excludes velocity signal.
    -AMOUNT     : Excludes amount statistics signal.
    -BEHAVIORAL : Excludes behavioral/device cardinality signal.
- Single-factor causation: Each ablation variant modifies EXACTLY ONE causal mechanism relative to FULL control.
- Control baseline configuration: FULL uses FrozenDetectorConfig (threshold=3.5, alpha=0.3, P=2, C=5).
- Scorer-level signal masking: Ablation occurs strictly inside Scorer.calculate_score(..., signal_mask=...).
  Does NOT zero FeatureSnapshot fields, does NOT alter FeatureEngine, does NOT starve BaselineEngine, does NOT alter evidence_state.
- Characterization dataset: Ablation runs EXCLUSIVELY against development/characterization datasets (data/development/). Zero holdout contamination!
- Holdout rejection: Passing locked holdout dataset or data/holdout/ to ablation runner raises ValueError.
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
    FrozenDetectorConfig,
)
from src.detector.pipeline import StreamingDetectorPipeline
from src.scoring.static import StaticThresholdScorer
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.holdout import HoldoutManifest


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
            parameters=e.get("params", {
                "excess_transaction_count": max(1.0, float(round(10.0 * e["sev"]))),
                "mean_transaction_amount": 50.0,
                "exposure_factor": 1.0,
            }),
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
        self.evaluator = AnomalyEvaluator(temporal_tolerance_seconds=temporal_tolerance_seconds)

    @classmethod
    def get_canonical_signal_ablation_variants(cls) -> List[AblationVariantConfig]:
        """Return the canonical five scorer-level signal ablation variants."""
        return [
            AblationVariantConfig(
                variant_id="FULL",
                description="Control full pipeline with all 4 feature groups enabled",
                signal_mask=None,
            ),
            AblationVariantConfig(
                variant_id="-VOLUME",
                description="Ablate volume signal (velocity, amount, behavioral active)",
                signal_mask=["velocity", "amount", "behavioral"],
            ),
            AblationVariantConfig(
                variant_id="-VELOCITY",
                description="Ablate velocity signal (volume, amount, behavioral active)",
                signal_mask=["volume", "amount", "behavioral"],
            ),
            AblationVariantConfig(
                variant_id="-AMOUNT",
                description="Ablate amount statistics signal (volume, velocity, behavioral active)",
                signal_mask=["volume", "velocity", "behavioral"],
            ),
            AblationVariantConfig(
                variant_id="-BEHAVIORAL",
                description="Ablate behavioral/device signal (volume, velocity, amount active)",
                signal_mask=["volume", "velocity", "amount"],
            ),
        ]

    def get_standard_ablation_variants(self) -> List[AblationVariantConfig]:
        """Return the standard ablation variants."""
        return self.get_canonical_signal_ablation_variants()

    def validate_single_factor_variant(self, variant: AblationVariantConfig) -> None:
        """Validate that an ablation variant modifies EXACTLY ONE causal mechanism relative to control config."""
        if variant.variant_id in ("FULL", "FULL_PIPELINE"):
            return

        diff_count = 0
        if variant.signal_mask is not None:
            diff_count += 1
        if variant.disable_ewma is True:
            diff_count += 1
        if variant.persistence is not None and variant.persistence != self.config.persistence:
            diff_count += 1
        if variant.cooldown_windows is not None and variant.cooldown_windows != self.config.cooldown_windows:
            diff_count += 1
        if variant.feature_subset is not None:
            diff_count += 1
        if variant.static_threshold is not None and variant.static_threshold != self.config.static_threshold:
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
            variants = self.get_canonical_signal_ablation_variants()

        # Enforce single-factor causation validation on all supplied variants
        for var in variants:
            self.validate_single_factor_variant(var)

        # 1. Run Control Baseline (FULL)
        control_variant = next(
            (v for v in variants if v.variant_id in ("FULL", "FULL_PIPELINE")),
            AblationVariantConfig(
                variant_id="FULL",
                description="Control full pipeline with all 4 feature groups enabled",
                signal_mask=None,
            ),
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
            if var.variant_id in ("FULL", "FULL_PIPELINE"):
                continue

            var_metrics = self.evaluate_variant(var, transactions, ground_truth_events)

            d_f1 = var_metrics.f1_score - control_metrics.f1_score
            d_p = var_metrics.precision - control_metrics.precision
            d_r = var_metrics.recall - control_metrics.recall

            d_lat = None
            if var_metrics.median_latency_seconds is not None and control_metrics.median_latency_seconds is not None:
                d_lat = var_metrics.median_latency_seconds - control_metrics.median_latency_seconds
            elif var_metrics.mean_latency_seconds is not None and control_metrics.mean_latency_seconds is not None:
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
        """Run detector evaluation pipeline for a single ablation variant using exact scorer-level signal masking."""
        eff_threshold = variant.static_threshold if variant.static_threshold is not None else self.config.static_threshold
        eff_persistence = variant.persistence if variant.persistence is not None else self.config.persistence
        eff_cooldown = variant.cooldown_windows if variant.cooldown_windows is not None else self.config.cooldown_windows

        if self.config.scorer == "StatisticalDeviationScorer":
            scorer = StatisticalDeviationScorer(
                static_threshold=eff_threshold,
                signal_weights=self.config.signal_weights,
            )
        elif self.config.scorer == "StaticThresholdScorer":
            scorer = StaticThresholdScorer(
                static_threshold=eff_threshold,
                signal_weights=self.config.signal_weights,
            )
        else:
            alpha = 1.0 if variant.disable_ewma else (self.config.ewma_alpha or 0.3)
            scorer = HybridEWMAScorer(
                alpha=alpha,
                static_threshold=eff_threshold,
                signal_weights=self.config.signal_weights,
            )

        cfg = self.config.model_copy(update={
            "persistence": eff_persistence,
            "cooldown_windows": eff_cooldown,
            "static_threshold": eff_threshold,
        })

        pipeline = StreamingDetectorPipeline(
            config=cfg,
            scorer=scorer,
            signal_mask=variant.signal_mask,
            db_path=":memory:",
        )

        alerts = pipeline.process_transactions(transactions)
        return self.evaluator.evaluate(alerts, ground_truth_events)
