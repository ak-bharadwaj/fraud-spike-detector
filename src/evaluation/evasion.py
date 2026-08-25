"""Adversarial Evasion Characterization Protocol (Day 12).

Measures detector performance degradation and evasion vulnerability when adversarial transaction
patterns undergo deliberate rate-dilution or attribute-spreading transformations.
Strictly isolated from Day 9 locked holdout.
"""

from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from pydantic import BaseModel

from src.contracts.contracts import (
    Transaction,
    GroundTruthEvent,
    EvaluationMetrics,
    EvasionConditionConfig,
    EvasionResult,
    FrozenDetectorConfig,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.evaluation.evaluator import AnomalyEvaluator


class EvasionManifest(BaseModel):
    """Manifest tracking dataset hashes and metadata for paired evasion characterization."""
    control_dataset_hash: str
    evasion_dataset_hash: str
    ground_truth_hash: str
    experiment_hash: str
    generator_version: str
    seed: int
    schema_version: str
    created_at: str


def compute_tx_hash(transactions: List[Transaction]) -> str:
    """Compute deterministic SHA-256 hash across sorted transaction records."""
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
    serialized = json.dumps(tx_list, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_gt_hash(gt_events: List[GroundTruthEvent]) -> str:
    """Compute deterministic SHA-256 hash across sorted GroundTruth events."""
    gt_list = [
        {
            "id": e.event_id,
            "m_id": e.merchant_id,
            "type": e.anomaly_type,
            "st": e.start_time.isoformat(),
            "et": e.end_time.isoformat(),
            "sev": float(e.severity),
        }
        for e in sorted(gt_events, key=lambda x: (x.start_time, x.event_id))
    ]
    serialized = json.dumps(gt_list, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_evasion_data(data_dir: Path | str) -> Tuple[EvasionManifest, List[Transaction], List[Transaction], List[GroundTruthEvent]]:
    """Load characterization evasion dataset with strict hash verification and holdout firewall check."""
    path = Path(data_dir).resolve()

    # Holdout Firewall Check: Ensure zero dependency or access to data/holdout/
    if "holdout" in path.parts:
        raise ValueError("Holdout contamination error: EvasionRunner cannot access data/holdout/")

    manifest_file = path / "manifest.json"
    control_file = path / "control_transactions.json"
    evasion_file = path / "evasion_transactions.json"
    gt_file = path / "ground_truth.json"

    if not (manifest_file.exists() and control_file.exists() and evasion_file.exists() and gt_file.exists()):
        raise FileNotFoundError(f"Missing required evasion artifacts in directory '{path}'")

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest = EvasionManifest(**manifest_data)

    def parse_txs(raw_list):
        txs = []
        for r in raw_list:
            ts = datetime.fromisoformat(r["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            txs.append(
                Transaction(
                    transaction_id=r["id"],
                    timestamp=ts,
                    merchant_id=r["m_id"],
                    customer_id=r["c_id"],
                    amount=r["amt"],
                    payment_method=r["pm"],
                    country=r["country"],
                    device_id=r["d_id"],
                )
            )
        return txs

    control_txs = parse_txs(json.loads(control_file.read_text(encoding="utf-8")))
    evasion_txs = parse_txs(json.loads(evasion_file.read_text(encoding="utf-8")))

    gt_raw = json.loads(gt_file.read_text(encoding="utf-8"))
    gt_events = []
    for g in gt_raw:
        st = datetime.fromisoformat(g["st"])
        et = datetime.fromisoformat(g["et"])
        if st.tzinfo is None:
            st = st.replace(tzinfo=timezone.utc)
        if et.tzinfo is None:
            et = et.replace(tzinfo=timezone.utc)
        gt_events.append(
            GroundTruthEvent(
                event_id=g["id"],
                merchant_id=g["m_id"],
                anomaly_type=g["type"],
                start_time=st,
                end_time=et,
                severity=g["sev"],
            )
        )

    # Verify SHA-256 Checksums
    actual_control_hash = compute_tx_hash(control_txs)
    actual_evasion_hash = compute_tx_hash(evasion_txs)
    actual_gt_hash = compute_gt_hash(gt_events)

    if actual_control_hash != manifest.control_dataset_hash:
        raise ValueError(
            f"Control dataset checksum mismatch: expected {manifest.control_dataset_hash}, got {actual_control_hash}"
        )
    if actual_evasion_hash != manifest.evasion_dataset_hash:
        raise ValueError(
            f"Evasion dataset checksum mismatch: expected {manifest.evasion_dataset_hash}, got {actual_evasion_hash}"
        )
    if actual_gt_hash != manifest.ground_truth_hash:
        raise ValueError(
            f"GroundTruth dataset checksum mismatch: expected {manifest.ground_truth_hash}, got {actual_gt_hash}"
        )

    actual_exp_payload = f"{actual_control_hash}:{actual_evasion_hash}:{actual_gt_hash}"
    actual_exp_hash = hashlib.sha256(actual_exp_payload.encode("utf-8")).hexdigest()
    if actual_exp_hash != manifest.experiment_hash:
        raise ValueError(
            f"Evasion experiment checksum mismatch: expected {manifest.experiment_hash}, got {actual_exp_hash}"
        )

    return manifest, control_txs, evasion_txs, gt_events


class EvasionRunner:
    """Executes adversarial evasion characterization experiments using the accepted frozen detector pipeline."""

    def __init__(self, config: Optional[FrozenDetectorConfig] = None):
        """Initialize EvasionRunner with explicit frozen detector configuration."""
        self.config = config or FrozenDetectorConfig()

    @staticmethod
    def get_standard_evasion_conditions() -> List[EvasionConditionConfig]:
        """Return standard suite of adversarial evasion characterization conditions."""
        return [
            EvasionConditionConfig(
                condition_id="SLOW_BURN_SPLIT_EVASION",
                description="Temporal dilution sub-threshold rate dilution evasion (spreading high-rate spike volume over 30 mins at 1.3x rate multiplier)",
                evasion_strategy="temporal_dilution",
                changed_factor="rate_multiplier_and_duration",
                magnitude=1.3,
                start_minute=50.0,
                duration_minutes=30.0,
            ),
        ]

    def run_evasion_suite(
        self,
        control_transactions: List[Transaction],
        evasion_transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
        conditions: Optional[List[EvasionConditionConfig]] = None,
    ) -> List[EvasionResult]:
        """Run standard evasion suite comparing control vs evasion detector execution."""
        cond_list = conditions or self.get_standard_evasion_conditions()
        results: List[EvasionResult] = []

        for cond in cond_list:
            # Validate declared transformation against protocol definition (BLOCKER 4)
            if cond.changed_factor != "rate_multiplier_and_duration" or cond.magnitude != 1.3:
                raise ValueError(
                    f"Invalid evasion condition: expected rate_multiplier_and_duration with magnitude 1.3, "
                    f"got {cond.changed_factor}={cond.magnitude}"
                )

            c_met, c_max_raw, c_max_ewma, c_ge_th, c_p2, c_alt = self._run_detector_pipeline(
                control_transactions, ground_truth_events
            )
            e_met, e_max_raw, e_max_ewma, e_ge_th, e_p2, e_alt = self._run_detector_pipeline(
                evasion_transactions, ground_truth_events
            )

            delta_f1 = e_met.f1_score - c_met.f1_score
            delta_prec = e_met.precision - c_met.precision
            delta_rec = e_met.recall - c_met.recall

            delta_lat = None
            if c_met.mean_latency_seconds is not None and e_met.mean_latency_seconds is not None:
                delta_lat = e_met.mean_latency_seconds - c_met.mean_latency_seconds

            degraded = c_met.f1_score > e_met.f1_score
            # Protocol definition (BLOCKER 3): 1.0 = successful evasion (GT exists & no alert emitted), 0.0 = failed evasion
            evasion_success = 1.0 if (c_met.tp > 0 and e_met.tp == 0) else 0.0

            results.append(
                EvasionResult(
                    condition_id=cond.condition_id,
                    control_metrics=c_met,
                    evasion_metrics=e_met,
                    delta_f1=delta_f1,
                    delta_precision=delta_prec,
                    delta_recall=delta_rec,
                    delta_latency_seconds=delta_lat,
                    detection_degraded=degraded,
                    evasion_success_rate=evasion_success,
                    control_max_raw_score=c_max_raw,
                    control_max_ewma_score=c_max_ewma,
                    control_score_ge_threshold=c_ge_th,
                    control_persistence_satisfied=c_p2,
                    control_alert_emitted=c_alt,
                    evasion_max_raw_score=e_max_raw,
                    evasion_max_ewma_score=e_max_ewma,
                    evasion_score_ge_threshold=e_ge_th,
                    evasion_persistence_satisfied=e_p2,
                    evasion_alert_emitted=e_alt,
                )
            )

        return results

    def _run_detector_pipeline(
        self,
        transactions: List[Transaction],
        ground_truth_events: List[GroundTruthEvent],
    ) -> Tuple[EvaluationMetrics, float, float, bool, bool, bool]:
        """Run frozen detector pipeline and capture raw/EWMA score trajectories and state machine evidence."""
        feature_engine = FeatureEngine()
        baseline_engine = BaselineEngine(min_window_count=self.config.min_window_count)
        scorer = HybridEWMAScorer(alpha=self.config.ewma_alpha)
        state_machine = AlertStateMachine(
            persistence=self.config.persistence,
            cooldown_windows=self.config.cooldown_windows,
            static_threshold=self.config.static_threshold,
        )

        if not transactions:
            evaluator = AnomalyEvaluator(temporal_tolerance_seconds=self.config.temporal_tolerance_seconds)
            return evaluator.evaluate([], ground_truth_events), 0.0, 0.0, False, False, False

        tx_by_merchant: Dict[str, List[Transaction]] = {}
        for t in transactions:
            tx_by_merchant.setdefault(t.merchant_id, []).append(t)

        start_time = min(t.timestamp for t in transactions)
        end_time = max(t.timestamp for t in transactions)

        all_emitted_alerts = []
        w_start = start_time
        window_delta = timedelta(minutes=1)

        max_raw_score = 0.0
        max_ewma_score = 0.0
        score_ge_threshold = False
        persistence_satisfied = False

        while w_start <= end_time:
            w_end = w_start + window_delta
            for m_id in sorted(tx_by_merchant.keys()):
                m_txs = tx_by_merchant[m_id]
                c_txs = [t for t in m_txs if w_start <= t.timestamp < w_end]
                snap = feature_engine.extract_snapshot(m_id, c_txs, w_start, w_end)
                base = baseline_engine.get_baseline(m_id, snap)
                risk = scorer.calculate_score(snap, base)
                baseline_engine.update(snap)

                if m_id == "EVASION_M1" and risk and risk.score is not None:
                    raw_z = snap.standardized_magnitudes.get("volume", 0.0) if hasattr(snap, "standardized_magnitudes") else 0.0
                    ewma_s = risk.score
                    if raw_z > max_raw_score:
                        max_raw_score = raw_z
                    if ewma_s > max_ewma_score:
                        max_ewma_score = ewma_s
                    if ewma_s >= self.config.static_threshold:
                        score_ge_threshold = True

                _, alert = state_machine.process_score(m_id, w_end, risk)
                if alert:
                    all_emitted_alerts.append(alert)
                    if m_id == "EVASION_M1":
                        persistence_satisfied = True

            w_start = w_end

        evaluator = AnomalyEvaluator(temporal_tolerance_seconds=self.config.temporal_tolerance_seconds)
        metrics = evaluator.evaluate(all_emitted_alerts, ground_truth_events)
        alert_emitted = len(all_emitted_alerts) > 0

        return metrics, max_raw_score, max_ewma_score, score_ge_threshold, persistence_satisfied, alert_emitted
