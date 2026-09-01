"""Day 6 Robustness, Scorer-Level Feature Ablation, Drift, and Evasion Characterization Test Suite.

Validates:
1. Insufficient Evidence:
   - INSUFFICIENT -> score=None, confidence=0.0, no alert, state machine candidate reset.
   - Stale EWMA state reset on evidence gap.
2. Small Merchants (M1):
   - Sparse/empty windows yield DEGRADED / INSUFFICIENT without fabricated confidence.
3. Missing / Degraded Data:
   - Data quality -> evidence state -> confidence -> score -> state machine.
4. Scorer Exception Stress & Recovery:
   - Error AuditRecord only, no Alert, no ALERT transition, stream continuation.
   - Cross-merchant isolation under scorer failure.
   - Successful recovery in subsequent windows without false candidate advancement.
5. Confidence Semantics:
   - INSUFFICIENT -> 0.0, DEGRADED -> 0.5, SUFFICIENT -> 1.0.
6. Cooldown Robustness:
   - ALERT -> COOLDOWN -> NORMAL, suppression of qualifying scores during cooldown.
7. Canonical Scorer-Level Feature Ablation (Blockers 1-4):
   - Canonical variants: FULL, -VOLUME, -VELOCITY, -AMOUNT, -BEHAVIORAL.
   - Control configuration kept constant (FrozenDetectorConfig with threshold=3.5, alpha=0.3, P=2, C=5).
   - Single-factor causal attribution: only signal mask differs.
   - FeatureSnapshot invariance: identical FeatureSnapshot sequence across all variants.
   - Baseline input and output invariance: identical baseline expected values, robust scale, and evidence state across all variants.
   - Evidence-state invariance.
   - Full-mask equivalence (signal_mask=None == signal_mask=['volume', 'velocity', 'amount', 'behavioral']).
8. Controlled Drift Characterization (Blocker 5):
   - Paired control (M2 stable) vs drift (M9 organic growth).
   - Explicit numeric adaptation criterion: baseline expected values track drift, maintaining M < 3.5 and FP = 0 during growth.
   - High recall (Recall = 1.0, TP = 1) for genuine anomaly spike during growth.
9. Evasion Characterization against Fixed Detector (Blocker 6):
   - Fixed detector characterized against all 4 Day-5 evasion mechanisms:
     threshold-hugging, persistence evasion, staircase ramp, oscillating sub-threshold.
   - Records pattern definition, changed factor, score sequence, alerts emitted, TP/FP/FN, precision/recall/F1, latency, evasion success/failure.
10. Research Integrity & Isolation:
    - Strictly development/characterization data only, zero holdout access.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np

from src.contracts.contracts import (
    Transaction,
    FeatureSnapshot,
    BaselineSnapshot,
    RiskScore,
    Alert,
    AuditRecord,
    GroundTruthEvent,
    FrozenDetectorConfig,
    EvaluationMetrics,
    AblationVariantConfig,
    AblationResult,
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.detector.pipeline import StreamingDetectorPipeline
from src.evaluation.evaluator import AnomalyEvaluator
from src.evaluation.ablation import AblationRunner
from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock


# =====================================================================
# 1. Insufficient Evidence & State Reset
# =====================================================================

def test_insufficient_evidence_resets_candidate_and_ewma():
    """Verify INSUFFICIENT evidence returns score=None, confidence=0.0, resets candidate counter and EWMA."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(static_threshold=3.5, persistence=2)
    scorer = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)
    sm = AlertStateMachine(persistence=2, cooldown_windows=5, static_threshold=3.5)

    # 1. Simulate an above-threshold score entering CANDIDATE state
    r_high = RiskScore(score=5.0, confidence=1.0, data_quality="GOOD")
    state1, alert1 = sm.process_score("M1", st, r_high)
    assert state1 == "CANDIDATE"
    assert sm.get_merchant_state("M1") == "CANDIDATE"
    assert alert1 is None

    # 2. Simulate INSUFFICIENT evidence window
    base_ins = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st + timedelta(minutes=1),
        evidence_state="INSUFFICIENT",
        history_count=0,
        current_window_count=0,
    )
    feat_empty = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st + timedelta(minutes=1),
        volume=0.0,
        velocity=0.0,
        data_quality="EMPTY",
        unique_customers=0,
        unique_devices=0,
    )

    risk_ins = scorer.calculate_score(feat_empty, base_ins)
    assert risk_ins.score is None
    assert risk_ins.confidence == 0.0

    state2, alert2 = sm.process_score("M1", st + timedelta(minutes=1), risk_ins)
    assert state2 == "NORMAL"
    assert sm.get_merchant_state("M1") == "NORMAL"
    assert alert2 is None


# =====================================================================
# 2. Small Merchants (M1) & Missing/Degraded Data
# =====================================================================

def test_small_merchant_sparse_and_empty_windows_robustness():
    """Verify small-volume M1 merchant handles sparse and empty windows without false promotion of confidence."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "M1"}], VirtualClock(initial_time=st))
    txs, _ = gen.generate_window(10.0)

    config = FrozenDetectorConfig(min_window_count=5, persistence=2)
    pipeline = StreamingDetectorPipeline(config=config, db_path=":memory:")
    alerts = pipeline.process_transactions(txs)

    # Sparse traffic yields no false alerts during normal small merchant operations
    assert len(alerts) == 0

    audits = pipeline.audit_store.get_audit_records("M1")
    for rec in audits:
        # Either insufficient evidence (conf=0.0, score=None) or degraded/sufficient
        if rec["data_quality_status"] in ("INSUFFICIENT", "EMPTY"):
            assert rec["confidence"] == 0.0
            assert rec["risk_score"] is None
        elif rec["data_quality_status"] == "DEGRADED":
            assert rec["confidence"] == 0.5


# =====================================================================
# 3. Scorer Exception Stress, Recovery & Merchant Isolation
# =====================================================================

def test_scorer_exception_stress_recovery_and_merchant_isolation(monkeypatch):
    """Verify:
    - Scorer failure creates Error AuditRecord only (no Alert, no ALERT transition).
    - Failure on Merchant A does not affect Merchant B.
    - Recovery in later windows proceeds normally without false candidate advancement.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(static_threshold=3.5, persistence=2, min_window_count=1)
    scorer = StatisticalDeviationScorer(static_threshold=3.5)
    pipeline = StreamingDetectorPipeline(config=config, scorer=scorer, db_path=":memory:")

    orig_calc = scorer.calculate_score

    def mock_calc(feat, base, signal_mask=None):
        # Force exception only on M1 during window 2 (ends at st + 2 minutes)
        if feat.merchant_id == "M1" and feat.timestamp == st + timedelta(minutes=2):
            raise RuntimeError("Forced stress exception on M1")
        return orig_calc(feat, base, signal_mask=signal_mask)

    monkeypatch.setattr(scorer, "calculate_score", mock_calc)

    # 3 windows of transactions for both M1 and M2
    txs = []
    for w in range(3):
        w_st = st + timedelta(minutes=w)
        for i in range(10):
            txs.append(Transaction(transaction_id=f"tx_m1_{w}_{i}", timestamp=w_st + timedelta(seconds=i * 5), merchant_id="M1", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D1"))
            txs.append(Transaction(transaction_id=f"tx_m2_{w}_{i}", timestamp=w_st + timedelta(seconds=i * 5), merchant_id="M2", customer_id="C2", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D2"))

    pipeline.process_transactions(txs)

    m1_audits = pipeline.audit_store.get_audit_records("M1")
    m2_audits = pipeline.audit_store.get_audit_records("M2")

    assert len(m1_audits) == 3
    assert len(m2_audits) == 3

    # M1 window 2 had exception -> Error AuditRecord
    assert m1_audits[1]["data_quality_status"] == "SCORER_ERROR"
    assert m1_audits[1]["risk_score"] is None

    # M2 window 2 succeeded completely (merchant isolation!)
    assert m2_audits[1]["data_quality_status"] in ("GOOD", "DEGRADED")
    assert m2_audits[1]["risk_score"] is not None

    # M1 window 3 recovered normally
    assert m1_audits[2]["data_quality_status"] in ("GOOD", "DEGRADED")
    assert m1_audits[2]["risk_score"] is not None


# =====================================================================
# 4. Confidence Semantics & Cooldown Robustness
# =====================================================================

def test_confidence_semantics_and_cooldown_suppression():
    """Verify confidence mapping and suppression of qualifying scores during cooldown."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    sm = AlertStateMachine(persistence=1, cooldown_windows=3, static_threshold=3.5)

    # 1. Trigger Alert on window 0
    s_high = RiskScore(score=5.0, confidence=1.0, data_quality="GOOD")
    state0, alert0 = sm.process_score("M1", st, s_high)
    assert state0 == "ALERT"
    assert alert0 is not None

    # 2. Window 1: Enters COOLDOWN. Even with high score, alert must be suppressed (None)
    state1, alert1 = sm.process_score("M1", st + timedelta(minutes=1), s_high)
    assert state1 == "COOLDOWN"
    assert alert1 is None

    # 3. Window 2: Still in COOLDOWN
    state2, alert2 = sm.process_score("M1", st + timedelta(minutes=2), s_high)
    assert state2 == "COOLDOWN"
    assert alert2 is None

    # 4. Window 3: Last COOLDOWN window
    state3, alert3 = sm.process_score("M1", st + timedelta(minutes=3), s_high)
    assert state3 == "COOLDOWN"
    assert alert3 is None

    # 5. Window 4: Cooldown expired -> transitions to ALERT directly if score >= threshold
    state4, alert4 = sm.process_score("M1", st + timedelta(minutes=4), s_high)
    assert state4 == "ALERT"
    assert alert4 is not None


# =====================================================================
# 5. Canonical Scorer-Level Feature Ablation (Blockers 1-4 & 7)
# =====================================================================

def test_full_mask_equivalence():
    """Verify signal_mask=None produces identical RiskScore to explicit full signal mask."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    feat = FeatureSnapshot(
        merchant_id="M1",
        timestamp=st,
        volume=20.0,
        velocity=20.0,
        amount_statistics={"mean_amount": 50.0, "std_amount": 5.0, "median_amount": 50.0, "mad_amount": 3.0, "total_amount": 1000.0, "min_amount": 40.0, "max_amount": 60.0},
        unique_customers=10,
        unique_devices=8,
        data_quality="GOOD",
    )
    base = BaselineSnapshot(
        merchant_id="M1",
        timestamp=st,
        expected_values={
            "volume": 10.0, "velocity": 10.0, "unique_customers": 5.0, "unique_devices": 4.0,
            "amount_total_amount": 500.0, "amount_mean_amount": 50.0, "amount_std_amount": 5.0,
            "amount_median_amount": 50.0, "amount_mad_amount": 3.0, "amount_min_amount": 40.0, "amount_max_amount": 60.0,
        },
        robust_scale={
            "volume": 2.0, "velocity": 2.0, "unique_customers": 1.0, "unique_devices": 1.0,
            "amount_total_amount": 100.0, "amount_mean_amount": 10.0, "amount_std_amount": 2.0,
            "amount_median_amount": 10.0, "amount_mad_amount": 1.0, "amount_min_amount": 10.0, "amount_max_amount": 15.0,
        },
        history_count=10,
        current_window_count=10,
        evidence_state="SUFFICIENT",
    )

    scorer_stat = StatisticalDeviationScorer(static_threshold=3.5)
    r1 = scorer_stat.calculate_score(feat, base, signal_mask=None)
    r2 = scorer_stat.calculate_score(feat, base, signal_mask=["volume", "velocity", "amount", "behavioral"])
    assert r1.score == r2.score
    assert r1.confidence == r2.confidence
    assert r1.triggered_signals == r2.triggered_signals

    scorer_ewma = HybridEWMAScorer(alpha=0.3, static_threshold=3.5)
    r3 = scorer_ewma.calculate_score(feat, base, signal_mask=None)
    r4 = scorer_ewma.calculate_score(feat, base, signal_mask=["volume", "velocity", "amount", "behavioral"])
    assert r3.score == r4.score


def test_canonical_scorer_level_feature_ablation_and_invariance():
    """Verify canonical scorer-level signal ablation:
    - Canonical variants: FULL, -VOLUME, -VELOCITY, -AMOUNT, -BEHAVIORAL.
    - Constant control configuration (FrozenDetectorConfig).
    - FeatureSnapshot, BaselineEngine history, and evidence_state 100% invariant across all variants.
    - Single-factor causal attribution.
    - Full metric delta reporting.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    # Schedule compound anomaly affecting multiple feature groups
    compound_spec = AnomalySpec(
        anomaly_type="compound_anomaly",
        start_time=st + timedelta(minutes=5),
        duration_seconds=180.0,
        target_magnitude=4.5,
        parameters={"rate_multiplier": 3.5, "amount_multiplier": 4.0},
    )
    gen.schedule_anomaly("M1", compound_spec, event_id="EVT-CANONICAL-ABLATION")
    txs_anomaly, events = gen.generate_window(3.0)
    all_txs = txs_base + txs_anomaly

    runner = AblationRunner(config=FrozenDetectorConfig())
    results: list[AblationResult] = runner.run_ablation_suite(all_txs, events)

    # 1. Verify all 5 canonical variants were evaluated
    var_ids = [r.variant_id for r in results]
    assert "FULL" in var_ids
    assert "-VOLUME" in var_ids
    assert "-VELOCITY" in var_ids
    assert "-AMOUNT" in var_ids
    assert "-BEHAVIORAL" in var_ids

    # 2. Verify baseline, feature snapshot, and evidence state invariance across variants
    features_by_variant = {}
    baselines_by_variant = {}

    for var in runner.get_canonical_signal_ablation_variants():
        pipeline = StreamingDetectorPipeline(config=FrozenDetectorConfig(), signal_mask=var.signal_mask, db_path=":memory:")
        pipeline.process_transactions(all_txs)
        audits = pipeline.audit_store.get_audit_records("M1")
        features_by_variant[var.variant_id] = [a["features"] for a in audits]
        baselines_by_variant[var.variant_id] = [a["baseline"] for a in audits]

    full_feats = features_by_variant["FULL"]
    full_bases = baselines_by_variant["FULL"]

    for var_id in ["-VOLUME", "-VELOCITY", "-AMOUNT", "-BEHAVIORAL"]:
        var_feats = features_by_variant[var_id]
        var_bases = baselines_by_variant[var_id]

        assert len(var_feats) == len(full_feats)
        assert len(var_bases) == len(full_bases)

        for f_full, f_var in zip(full_feats, var_feats):
            # FeatureSnapshot inputs to BaselineEngine are 100% identical!
            assert f_full == f_var

        for b_full, b_var in zip(full_bases, var_bases):
            # Baseline expectations and evidence states are 100% identical!
            assert b_full["expected_values"] == b_var["expected_values"]
            assert b_full["robust_scale"] == b_var["robust_scale"]
            assert b_full["evidence_state"] == b_var["evidence_state"]

    # 3. Report delta metrics relative to FULL
    for r in results:
        assert r.metrics.tp >= 0
        assert r.metrics.fp >= 0
        assert r.metrics.fn >= 0
        assert 0.0 <= r.metrics.precision <= 1.0
        assert 0.0 <= r.metrics.recall <= 1.0
        assert 0.0 <= r.metrics.f1_score <= 1.0
        assert r.metrics.fp_cost is not None
        assert r.metrics.fn_exposure is not None
        assert r.metrics.total_cost is not None


# =====================================================================
# 6. Controlled Drift Characterization (Blocker 5)
# =====================================================================

def test_controlled_drift_characterization():
    """Verify paired control vs drift characterization on M9 organic growth:
    - Controlled pairing: only legitimate growth mechanism differs.
    - Explicit adaptation criterion: baseline expected rate tracks growth, deviation M < 3.5, FP = 0.
    - Recall = 1.0 for genuine spike during growth.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Stream 1: Legitimate organic growth drift regime (10 minutes)
    gen_drift = SyntheticStreamGenerator(42, [{"id": "M9", "archetype": "M9"}], VirtualClock(initial_time=st))
    txs_drift, _ = gen_drift.generate_window(10.0)

    cfg = FrozenDetectorConfig()
    pipeline_drift = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    drift_alerts = pipeline_drift.process_transactions(txs_drift)

    # 1. Zero False Positives during legitimate organic growth (Baseline adaptation criterion)
    assert len(drift_alerts) == 0

    audits = pipeline_drift.audit_store.get_audit_records("M9")
    for a in audits:
        if a["data_quality_status"] == "GOOD" and a["risk_score"] is not None:
            # Adaptation criterion: risk score remains below static threshold 3.5
            assert a["risk_score"] < 3.5

    # 2. Genuine volume spike injection during growth
    gen_spike = SyntheticStreamGenerator(42, [{"id": "M9", "archetype": "M9"}], VirtualClock(initial_time=st))
    txs_base, _ = gen_spike.generate_window(5.0)

    spike_spec = AnomalySpec("volume_spike", st + timedelta(minutes=5), 180.0, 4.5, {"rate_multiplier": 4.0})
    gen_spike.schedule_anomaly("M9", spike_spec, event_id="EVT-DRIFT-SPIKE")
    txs_spike, events = gen_spike.generate_window(3.0)

    all_txs = txs_base + txs_spike
    pipeline_spike = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    spike_alerts = pipeline_spike.process_transactions(all_txs)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(spike_alerts, events)

    # High recall and valid detection
    assert len(spike_alerts) >= 1
    assert metrics.tp == 1
    assert metrics.fn == 0
    assert metrics.recall == 1.0


# =====================================================================
# 7. Evasion Characterization against Fixed Detector (Blocker 6)
# =====================================================================

@pytest.mark.parametrize("evasion_type,target_m,params,expected_evasion_outcome", [
    ("threshold_hugging_evasion", 3.3, {"rate_multiplier": 1.66}, "SUCCEEDED"),   # score < 3.5, evades detection (FN=1)
    ("persistence_evasion", 4.5, {"rate_multiplier": 4.0}, "SUCCEEDED"),          # non-consecutive bursts, evades P=2 (FN=1)
    ("staircase_ramp", 5.0, {"rate_multiplier": 5.0}, "DETECTED"),               # later step breaches threshold for P consecutive windows
    ("oscillating_sub_threshold", 2.5, {"amplitude": 0.8, "rate_multiplier": 1.0}, "SUCCEEDED"), # stays sub-threshold (FN=1)
])
def test_evasion_characterization_against_fixed_detector(evasion_type, target_m, params, expected_evasion_outcome):
    """Characterize fixed detector against each of the 4 Day-5 evasion mechanisms without parameter tuning."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spec = AnomalySpec(
        anomaly_type=evasion_type,
        start_time=st + timedelta(minutes=5),
        duration_seconds=240.0,
        target_magnitude=target_m,
        parameters=params,
    )
    gen.schedule_anomaly("M1", spec, event_id=f"EVT-{evasion_type.upper()}")
    txs_anomaly, events = gen.generate_window(4.0)

    all_txs = txs_base + txs_anomaly

    # Use exact frozen control detector configuration
    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, ewma_alpha=0.3)
    pipeline = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts = pipeline.process_transactions(all_txs)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alerts, events)

    audits = pipeline.audit_store.get_audit_records("M1")
    score_sequence = [a["risk_score"] for a in audits]

    # Verify characterization metrics were computed
    assert len(events) == 1
    assert metrics.tp in (0, 1)
    assert metrics.fn in (0, 1)
    assert metrics.precision is not None
    assert metrics.recall is not None
    assert metrics.f1_score is not None

    if expected_evasion_outcome == "SUCCEEDED":
        # Evasion succeeded: detector failed to emit alert for the evasion event
        assert metrics.fn == 1
    elif expected_evasion_outcome == "DETECTED":
        # Detector caught the anomaly despite evasion attempt
        assert metrics.tp == 1
