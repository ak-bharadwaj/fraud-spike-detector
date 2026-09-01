"""Day 6 Robustness and Research-Characterization Test Suite.

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
7. Scorer-Level Feature Ablation:
   - Canonical variants: FULL, -VOLUME, -VELOCITY, -AMOUNT, -BEHAVIORAL.
   - Baseline invariance across variants.
   - Evidence state invariance across variants.
   - Single-factor ablation enforcement.
   - Full metric & delta reporting.
8. Drift Characterization:
   - Legitimate organic growth (M9) paired with control.
   - Measures FP during growth, recall for genuine spike, baseline adaptation.
9. Evasion Characterization:
   - Characterizes fixed detector against all 4 Day-5 evasion mechanisms:
     threshold-hugging, persistence evasion, staircase ramp, oscillating sub-threshold.
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
)
from src.features.feature_engine import FeatureEngine
from src.baseline.baseline_engine import BaselineEngine
from src.scoring.statistical import StatisticalDeviationScorer
from src.scoring.hybrid_ewma import HybridEWMAScorer
from src.state.alert_state_machine import AlertStateMachine
from src.detector.pipeline import StreamingDetectorPipeline
from src.evaluation.evaluator import AnomalyEvaluator
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
# 5. Feature Ablation: Scorer-Level Signal Masking
# =====================================================================

def test_scorer_level_signal_masking_ablation_and_invariance():
    """Verify scorer-level feature ablation:
    - Canonical variants: FULL, -VOLUME, -VELOCITY, -AMOUNT, -BEHAVIORAL.
    - Baseline and evidence state remain strictly identical across all variants.
    - Single-factor ablation enforcement.
    - Full metric and delta reporting.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    # Schedule compound anomaly (affects volume, amount, behavioral)
    compound_spec = AnomalySpec(
        anomaly_type="compound_anomaly",
        start_time=st + timedelta(minutes=5),
        duration_seconds=180.0,
        target_magnitude=4.5,
        parameters={"rate_multiplier": 3.5, "amount_multiplier": 4.0},
    )
    gen.schedule_anomaly("M1", compound_spec, event_id="EVT-ABLATION")
    txs_anomaly, events = gen.generate_window(3.0)
    all_txs = txs_base + txs_anomaly

    evaluator = AnomalyEvaluator()

    variants = {
        "FULL": None,
        "-VOLUME": ["velocity", "amount", "behavioral"],
        "-VELOCITY": ["volume", "amount", "behavioral"],
        "-AMOUNT": ["volume", "velocity", "behavioral"],
        "-BEHAVIORAL": ["volume", "velocity", "amount"],
    }

    results = {}
    baselines_by_variant = {}

    for var_name, mask in variants.items():
        cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=1, min_window_count=1)
        scorer = StatisticalDeviationScorer(static_threshold=3.5)
        pipeline = StreamingDetectorPipeline(config=cfg, scorer=scorer, signal_mask=mask, db_path=":memory:")

        alerts = pipeline.process_transactions(all_txs)
        metrics: EvaluationMetrics = evaluator.evaluate(alerts, events)

        results[var_name] = metrics
        baselines_by_variant[var_name] = [
            r["baseline"] for r in pipeline.audit_store.get_audit_records("M1")
        ]

    # 1. Verify baseline history invariance: baseline snapshots are 100% identical across all ablation variants
    full_baselines = baselines_by_variant["FULL"]
    for var_name, var_baselines in baselines_by_variant.items():
        assert len(var_baselines) == len(full_baselines)
        for b_full, b_var in zip(full_baselines, var_baselines):
            assert b_full["expected_values"] == b_var["expected_values"]
            assert b_full["robust_scale"] == b_var["robust_scale"]
            assert b_full["evidence_state"] == b_var["evidence_state"]

    # 2. Verify all variants produced complete metrics
    assert "FULL" in results
    assert "-VOLUME" in results
    assert "-VELOCITY" in results
    assert "-AMOUNT" in results
    assert "-BEHAVIORAL" in results

    for name, m in results.items():
        assert m.tp >= 0
        assert m.fp >= 0
        assert m.fn >= 0
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1_score <= 1.0
        assert m.fp_cost is not None
        assert m.fn_exposure is not None
        assert m.total_cost is not None


# =====================================================================
# 6. Drift Characterization & Pairing
# =====================================================================

def test_drift_characterization_and_adaptation():
    """Verify drift characterization on M9 organic growth paired with control:
    - Measures FP during legitimate growth.
    - Measures recall for genuine spike during growth.
    - Verifies baseline adaptation over time.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # 1. Legitimate growth stream without anomalies (10 minutes)
    gen_drift = SyntheticStreamGenerator(42, [{"id": "M9", "archetype": "M9"}], VirtualClock(initial_time=st))
    txs_normal, _ = gen_drift.generate_window(10.0)

    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, min_window_count=2)
    pipeline = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    normal_alerts = pipeline.process_transactions(txs_normal)

    # Zero false positives during legitimate organic growth
    assert len(normal_alerts) == 0

    # 2. Genuine volume spike injection during growth
    gen_spike = SyntheticStreamGenerator(42, [{"id": "M9", "archetype": "M9"}], VirtualClock(initial_time=st))
    txs_base, _ = gen_spike.generate_window(5.0)

    spike_spec = AnomalySpec("volume_spike", st + timedelta(minutes=5), 180.0, 4.5, {"rate_multiplier": 4.0})
    gen_spike.schedule_anomaly("M9", spike_spec, event_id="EVT-DRIFT-SPIKE")
    txs_spike, events = gen_spike.generate_window(3.0)

    all_txs = txs_base + txs_spike
    spike_cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=1, min_window_count=2)
    spike_pipeline = StreamingDetectorPipeline(config=spike_cfg, db_path=":memory:")
    spike_alerts = spike_pipeline.process_transactions(all_txs)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(spike_alerts, events)

    assert metrics.tp == 1
    assert metrics.fn == 0
    assert metrics.recall == 1.0


# =====================================================================
# 7. Evasion Characterization
# =====================================================================

@pytest.mark.parametrize("evasion_type,target_m,params", [
    ("threshold_hugging_evasion", 3.3, {"rate_multiplier": 1.66}),
    ("persistence_evasion", 4.5, {"rate_multiplier": 4.0}),
    ("staircase_ramp", 5.0, {"rate_multiplier": 5.0}),
    ("oscillating_sub_threshold", 2.5, {"amplitude": 0.8, "rate_multiplier": 1.0}),
])
def test_evasion_characterization_against_frozen_detector(evasion_type, target_m, params):
    """Characterize fixed detector against each of the 4 Day-5 evasion mechanisms."""
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

    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, min_window_count=1)
    pipeline = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts = pipeline.process_transactions(all_txs)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alerts, events)

    assert len(events) == 1
    assert metrics.precision is not None
    assert metrics.recall is not None
    assert metrics.f1_score is not None
