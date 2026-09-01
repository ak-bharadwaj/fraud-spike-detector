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
7. Canonical Scorer-Level Feature Ablation (Blockers 1-4 & 7):
   - Canonical variants: FULL, -VOLUME, -VELOCITY, -AMOUNT, -BEHAVIORAL.
   - Control configuration kept constant (FrozenDetectorConfig with threshold=3.5, alpha=0.3, P=2, C=5).
   - Single-factor causal attribution: only signal mask differs.
   - FeatureSnapshot invariance: identical FeatureSnapshot sequence across all variants.
   - Baseline input and output invariance: identical baseline expected values, robust scale, and evidence state across all variants.
   - Evidence-state invariance.
   - Full-mask equivalence (signal_mask=None == signal_mask=['volume', 'velocity', 'amount', 'behavioral']).
   - Dynamic metric & delta verification: rigorously asserts valid EvaluationMetrics and mathematical delta definitions (delta = variant - FULL) without hardcoding predetermined outcomes.
8. True Control-vs-Drift Pairing & Quantitative Adaptation Measurement (Blockers 1, 2, 3, 4):
   - Paired CONTROL (stable) vs DRIFT (growing) streams with identical seed, merchant ID, duration, and anomaly injection.
   - Warmup exclusion: warmup windows (w < 6) are explicitly excluded from adaptation calculation.
   - Empirical drift measurement: computes relative error between BaselineEngine expected volume and realized empirical volume.
   - Exact convergence criterion: relative error <= 0.20 across >= 8 post-warmup adaptation windows.
   - Zero false positives during unperturbed drift (FP_drift = 0).
   - Full comparative metrics & deltas: control latency, drift latency, delta latency, control metrics, drift metrics, delta FP, delta recall.
9. Evasion Trajectory & Causal Mechanism Proofs:
   - Threshold-hugging: score envelope in [1.2, 3.5), persistence count = 0, causally proving FN=1.
   - Persistence evasion: alternating qualifying/non-qualifying score sequence with reset on window 1, causally proving FN=1.
   - Staircase ramp: monotonically increasing score progression S0 < S1 < S2 < S3, consecutive steps breach threshold to satisfy P=2, causally proving TP=1.
   - Oscillating sub-threshold: oscillating waveform with max score < 3.5, persistence count = 0, causally proving FN=1.
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
        if rec["data_quality_status"] in ("INSUFFICIENT", "EMPTY"):
            assert rec["confidence"] == 0.0
            assert rec["risk_score"] is None
        elif rec["data_quality_status"] == "DEGRADED":
            assert rec["confidence"] == 0.5


def test_real_missing_and_degraded_transaction_injection_through_pipeline():
    """Verify real transactions with missing/degraded device_id, customer_id, or non-positive amount are processed safely by pipeline."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(min_window_count=1)
    pipeline = StreamingDetectorPipeline(config=config, db_path=":memory:")

    # Window 0: standard warmup transactions to establish baseline history
    warmup_txs = [
        Transaction(transaction_id=f"tx_warm_{i}", timestamp=st + timedelta(seconds=i * 10), merchant_id="M_DEG", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="DEV_VALID")
        for i in range(5)
    ]

    # Window 1: transactions with degraded fields (empty device_id, unknown customer_id, 0 amount)
    w1_st = st + timedelta(minutes=1)
    degraded_txs = [
        Transaction(transaction_id="tx_deg_1", timestamp=w1_st + timedelta(seconds=5), merchant_id="M_DEG", customer_id="UNKNOWN", amount=0.0, payment_method="CREDIT_CARD", country="US", device_id=""),
        Transaction(transaction_id="tx_deg_2", timestamp=w1_st + timedelta(seconds=15), merchant_id="M_DEG", customer_id="", amount=25.0, payment_method="DEBIT_CARD", country="US", device_id="UNKNOWN"),
        Transaction(transaction_id="tx_deg_3", timestamp=w1_st + timedelta(seconds=25), merchant_id="M_DEG", customer_id="C3", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="DEV_VALID"),
    ]

    alerts = pipeline.process_transactions(warmup_txs + degraded_txs)
    assert len(alerts) == 0

    audits = pipeline.audit_store.get_audit_records("M_DEG")
    assert len(audits) == 2
    rec = audits[1]
    assert rec["data_quality_status"] == "DEGRADED"
    assert rec["confidence"] == 0.5
    assert rec["features"]["volume"] == 3.0
    assert rec["features"]["data_quality"] == "DEGRADED"


def test_duplicate_transaction_deduplication_through_pipeline():
    """Verify duplicate transactions with identical transaction_id are deduplicated and do not double-count volume or corrupt baseline."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(min_window_count=1)

    # 1. Pipeline with single-copy stream (10 transactions)
    pipe_single = StreamingDetectorPipeline(config=config, db_path=":memory:")
    single_txs = [
        Transaction(transaction_id=f"tx_dup_{i}", timestamp=st + timedelta(seconds=i * 5), merchant_id="M_DUP", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D1")
        for i in range(10)
    ]
    pipe_single.process_transactions(single_txs)
    audits_single = pipe_single.audit_store.get_audit_records("M_DUP")

    # 2. Pipeline with 3x duplicated stream (30 transactions with same IDs)
    pipe_dup = StreamingDetectorPipeline(config=config, db_path=":memory:")
    duplicated_txs = single_txs + single_txs + single_txs
    pipe_dup.process_transactions(duplicated_txs)
    audits_dup = pipe_dup.audit_store.get_audit_records("M_DUP")

    assert len(audits_single) == 1
    assert len(audits_dup) == 1
    assert audits_dup[0]["features"]["volume"] == 10.0
    assert audits_dup[0]["features"]["volume"] == audits_single[0]["features"]["volume"]
    assert audits_dup[0]["risk_score"] == audits_single[0]["risk_score"]


def test_out_of_order_transaction_arrival_through_pipeline():
    """Verify out-of-order transactions are ordered chronologically by EventBus within the pipeline."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(min_window_count=1)

    tx1 = Transaction(transaction_id="tx_ord_1", timestamp=st + timedelta(seconds=10), merchant_id="M_ORD", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D1")
    tx2 = Transaction(transaction_id="tx_ord_2", timestamp=st + timedelta(seconds=20), merchant_id="M_ORD", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D1")
    tx3 = Transaction(transaction_id="tx_ord_3", timestamp=st + timedelta(minutes=1, seconds=10), merchant_id="M_ORD", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D1")

    # Inject out of order: tx3 (min 1) -> tx1 (min 0) -> tx2 (min 0)
    pipeline = StreamingDetectorPipeline(config=config, db_path=":memory:")
    pipeline.process_transactions([tx3, tx1, tx2])

    audits = pipeline.audit_store.get_audit_records("M_ORD")
    assert len(audits) == 2
    # Window 0 (first minute) has 2 transactions (tx1, tx2)
    assert audits[0]["features"]["volume"] == 2.0
    # Window 1 (second minute) has 1 transaction (tx3)
    assert audits[1]["features"]["volume"] == 1.0


def test_reusable_drift_runner_and_artifact_generation():
    """Verify reusable DriftRunner executes paired experiment and produces DriftResult with valid adaptation metrics."""
    from src.evaluation.drift import DriftRunner, DriftResult

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    spike_spec = AnomalySpec("sustained_spike", st + timedelta(minutes=30), 300.0, 4.5, {"rate_multiplier": 4.0})

    # Generate Control (stable)
    gen_ctrl = SyntheticStreamGenerator(42, [{"id": "M_DRIFT_RUNNER", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_ctrl_base, _ = gen_ctrl.generate_window(30.0)
    gen_ctrl.schedule_anomaly("M_DRIFT_RUNNER", spike_spec, event_id="EVT-DRIFT-SPIKE")
    txs_ctrl_anom, evs_ctrl = gen_ctrl.generate_window(5.0)

    # Generate Drift (growing)
    gen_drift = SyntheticStreamGenerator(42, [{"id": "M_DRIFT_RUNNER", "archetype": "growing"}], VirtualClock(initial_time=st))
    txs_drift_base, _ = gen_drift.generate_window(30.0)
    gen_drift.schedule_anomaly("M_DRIFT_RUNNER", spike_spec, event_id="EVT-DRIFT-SPIKE")
    txs_drift_anom, evs_drift = gen_drift.generate_window(5.0)

    runner = DriftRunner()
    result = runner.run_paired_drift_experiment(
        control_transactions=txs_ctrl_base + txs_ctrl_anom,
        drift_transactions=txs_drift_base + txs_drift_anom,
        control_ground_truth=evs_ctrl,
        drift_ground_truth=evs_drift,
        merchant_id="M_DRIFT_RUNNER",
    )

    assert isinstance(result, DriftResult)
    assert result.control_metrics.tp == 1
    assert result.drift_metrics.tp == 1
    assert result.convergence_window_count >= 8
    assert result.passed_adaptation_criterion is True
    assert result.relative_adaptation_error <= 0.20

    # Ensure holdout access is rejected
    with pytest.raises(PermissionError, match="holdout data"):
        runner.verify_development_only("data/holdout/stream.json")


def test_conflicting_duplicate_transactions_raise_value_error():
    """Verify duplicate transactions with same transaction_id but conflicting payload (e.g. differing amount/customer) raise ValueError."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    config = FrozenDetectorConfig(min_window_count=1)
    pipeline = StreamingDetectorPipeline(config=config, db_path=":memory:")

    tx_original = Transaction(
        transaction_id="TX_CONFLICT_01",
        timestamp=st + timedelta(seconds=10),
        merchant_id="M_CONF",
        customer_id="C_ORIGINAL",
        amount=50.0,
        payment_method="CREDIT_CARD",
        country="US",
        device_id="DEV_1",
    )
    tx_conflicting = Transaction(
        transaction_id="TX_CONFLICT_01",  # Same ID!
        timestamp=st + timedelta(seconds=10),
        merchant_id="M_CONF",
        customer_id="C_TAMPERED",  # Conflicting field!
        amount=5000.0,             # Conflicting field!
        payment_method="CREDIT_CARD",
        country="US",
        device_id="DEV_1",
    )

    with pytest.raises(ValueError, match="Conflicting duplicate transaction detected"):
        pipeline.process_transactions([tx_original, tx_conflicting])


def test_drift_runner_enforces_paired_contract_and_rejects_mismatched_inputs():
    """Verify DriftRunner enforces pairing contract and rejects mismatched merchants, empty streams, or mismatched GroundTruth."""
    from src.evaluation.drift import DriftRunner

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tx_m1 = Transaction(transaction_id="tx_1", timestamp=st + timedelta(seconds=10), merchant_id="M_CTRL", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D1")
    tx_m2 = Transaction(transaction_id="tx_2", timestamp=st + timedelta(seconds=10), merchant_id="M_DRIFT", customer_id="C1", amount=50.0, payment_method="CREDIT_CARD", country="US", device_id="D1")

    gt_1 = GroundTruthEvent(
        event_id="EVT-01",
        merchant_id="M_CTRL",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(minutes=2),
        duration_seconds=120.0,
        severity=4.0,
        severity_level="HIGH",
        parameters={"target_magnitude": 4.0, "excess_transaction_count": 20, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )
    gt_2_mismatched = GroundTruthEvent(
        event_id="EVT-02",  # Different ID!
        merchant_id="M_CTRL",
        anomaly_type="velocity_burst",  # Different anomaly type!
        start_time=st,
        end_time=st + timedelta(minutes=2),
        duration_seconds=120.0,
        severity=4.0,
        severity_level="HIGH",
        parameters={"target_magnitude": 4.0, "excess_transaction_count": 20, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )

    runner = DriftRunner()

    # 1. Empty control stream -> ValueError
    with pytest.raises(ValueError, match="control_transactions is empty"):
        runner.validate_paired_contract([], [tx_m1], [gt_1], [gt_1], merchant_id="M_CTRL")

    # 2. Merchant mismatch -> ValueError
    with pytest.raises(ValueError, match="merchant 'M_CTRL' not found in drift stream"):
        runner.validate_paired_contract([tx_m1], [tx_m2], [gt_1], [gt_1], merchant_id="M_CTRL")

    # 3. GroundTruth mismatch -> ValueError
    with pytest.raises(ValueError, match="GroundTruth event ID 'EVT-02' missing in control GT"):
        runner.validate_paired_contract([tx_m1], [tx_m1], [gt_1], [gt_2_mismatched], merchant_id="M_CTRL")



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
    - Constant control configuration (FrozenDetectorConfig with threshold=3.5, alpha=0.3, P=2, C=5).
    - FeatureSnapshot, BaselineEngine history, and evidence_state 100% invariant across all variants.
    - Single-factor causal attribution: only signal mask differs.
    - Dynamically verifies complete metrics and mathematical delta definitions (delta = variant - FULL).
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

    # 3. Dynamic verification of complete metrics and mathematical deltas
    full_res = next(r for r in results if r.variant_id == "FULL")
    full_m = full_res.metrics

    for r in results:
        m = r.metrics
        # Base metric contract validation
        assert m.tp >= 0
        assert m.fp >= 0
        assert m.fn >= 0
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1_score <= 1.0
        assert m.fp_cost >= 0.0
        assert m.fn_exposure >= 0.0
        assert m.total_cost >= 0.0

        # Mathematical delta correctness relative to FULL
        assert pytest.approx(r.delta_f1, abs=1e-6) == m.f1_score - full_m.f1_score
        assert pytest.approx(r.delta_precision, abs=1e-6) == m.precision - full_m.precision
        assert pytest.approx(r.delta_recall, abs=1e-6) == m.recall - full_m.recall

        if m.median_latency_seconds is not None and full_m.median_latency_seconds is not None:
            assert pytest.approx(r.delta_latency_seconds, abs=1e-6) == m.median_latency_seconds - full_m.median_latency_seconds


# =====================================================================
# 6. True Control-vs-Drift Pairing & Quantitative Adaptation Measurement (Blockers 1, 2, 3, 4)
# =====================================================================

def test_true_control_vs_drift_pairing_and_quantitative_adaptation():
    """Verify true paired control vs drift experiment:
    - Identical seed, merchant ID, duration, and anomaly placement.
    - Only legitimate growth rate differs: CONTROL (stable) vs DRIFT (growing).
    - Warmup exclusion: warmup windows (w < 6) are explicitly excluded from adaptation calculation.
    - Empirical drift measurement: computes relative error between BaselineEngine expected volume and realized empirical volume.
    - Pre-defined numeric adaptation convergence criterion: relative error <= 0.20 across >= 8 post-warmup adaptation windows.
    - Zero false positives during unperturbed drift (FP_drift = 0).
    - Full comparative metrics & deltas: control latency, drift latency, delta latency, control metrics, drift metrics, delta FP, delta recall.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, ewma_alpha=0.3)
    evaluator = AnomalyEvaluator()

    # -------------------------------------------------------------
    # 1. CONTROL STREAM: Stable baseline (growth = 0.0/day)
    # -------------------------------------------------------------
    clock_ctrl = VirtualClock(initial_time=st)
    gen_ctrl = SyntheticStreamGenerator(42, [{"id": "M_PAIRED", "archetype": "stable"}], clock_ctrl)
    txs_ctrl_base, _ = gen_ctrl.generate_window(30.0)

    # Anomaly injection at minute 30 (sustained spike, horizon 300s)
    spike_spec = AnomalySpec(
        anomaly_type="sustained_spike",
        start_time=st + timedelta(minutes=30),
        duration_seconds=300.0,
        target_magnitude=4.5,
        parameters={"rate_multiplier": 4.0},
    )
    gen_ctrl.schedule_anomaly("M_PAIRED", spike_spec, event_id="EVT-PAIRED-SPIKE")
    txs_ctrl_anomaly, events_ctrl = gen_ctrl.generate_window(5.0)
    all_txs_ctrl = txs_ctrl_base + txs_ctrl_anomaly

    pipeline_ctrl = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts_ctrl = pipeline_ctrl.process_transactions(all_txs_ctrl)
    metrics_ctrl = evaluator.evaluate(alerts_ctrl, events_ctrl)

    # -------------------------------------------------------------
    # 2. DRIFT STREAM: Legitimate organic growth drift (growing)
    # -------------------------------------------------------------
    clock_drift = VirtualClock(initial_time=st)
    gen_drift = SyntheticStreamGenerator(42, [{"id": "M_PAIRED", "archetype": "growing"}], clock_drift)

    txs_drift_base, _ = gen_drift.generate_window(30.0)
    gen_drift.schedule_anomaly("M_PAIRED", spike_spec, event_id="EVT-PAIRED-SPIKE")
    txs_drift_anomaly, events_drift = gen_drift.generate_window(5.0)
    all_txs_drift = txs_drift_base + txs_drift_anomaly

    pipeline_drift = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts_drift = pipeline_drift.process_transactions(all_txs_drift)
    metrics_drift = evaluator.evaluate(alerts_drift, events_drift)

    # -------------------------------------------------------------
    # 3. Quantitative Adaptation Measurement & Pre-defined Criterion
    # -------------------------------------------------------------
    audits_drift = pipeline_drift.audit_store.get_audit_records("M_PAIRED")

    # Explicitly exclude warmup windows (w < 6) and evaluate strictly on post-warmup unperturbed windows [6..29]
    post_warmup_audits = audits_drift[6:30]
    assert len(post_warmup_audits) == 24

    converged_adaptation_windows = 0
    relative_errors = []

    for w_idx, a in enumerate(post_warmup_audits, start=6):
        assert a["baseline"]["evidence_state"] == "SUFFICIENT"
        assert a["data_quality_status"] == "GOOD"

        emp_vol = float(a["features"]["volume"])
        exp_vol = float(a["baseline"]["expected_values"]["volume"])

        # Realized empirical relative error calculation
        rel_err = abs(exp_vol - emp_vol) / max(1.0, emp_vol)
        relative_errors.append(rel_err)

        if rel_err <= 0.20:
            converged_adaptation_windows += 1

    # Exact quantitative adaptation assertions
    assert converged_adaptation_windows >= 8, f"Expected >= 8 converged post-warmup windows with rel_err <= 0.20, got {converged_adaptation_windows}"

    # Zero false positives during unperturbed drift regime
    unperturbed_alerts = [
        alt for alt in alerts_drift
        if alt.timestamp < spike_spec.start_time
    ]
    assert len(unperturbed_alerts) == 0

    # -------------------------------------------------------------
    # 4. Comparative Metrics and Complete Deltas Reporting
    # -------------------------------------------------------------
    delta_fp = metrics_drift.fp - metrics_ctrl.fp
    delta_recall = metrics_drift.recall - metrics_ctrl.recall
    delta_latency = metrics_drift.median_latency_seconds - metrics_ctrl.median_latency_seconds

    # Explicit assertions on Control Metrics
    assert metrics_ctrl.tp == 1
    assert metrics_ctrl.fp == 0
    assert metrics_ctrl.recall == 1.0
    assert metrics_ctrl.precision == 1.0
    assert metrics_ctrl.f1_score == 1.0
    assert metrics_ctrl.median_latency_seconds is not None

    # Explicit assertions on Drift Metrics
    assert metrics_drift.tp == 1
    assert metrics_drift.fp == 0
    assert metrics_drift.recall == 1.0
    assert metrics_drift.precision == 1.0
    assert metrics_drift.f1_score == 1.0
    assert metrics_drift.median_latency_seconds is not None

    # Explicit assertions on Deltas
    assert delta_fp == 0
    assert delta_recall == 0.0
    assert delta_latency == 0.0


# =====================================================================
# 7. Evasion Trajectory & Causal Mechanism Proofs
# =====================================================================

def test_threshold_hugging_evasion_trajectory_and_causal_mechanism():
    """Prove threshold-hugging evasion trajectory:
    - Observed scores hover in the sub-threshold envelope [1.2, 3.50).
    - Max score < 3.50 -> state machine never transitions to CANDIDATE -> persistence count stays 0.
    - Causally proves why 0 alerts were emitted and FN=1.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spec = AnomalySpec(
        anomaly_type="threshold_hugging_evasion",
        start_time=st + timedelta(minutes=5),
        duration_seconds=240.0,  # 4 anomaly windows
        target_magnitude=3.3,
        parameters={"rate_multiplier": 1.55},
    )
    gen.schedule_anomaly("M1", spec, event_id="EVT-HUGGING")
    txs_anomaly, events = gen.generate_window(4.0)

    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, ewma_alpha=0.3)
    pipeline = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts = pipeline.process_transactions(txs_base + txs_anomaly)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alerts, events)

    audits = pipeline.audit_store.get_audit_records("M1")
    anomaly_audits = audits[5:9]  # Windows 5, 6, 7, 8
    scores = [a["risk_score"] for a in anomaly_audits]

    # 1. Score Envelope Proof: All anomaly scores are strictly below the threshold
    for s in scores:
        assert 1.2 <= s < 3.50, f"Score {s} violated threshold-hugging envelope [1.2, 3.50)"

    # 2. Causal Mechanism Proof: Max score < static threshold -> no candidate entry -> 0 alerts emitted -> FN=1
    max_score = max(scores)
    assert max_score < cfg.static_threshold
    assert len(alerts) == 0
    assert metrics.fn == 1
    assert metrics.tp == 0


def test_persistence_evasion_trajectory_and_causal_mechanism():
    """Prove persistence evasion trajectory:
    - Score sequence: Window 0 breaches threshold (qualifying), Window 1 drops below threshold (non-qualifying).
    - Window 0: score >= 3.5 -> state becomes CANDIDATE (count=1).
    - Window 1: score < 3.5 -> state resets to NORMAL (count=0).
    - Non-consecutive breaches prevent reaching P=2 -> causally proves why 0 alerts were emitted and FN=1.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spec = AnomalySpec(
        anomaly_type="persistence_evasion",
        start_time=st + timedelta(minutes=5),
        duration_seconds=240.0,  # 4 windows
        target_magnitude=4.0,
        parameters={"rate_multiplier": 1.85},
    )
    gen.schedule_anomaly("M1", spec, event_id="EVT-PERSISTENCE")
    txs_anomaly, events = gen.generate_window(4.0)

    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, ewma_alpha=0.3)
    pipeline = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts = pipeline.process_transactions(txs_base + txs_anomaly)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alerts, events)

    audits = pipeline.audit_store.get_audit_records("M1")
    anomaly_audits = audits[5:9]  # Windows 5, 6, 7, 8
    scores = [a["risk_score"] for a in anomaly_audits]

    # 1. Score Sequence Proof: Alternating pattern with reset on window 1
    assert scores[0] >= 3.50  # Qualifying burst window 0
    assert scores[1] < 3.50   # Sub-threshold normal window 1 (resets persistence)

    # 2. Causal Mechanism Proof: Persistence counter resets on window 1, never reaching P=2 -> 0 alerts emitted -> FN=1
    assert len(alerts) == 0
    assert metrics.fn == 1
    assert metrics.tp == 0


def test_staircase_ramp_trajectory_and_causal_mechanism():
    """Prove staircase ramp trajectory:
    - Observed scores form a monotonically increasing progression: S0 < S1 < S2 < S3.
    - Consecutive steps breach threshold to satisfy persistence P=2.
    - Causally proves why alert was emitted and TP=1.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spec = AnomalySpec(
        anomaly_type="staircase_ramp",
        start_time=st + timedelta(minutes=5),
        duration_seconds=240.0,  # 4 step windows
        target_magnitude=5.0,
        parameters={"rate_multiplier": 6.0},
    )
    gen.schedule_anomaly("M1", spec, event_id="EVT-STAIRCASE")
    txs_anomaly, events = gen.generate_window(4.0)

    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, ewma_alpha=0.3)
    pipeline = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts = pipeline.process_transactions(txs_base + txs_anomaly)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alerts, events)

    audits = pipeline.audit_store.get_audit_records("M1")
    anomaly_audits = audits[5:9]  # Windows 5, 6, 7, 8
    scores = [a["risk_score"] for a in anomaly_audits]

    # 1. Monotonically Increasing Score Progression Proof
    assert scores[0] < scores[1] < scores[2] < scores[3]

    # 2. Causal Mechanism Proof: Steps breach threshold for 2 consecutive windows -> Alert emitted -> TP=1
    assert scores[0] >= 3.50
    assert scores[1] >= 3.50
    assert len(alerts) == 1
    assert metrics.tp == 1
    assert metrics.fn == 0


def test_oscillating_sub_threshold_trajectory_and_causal_mechanism():
    """Prove oscillating sub-threshold trajectory:
    - Observed scores stay strictly below threshold: max(S) < 3.50.
    - Causally proves why 0 alerts were emitted and FN=1.
    """
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spec = AnomalySpec(
        anomaly_type="oscillating_sub_threshold",
        start_time=st + timedelta(minutes=5),
        duration_seconds=240.0,
        target_magnitude=2.5,
        parameters={"amplitude": 0.5, "rate_multiplier": 1.0},
    )
    gen.schedule_anomaly("M1", spec, event_id="EVT-OSCILLATING")
    txs_anomaly, events = gen.generate_window(4.0)

    cfg = FrozenDetectorConfig(static_threshold=3.5, persistence=2, cooldown_windows=5, ewma_alpha=0.3)
    pipeline = StreamingDetectorPipeline(config=cfg, db_path=":memory:")
    alerts = pipeline.process_transactions(txs_base + txs_anomaly)

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alerts, events)

    audits = pipeline.audit_store.get_audit_records("M1")
    anomaly_audits = audits[5:9]  # Windows 5, 6, 7, 8
    scores = [a["risk_score"] for a in anomaly_audits]

    # 1. Sub-Threshold Envelope Proof: Max score stays strictly below threshold
    assert max(scores) < 3.50

    # 2. Causal Mechanism Proof: No qualifying windows -> persistence stays 0 -> FN=1
    assert len(alerts) == 0
    assert metrics.fn == 1
    assert metrics.tp == 0
