"""Comprehensive unit tests for Day 4 Trusted Evaluator and Development-Only Parameter Sweeps.

Validates:
1. Config-Driven Detection Horizons & Authoritative evaluation.yaml:
   - velocity_burst = 60s
   - volume_spike = 120s
   - amount_shift = 180s
   - behavioral_anomaly = 180s
   - attribute_shift = 180s
   - sustained_spike = 300s
   - compound_anomaly = 300s
   - evasive_patterns = 300s
2. Exact Canonical Anomaly Mapping & Rejection of Unknown Anomaly Types:
   - Validates exact dictionary lookup without substring guessing.
   - Raises ValueError on unknown anomaly types.
3. Cost Model & FN Exposure Semantics:
   - fp_cost = fp * fp_review_cost
   - fn_exposure = sum(fn_exposure_factor * event_exposure for unmatched GT)
   - total_cost = fp_cost + fn_exposure
4. Configuration Integration:
   - Proves changing EvaluationConfig changes horizons, FP cost, and FN exposure dynamically.
5. Boundary & One-to-One Greedy Matching:
   - Alert exactly at GT.start_time (latency = 0.0s)
   - Alert exactly at GT.start_time + horizon
   - Alert just beyond horizon (FP)
   - Pre-onset alert (alert.timestamp < GT.start_time -> FP)
   - 1 GT + multiple alerts (TP=1, remaining FP)
   - Multiple GT + 1 alert (TP=1, remaining FN)
6. Latency Calculation & Statistics:
   - Mean, Median, P95 latency
7. Zero-Denominator Matrix.
8. Development Parameter Sweeps:
   - Alpha sweep across {0.2, 0.3, 0.5, 0.7, 0.9}
   - Persistence sweep across {1, 2, 3}
   - Threshold operating point search over [1.0, 10.0] with step 0.5
9. Hard Holdout Isolation:
   - Prohibits holdout data access in sweeps.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from src.contracts.contracts import (
    Transaction,
    Alert,
    GroundTruthEvent,
    FrozenDetectorConfig,
    EvaluationMetrics,
)
from src.contracts.config_schemas import EvaluationConfig, CostModelConfig
from src.contracts.config_loader import load_evaluation_config
from src.evaluation.evaluator import (
    AnomalyEvaluator,
    ANOMALY_HORIZON_MAP,
)
from src.evaluation.sweeps import (
    run_alpha_sweep,
    run_persistence_sweep,
    run_threshold_operating_point_sweep,
    HoldoutAccessViolationError,
    _verify_development_only_data,
)
from src.generator.stream_generator import SyntheticStreamGenerator
from src.generator.anomalies import AnomalySpec
from src.stream.clock import VirtualClock


# =====================================================================
# 1. Authoritative Config & Exact Horizon Resolution Tests
# =====================================================================

def test_authoritative_evaluation_yaml_loading():
    """Verify AnomalyEvaluator loads horizons and cost model strictly from config/evaluation.yaml."""
    evaluator = AnomalyEvaluator()
    assert evaluator.resolve_horizon("velocity_burst") == 60.0
    assert evaluator.resolve_horizon("volume_spike") == 120.0
    assert evaluator.resolve_horizon("amount_shift") == 180.0
    assert evaluator.resolve_horizon("behavioral_anomaly") == 180.0
    assert evaluator.resolve_horizon("attribute_shift") == 180.0
    assert evaluator.resolve_horizon("sustained_spike") == 300.0
    assert evaluator.resolve_horizon("compound_anomaly") == 300.0
    assert evaluator.resolve_horizon("evasive_patterns") == 300.0
    assert evaluator.fp_review_cost == 50.0
    assert evaluator.fn_exposure_factor == 1.0


def test_unknown_anomaly_type_rejection():
    """Verify unknown or unconfigured anomaly types raise ValueError (no substring guessing)."""
    evaluator = AnomalyEvaluator()
    with pytest.raises(ValueError, match="Unknown or unconfigured anomaly type"):
        evaluator.resolve_horizon("arbitrary_unknown_anomaly")


def test_custom_evaluation_config_integration():
    """Verify changing validated EvaluationConfig alters horizons, FP cost, and FN exposure."""
    custom_cfg = EvaluationConfig(
        horizons={
            "velocity_burst": 45,
            "volume_spike": 90,
            "amount_shift": 150,
            "behavioral_anomaly": 150,
            "attribute_shift": 150,
            "sustained_spike": 240,
            "compound_anomaly": 240,
            "evasive_patterns": 240,
        },
        cost_model=CostModelConfig(
            fp_review_cost=75.0,
            fn_exposure_factor=2.0,
        ),
    )
    evaluator = AnomalyEvaluator(config=custom_cfg)
    assert evaluator.resolve_horizon("velocity_burst") == 45.0
    assert evaluator.resolve_horizon("volume_spike") == 90.0
    assert evaluator.fp_review_cost == 75.0
    assert evaluator.fn_exposure_factor == 2.0

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="GT-01",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=3.0,
        parameters={"excess_transaction_count": 4, "mean_transaction_amount": 50.0, "exposure_factor": 2.0},
    )
    # Alert at 100s (> 90s custom horizon) -> FP=1, FN=1
    alt = Alert(
        alert_id="ALT-01",
        merchant_id="M1",
        timestamp=st + timedelta(seconds=100.0),
        risk_score=4.0,
        confidence=1.0,
        reason="breach",
        detector_version="1.0.0",
    )
    res = evaluator.evaluate([alt], [gt])
    assert res.tp == 0
    assert res.fp == 1
    assert res.fn == 1
    assert res.fp_cost == 75.0  # 1 * 75.0
    assert res.fn_exposure == 400.0  # 4 * 50.0 * 2.0
    assert res.total_cost == 475.0


# =====================================================================
# 2. Boundary & Pre-Onset Horizon Matching Tests
# =====================================================================

def test_evaluator_alert_exactly_at_gt_start():
    """Verify alert occurring exactly at GT.start_time is a valid detection with latency = 0.0s."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="GT-01",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=4.0,
        parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )
    alt = Alert(
        alert_id="ALT-01",
        merchant_id="M1",
        timestamp=st,
        risk_score=4.0,
        confidence=1.0,
        reason="Threshold breach",
        detector_version="1.0.0",
    )
    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate([alt], [gt])

    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.mean_latency_seconds == 0.0
    assert metrics.median_latency_seconds == 0.0
    assert metrics.p95_latency_seconds == 0.0


def test_evaluator_alert_exactly_at_horizon_end():
    """Verify alert occurring exactly at GT.start_time + horizon (120s for volume) is valid."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="GT-01",
        merchant_id="M1",
        anomaly_type="volume_spike",  # horizon = 120s
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=4.0,
        parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )
    alt = Alert(
        alert_id="ALT-01",
        merchant_id="M1",
        timestamp=st + timedelta(seconds=120.0),  # Exactly at horizon
        risk_score=4.0,
        confidence=1.0,
        reason="Threshold breach",
        detector_version="1.0.0",
    )
    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate([alt], [gt])

    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.median_latency_seconds == 120.0


def test_evaluator_alert_just_beyond_horizon():
    """Verify alert occurring past GT.start_time + horizon (121s for volume) is NOT matched (FP=1, FN=1)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="GT-01",
        merchant_id="M1",
        anomaly_type="volume_spike",  # horizon = 120s
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=4.0,
        parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )
    alt = Alert(
        alert_id="ALT-01",
        merchant_id="M1",
        timestamp=st + timedelta(seconds=121.0),  # Beyond horizon
        risk_score=4.0,
        confidence=1.0,
        reason="Threshold breach",
        detector_version="1.0.0",
    )
    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate([alt], [gt])

    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_evaluator_pre_onset_alert_is_false_positive():
    """Verify alert occurring before GT.start_time (even by 1s) is strictly a False Positive."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="GT-01",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=4.0,
        parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )
    alt_pre = Alert(
        alert_id="ALT-PRE",
        merchant_id="M1",
        timestamp=st - timedelta(seconds=1.0),  # Pre-onset
        risk_score=4.0,
        confidence=1.0,
        reason="Pre-onset threshold breach",
        detector_version="1.0.0",
    )
    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate([alt_pre], [gt])

    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.unmatched_alerts == ["ALT-PRE"]
    assert metrics.unmatched_events == ["GT-01"]


# =====================================================================
# 3. Greedy One-to-One Matching & Cost Model Tests
# =====================================================================

def test_evaluator_one_gt_multiple_alerts_and_cost_model():
    """Verify 1 GT with 3 alerts produces TP=1, FP=2 and calculates exact separate FP cost and FN exposure."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="GT-01",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=4.0,
        parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )
    alts = [
        Alert(alert_id="ALT-1", merchant_id="M1", timestamp=st + timedelta(seconds=30), risk_score=4.0, confidence=1.0, reason="breach", detector_version="1.0.0"),
        Alert(alert_id="ALT-2", merchant_id="M1", timestamp=st + timedelta(seconds=60), risk_score=4.5, confidence=1.0, reason="breach", detector_version="1.0.0"),
        Alert(alert_id="ALT-3", merchant_id="M1", timestamp=st + timedelta(seconds=90), risk_score=5.0, confidence=1.0, reason="breach", detector_version="1.0.0"),
    ]
    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate(alts, [gt])

    assert metrics.tp == 1
    assert metrics.fp == 2
    assert metrics.fn == 0
    assert metrics.precision == 1.0 / 3.0
    assert metrics.recall == 1.0
    assert metrics.fp_cost == 100.0  # 2 * 50.0
    assert metrics.fn_exposure == 0.0
    assert metrics.total_cost == 100.0
    assert metrics.median_latency_seconds == 30.0
    assert metrics.p95_latency_seconds == 30.0


def test_evaluator_multiple_gt_one_alert():
    """Verify 2 GT events with 1 alert produces TP=1, FN=1, FP=0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt1 = GroundTruthEvent(event_id="GT-01", merchant_id="M1", anomaly_type="volume_spike", start_time=st, end_time=st + timedelta(minutes=2), severity=4.0, parameters={"excess_transaction_count": 6, "mean_transaction_amount": 50.0, "exposure_factor": 1.0})
    gt2 = GroundTruthEvent(event_id="GT-02", merchant_id="M1", anomaly_type="volume_spike", start_time=st + timedelta(minutes=3), end_time=st + timedelta(minutes=5), severity=4.0, parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0})

    alt = Alert(alert_id="ALT-1", merchant_id="M1", timestamp=st + timedelta(seconds=30), risk_score=4.0, confidence=1.0, reason="breach", detector_version="1.0.0")

    evaluator = AnomalyEvaluator()
    metrics = evaluator.evaluate([alt], [gt1, gt2])

    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.fp_cost == 0.0
    assert metrics.fn_exposure == 500.0  # 10 * 50.0 * 1.0
    assert metrics.total_cost == 500.0


def test_evaluator_zero_denominator_matrix():
    """Verify zero-denominator edge cases."""
    evaluator = AnomalyEvaluator()

    # Case 1: Empty alerts, empty GT events -> P=1, R=1, F1=1
    m1 = evaluator.evaluate([], [])
    assert m1.tp == 0 and m1.fp == 0 and m1.fn == 0
    assert m1.precision == 1.0 and m1.recall == 1.0 and m1.f1_score == 1.0
    assert m1.median_latency_seconds is None and m1.p95_latency_seconds is None
    assert m1.fp_cost == 0.0 and m1.fn_exposure == 0.0

    # Case 2: Events with zero alerts -> P=0, R=0, F1=0
    gt = GroundTruthEvent(event_id="GT-01", merchant_id="M1", anomaly_type="volume_spike", start_time=datetime(2026, 1, 1, tzinfo=timezone.utc), end_time=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc), severity=4.0, parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0})
    m2 = evaluator.evaluate([], [gt])
    assert m2.tp == 0 and m2.fp == 0 and m2.fn == 1
    assert m2.precision == 0.0 and m2.recall == 0.0 and m2.f1_score == 0.0


    # Case 3: Alerts with zero events -> P=0, R=1, F1=0
    alt = Alert(alert_id="ALT-01", merchant_id="M1", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), risk_score=4.0, confidence=1.0, reason="r", detector_version="1.0.0")
    m3 = evaluator.evaluate([alt], [])
    assert m3.tp == 0 and m3.fp == 1 and m3.fn == 0
    assert m3.precision == 0.0 and m3.recall == 1.0 and m3.f1_score == 0.0


# =====================================================================
# 4. Development Parameter Sweeps & Hard Holdout Protection
# =====================================================================

def test_development_alpha_sweep_execution():
    """Verify development alpha sweep evaluates across {0.2, 0.3, 0.5, 0.7, 0.9}."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spike_spec = AnomalySpec("volume_spike", st + timedelta(minutes=5), 180.0, 4.5, {"rate_multiplier": 4.0})
    gen.schedule_anomaly("M1", spike_spec, event_id="EVT-ALPHA-SWEEP")
    txs_spike, events = gen.generate_window(3.0)

    all_txs = txs_base + txs_spike

    results = run_alpha_sweep(all_txs, events, alphas=[0.2, 0.3, 0.5, 0.7, 0.9])
    assert len(results) == 5

    for row in results:
        assert "alpha" in row
        assert "precision" in row
        assert "recall" in row
        assert "f1_score" in row
        assert "median_latency_seconds" in row
        assert "p95_latency_seconds" in row
        assert "fp_cost" in row
        assert "fn_exposure" in row


def test_development_persistence_sweep_execution():
    """Verify development persistence sweep evaluates across {1, 2, 3}."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spike_spec = AnomalySpec("volume_spike", st + timedelta(minutes=5), 180.0, 4.5, {"rate_multiplier": 4.0})
    gen.schedule_anomaly("M1", spike_spec, event_id="EVT-PERSIST-SWEEP")
    txs_spike, events = gen.generate_window(3.0)

    all_txs = txs_base + txs_spike

    results = run_persistence_sweep(all_txs, events, persistences=[1, 2, 3])
    assert len(results) == 3

    for row in results:
        assert "persistence" in row
        assert "precision" in row
        assert "recall" in row
        assert "f1_score" in row


def test_development_threshold_operating_point_sweep():
    """Verify development threshold operating point sweep evaluates across grid [1.0, 10.0] step 0.5."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))
    txs_base, _ = gen.generate_window(5.0)

    spike_spec = AnomalySpec("volume_spike", st + timedelta(minutes=5), 180.0, 4.5, {"rate_multiplier": 4.0})
    gen.schedule_anomaly("M1", spike_spec, event_id="EVT-TH-SWEEP")
    txs_spike, events = gen.generate_window(3.0)

    all_txs = txs_base + txs_spike

    results = run_threshold_operating_point_sweep(all_txs, events)
    assert len(results) == 19  # [1.0, 1.5, 2.0, ..., 10.0]

    for row in results:
        assert "threshold" in row
        assert "f1_score" in row
        assert "fp_cost" in row
        assert "fn_exposure" in row


def test_development_sweep_holdout_access_prohibited():
    """Verify development parameter sweep raises HoldoutAccessViolationError if holdout path is supplied."""
    with pytest.raises(HoldoutAccessViolationError, match="strictly prohibited on holdout data"):
        _verify_development_only_data("data/holdout/transactions.json")
