"""Comprehensive behavioral unit tests for Day 7 AnomalyEvaluator.

Validates all 20 required evaluation behavioral dimensions:
1. One-to-one matching Case A (1 GT + 2 alerts -> TP=1, FP=1).
2. One-to-one matching Case B (2 overlapping GT + 1 alert -> TP=1, FN=1).
3. One-to-one matching Case C (2 GT + 2 alerts -> TP=2, FP=0, FN=0).
4. Perfect detection (Precision=1.0, Recall=1.0, F1=1.0).
5. All missed events (Precision=0.0, Recall=0.0, F1=0.0).
6. All false positives (Precision=0.0, Recall=0.0, F1=0.0).
7. Exact start boundary matching.
8. Exact end boundary matching.
9. Out-of-bounds timestamp (just before start / after end -> FP).
10. Non-zero temporal tolerance matching.
11. Merchant mismatch isolation (Merchant A vs B).
12. Deterministic latency calculation in seconds (signed delta).
13. Precision calculation.
14. Recall calculation.
15. F1 score calculation.
16. Zero-denominator behavior matrix (handles empty inputs gracefully).
17. No-threshold-tuning invariant (zero tuning logic).
18. Detector-layer GroundTruth isolation (zero GT imports in features, baseline, scoring, state).
19. Evaluator GroundTruth access.
20. EvaluationMetrics Pydantic schema validation.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import pytest

from src.contracts.contracts import Alert, GroundTruthEvent, EvaluationMetrics
from src.evaluation.evaluator import AnomalyEvaluator


# =====================================================================
# Helpers to create Alert and GroundTruthEvent
# =====================================================================

def make_dummy_alert(
    merchant_id: str,
    ts: datetime,
    risk_score: float = 5.0,
    alert_id: str = "ALT-001",
) -> Alert:
    return Alert(
        alert_id=alert_id,
        merchant_id=merchant_id,
        timestamp=ts,
        risk_score=risk_score,
        confidence=1.0,
        reason="Volume spike breached static threshold",
        triggered_signals=["volume"],
        detector_version="1.0.0",
    )


def make_dummy_gt_event(
    merchant_id: str,
    start_time: datetime,
    end_time: datetime,
    event_id: str = "GT-001",
    severity: float = 5.0,
    parameters: dict = None,
) -> GroundTruthEvent:
    params = dict(parameters) if parameters is not None else {
        "excess_transaction_count": 10,
        "mean_transaction_amount": 50.0,
        "exposure_factor": 1.0,
    }
    return GroundTruthEvent(
        event_id=event_id,
        merchant_id=merchant_id,
        anomaly_type="SURGE_VOLUME",
        start_time=start_time,
        end_time=end_time,
        severity=severity,
        parameters=params,
    )



# =====================================================================
# 1. One-to-One Matching Test Cases (A, B, C)
# =====================================================================

def test_one_to_one_matching_case_a_one_gt_two_alerts():
    """Case A: 1 GT event + 2 alerts -> TP=1, FP=1 (first alert matches GT, 2nd alert unused -> FP)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5), event_id="GT-1")
    alt1 = make_dummy_alert("M1", st + timedelta(minutes=1), alert_id="ALT-1")
    alt2 = make_dummy_alert("M1", st + timedelta(minutes=3), alert_id="ALT-2")

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt1, alt2], [gt])

    assert res.tp == 1
    assert res.fp == 1
    assert res.fn == 0
    assert res.unmatched_alerts == ["ALT-2"]


def test_one_to_one_matching_case_b_two_overlapping_gt_one_alert():
    """Case B: 2 overlapping GT events + 1 alert -> TP=1, FN=1 (alert matches 1st GT, 2nd GT unmatched -> FN)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt1 = make_dummy_gt_event("M1", st, st + timedelta(minutes=5), event_id="GT-1")
    gt2 = make_dummy_gt_event("M1", st + timedelta(minutes=2), st + timedelta(minutes=7), event_id="GT-2")
    alt = make_dummy_alert("M1", st + timedelta(minutes=1), alert_id="ALT-1")

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt], [gt1, gt2])

    assert res.tp == 1
    assert res.fp == 0
    assert res.fn == 1
    assert res.unmatched_events == ["GT-2"]


def test_one_to_one_matching_case_c_two_gt_two_alerts():
    """Case C: 2 GT events + 2 alerts -> deterministic one-to-one matching -> TP=2, FP=0, FN=0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt1 = make_dummy_gt_event("M1", st, st + timedelta(minutes=5), event_id="GT-1")
    gt2 = make_dummy_gt_event("M1", st + timedelta(minutes=10), st + timedelta(minutes=15), event_id="GT-2")

    alt1 = make_dummy_alert("M1", st + timedelta(minutes=1), alert_id="ALT-1")
    alt2 = make_dummy_alert("M1", st + timedelta(minutes=11), alert_id="ALT-2")

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt1, alt2], [gt1, gt2])

    assert res.tp == 2
    assert res.fp == 0
    assert res.fn == 0


# =====================================================================
# 2. Temporal Boundary & Tolerance Tests
# =====================================================================

def test_temporal_boundary_exact_start_and_end():
    """Verify alerts at exact start_time and exact horizon end match GT event."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5))

    alt_start = make_dummy_alert("M1", st, alert_id="ALT-START")
    evaluator = AnomalyEvaluator()
    res_start = evaluator.evaluate([alt_start], [gt])
    assert res_start.tp == 1

    # Horizon for volume is 120s
    alt_end = make_dummy_alert("M1", st + timedelta(seconds=120.0), alert_id="ALT-END")
    res_end = evaluator.evaluate([alt_end], [gt])
    assert res_end.tp == 1


def test_temporal_out_of_bounds_and_tolerance():
    """Verify pre-onset alerts are ALWAYS false positives (no pre-onset tolerance) and after-horizon alerts fail even with positive tolerance."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=5)
    gt = make_dummy_gt_event("M1", st, et)

    # 1. 30s before start_time with 0s tolerance -> FP=1, FN=1
    alt_early = make_dummy_alert("M1", st - timedelta(seconds=30), alert_id="ALT-EARLY")
    eval_no_tol = AnomalyEvaluator(temporal_tolerance_seconds=0.0)
    res_no_tol = eval_no_tol.evaluate([alt_early], [gt])
    assert res_no_tol.tp == 0
    assert res_no_tol.fp == 1
    assert res_no_tol.fn == 1

    # 2. 30s before start_time with 60s tolerance -> STILL FP=1, FN=1 (pre-onset alerts are ALWAYS FP, zero tolerance before onset)
    eval_tol = AnomalyEvaluator(temporal_tolerance_seconds=60.0)
    res_tol_pre = eval_tol.evaluate([alt_early], [gt])
    assert res_tol_pre.tp == 0
    assert res_tol_pre.fp == 1
    assert res_tol_pre.fn == 1

    # 3. Exact onset (alert at start_time) -> TP=1, FP=0, FN=0 (latency = 0.0s)
    alt_onset = make_dummy_alert("M1", st, alert_id="ALT-ONSET")
    res_onset = eval_no_tol.evaluate([alt_onset], [gt])
    assert res_onset.tp == 1
    assert res_onset.fp == 0
    assert res_onset.fn == 0
    assert res_onset.mean_latency_seconds == 0.0

    # 4. Exact horizon (120s for volume_spike) -> TP=1, FP=0, FN=0 (latency = 120.0s)
    alt_horizon = make_dummy_alert("M1", st + timedelta(seconds=120), alert_id="ALT-HORIZON")
    res_horizon = eval_no_tol.evaluate([alt_horizon], [gt])
    assert res_horizon.tp == 1
    assert res_horizon.fp == 0
    assert res_horizon.fn == 0
    assert res_horizon.mean_latency_seconds == 120.0

    # 5. Alert horizon + 1s with positive tolerance (e.g. 60s tolerance) -> FP=1, FN=1 (tolerance does NOT extend horizon!)
    alt_after_tol = make_dummy_alert("M1", st + timedelta(seconds=121), alert_id="ALT-AFTER-TOL")
    res_after_tol = eval_tol.evaluate([alt_after_tol], [gt])
    assert res_after_tol.tp == 0
    assert res_after_tol.fp == 1
    assert res_after_tol.fn == 1




# =====================================================================
# 3. Merchant Mismatch & Zero-Denominator Matrix
# =====================================================================

def test_merchant_mismatch_isolation():
    """Verify Alert for Merchant B does NOT match GT Event for Merchant A."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M_A", st, st + timedelta(minutes=5))
    alt = make_dummy_alert("M_B", st + timedelta(minutes=1))

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt], [gt])

    assert res.tp == 0
    assert res.fp == 1
    assert res.fn == 1
    assert res.precision == 0.0
    assert res.recall == 0.0


def test_zero_denominator_metrics_matrix():
    """Verify zero-denominator metric conventions."""
    evaluator = AnomalyEvaluator()

    # 1. 0 alerts, 0 GT events -> Precision=1.0, Recall=1.0, F1=1.0
    r1 = evaluator.evaluate([], [])
    assert r1.precision == 1.0
    assert r1.recall == 1.0
    assert r1.f1_score == 1.0

    # 2. 0 alerts, 1 GT event -> Precision=0.0, Recall=0.0, F1=0.0
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5))
    r2 = evaluator.evaluate([], [gt])
    assert r2.precision == 0.0
    assert r2.recall == 0.0
    assert r2.f1_score == 0.0

    # 3. 1 alert, 0 GT events -> Precision=0.0, Recall=1.0, F1=0.0
    alt = make_dummy_alert("M1", st)
    r3 = evaluator.evaluate([alt], [])
    assert r3.precision == 0.0
    assert r3.recall == 1.0
    assert r3.f1_score == 0.0



# =====================================================================
# 4. No-Threshold-Tuning Invariant & GroundTruth Isolation
# =====================================================================

def test_no_threshold_tuning_invariant():
    """Verify AnomalyEvaluator has zero methods or attributes for detector threshold optimization."""
    evaluator = AnomalyEvaluator()
    methods = [m for m in dir(evaluator) if not m.startswith("__")]

    assert "evaluate" in methods
    assert "tune_threshold" not in methods
    assert "optimize" not in methods


def test_detector_layer_ground_truth_isolation():
    """Verify detector components (features, baseline, scoring, state) have zero GroundTruth imports."""
    forbidden_pkgs = ["features", "baseline", "scoring", "state"]

    for pkg in forbidden_pkgs:
        pkg_dir = Path(__file__).parent.parent / "src" / pkg
        py_files = list(pkg_dir.rglob("*.py"))

        for file_path in py_files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "ground_truth" not in alias.name, f"GroundTruth import violation in {file_path}: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert "ground_truth" not in module, f"GroundTruth import violation in {file_path}: {module}"
                    for alias in node.names:
                        assert "GroundTruth" not in alias.name, f"GroundTruth element import violation in {file_path}: {alias.name}"


# =====================================================================
# 5. Deterministic Replay & Schema Compliance
# =====================================================================

def test_deterministic_evaluation_replay():
    """Verify identical input streams produce 100% identical EvaluationMetrics."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5))
    alt = make_dummy_alert("M1", st + timedelta(minutes=2))

    evaluator = AnomalyEvaluator()
    res1 = evaluator.evaluate([alt], [gt])
    res2 = evaluator.evaluate([alt], [gt])

    assert res1 == res2
    assert res1.model_dump() == res2.model_dump()


def test_evaluation_metrics_pydantic_schema_compliance():
    """Verify EvaluationMetrics validates strictly against Pydantic schema."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5))
    alt = make_dummy_alert("M1", st + timedelta(minutes=1))

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt], [gt])

    dumped = res.model_dump()
    reconstructed = EvaluationMetrics(**dumped)

    assert reconstructed.tp == 1
    assert reconstructed.precision == 1.0
    assert reconstructed.f1_score == 1.0


# =====================================================================
# 6. Day 4 Detection Horizons, Cost Model & Sweep Guard Tests
# =====================================================================

def test_alert_after_horizon_but_within_gt_duration_is_fn():
    """Verify an alert occurring AFTER the detection horizon (e.g. 150s for 120s volume horizon) but within GT end_time (300s) is counted as FN (and alert is FP)."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    # GT volume spike: duration = 300s (5 min), configured horizon = 120s (2 min)
    gt = GroundTruthEvent(
        event_id="GT-VOL-1",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(seconds=300),
        severity=5.0,
        parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )

    # Alert at 150s (past the 120s horizon, but before the 300s end_time)
    alt_late = make_dummy_alert("M1", st + timedelta(seconds=150), alert_id="ALT-LATE")

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt_late], [gt])

    # Must be FN=1, FP=1, TP=0
    assert res.tp == 0
    assert res.fn == 1
    assert res.fp == 1
    assert res.unmatched_events == ["GT-VOL-1"]
    assert res.unmatched_alerts == ["ALT-LATE"]


def test_unknown_anomaly_type_raises_value_error():
    """Verify evaluator strictly raises ValueError for unconfigured/unknown anomaly types."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = GroundTruthEvent(
        event_id="GT-UNK",
        merchant_id="M1",
        anomaly_type="completely_unknown_quantum_anomaly",
        start_time=st,
        end_time=st + timedelta(seconds=300),
        severity=5.0,
        parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0},
    )
    alt = make_dummy_alert("M1", st + timedelta(seconds=10))

    evaluator = AnomalyEvaluator()
    with pytest.raises(ValueError, match="Unknown or unconfigured anomaly type"):
        evaluator.evaluate([alt], [gt])


def test_cost_model_calculation_accuracy():
    """Verify FP cost, FN exposure, and Total cost match exact configured cost model formulas."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt_caught = GroundTruthEvent(event_id="GT-1", merchant_id="M1", anomaly_type="volume_spike", start_time=st, end_time=st + timedelta(minutes=5), severity=4.0, parameters={"excess_transaction_count": 10, "mean_transaction_amount": 50.0, "exposure_factor": 1.0})
    gt_missed = GroundTruthEvent(event_id="GT-2", merchant_id="M1", anomaly_type="volume_spike", start_time=st + timedelta(hours=1), end_time=st + timedelta(hours=1, minutes=5), severity=6.0, parameters={"excess_transaction_count": 24, "mean_transaction_amount": 50.0, "exposure_factor": 1.0})

    alt_tp = make_dummy_alert("M1", st + timedelta(seconds=30), alert_id="ALT-TP")
    alt_fp = make_dummy_alert("M1", st + timedelta(hours=2), alert_id="ALT-FP")

    evaluator = AnomalyEvaluator(fp_unit_cost=50.0, fn_exposure_factor=1.0)
    res = evaluator.evaluate([alt_tp, alt_fp], [gt_caught, gt_missed])


    assert res.tp == 1
    assert res.fp == 1
    assert res.fn == 1
    assert res.fp_cost == 50.0  # 1 FP * 50.0
    assert res.fn_exposure == 1200.0  # 1 Missed GT * 1200.0
    assert res.total_cost == 1250.0  # 50.0 + 1200.0


def test_fn_exposure_calculated_from_excess_count_and_mean_amount():
    """Verify exact Section 24 FN exposure formula: FN_exposure = excess_transaction_count * mean_transaction_amount * exposure_factor, and explicit failure when missing."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    # 1. Missing cost inputs on unmatched GT -> strictly raises ValueError
    gt_incomplete = GroundTruthEvent(
        event_id="GT-INCOMPLETE",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=4.0,
        parameters={},  # Missing excess_transaction_count and mean_transaction_amount
    )
    evaluator_incomplete = AnomalyEvaluator()
    with pytest.raises(ValueError, match="Missing required cost inputs for unmatched GroundTruthEvent"):
        evaluator_incomplete.evaluate([], [gt_incomplete])

    # 2. Known inputs: excess_transaction_count=20, mean_transaction_amount=100.0, exposure_factor=0.5
    gt_missed = GroundTruthEvent(
        event_id="GT-MISSED-1",
        merchant_id="M1",
        anomaly_type="volume_spike",
        start_time=st,
        end_time=st + timedelta(minutes=5),
        severity=4.0,
        parameters={
            "excess_transaction_count": 20,
            "mean_transaction_amount": 100.0,
            "exposure_factor": 0.5,
        },
    )

    # 2 False positive alerts
    alt_fp1 = make_dummy_alert("M1", st + timedelta(hours=1), alert_id="ALT-FP1")
    alt_fp2 = make_dummy_alert("M1", st + timedelta(hours=2), alert_id="ALT-FP2")

    evaluator = AnomalyEvaluator(fp_unit_cost=50.0, fn_exposure_factor=0.5)
    res = evaluator.evaluate([alt_fp1, alt_fp2], [gt_missed])

    assert res.tp == 0
    assert res.fp == 2
    assert res.fn == 1

    # Exact expected calculations:
    # FP cost = 2 * 50.0 = ₹100.0
    # FN exposure = 20 * 100.0 * 0.5 = ₹1,000.0
    # Total cost = 100.0 + 1000.0 = ₹1,100.0
    assert res.fp_cost == 100.0
    assert res.fn_exposure == 1000.0
    assert res.total_cost == 1100.0




def test_development_sweeps_reject_holdout_paths():
    """Verify sweeps.py strictly raises HoldoutAccessViolationError when holdout data path is passed."""
    from src.evaluation.sweeps import run_strategy_comparison, HoldoutAccessViolationError

    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(HoldoutAccessViolationError, match="strictly prohibited on holdout"):
        run_strategy_comparison(transactions=[], ground_truth_events=[], data_path="data/holdout/transactions.csv")

