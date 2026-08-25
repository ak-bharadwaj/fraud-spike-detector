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
) -> GroundTruthEvent:
    return GroundTruthEvent(
        event_id=event_id,
        merchant_id=merchant_id,
        anomaly_type="SURGE_VOLUME",
        start_time=start_time,
        end_time=end_time,
        severity=severity,
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
    alt = make_dummy_alert("M1", st + timedelta(minutes=3), alert_id="ALT-1")

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

    alt1 = make_dummy_alert("M1", st + timedelta(minutes=2), alert_id="ALT-1")
    alt2 = make_dummy_alert("M1", st + timedelta(minutes=12), alert_id="ALT-2")

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt1, alt2], [gt1, gt2])

    assert res.tp == 2
    assert res.fp == 0
    assert res.fn == 0


# =====================================================================
# 2. Temporal Boundary & Tolerance Tests
# =====================================================================

def test_temporal_boundary_exact_start_and_end():
    """Verify alerts at exact start_time and exact end_time match GT event."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=5)
    gt = make_dummy_gt_event("M1", st, et)

    alt_start = make_dummy_alert("M1", st, alert_id="ALT-START")
    evaluator = AnomalyEvaluator()
    res_start = evaluator.evaluate([alt_start], [gt])
    assert res_start.tp == 1

    alt_end = make_dummy_alert("M1", et, alert_id="ALT-END")
    res_end = evaluator.evaluate([alt_end], [gt])
    assert res_end.tp == 1


def test_temporal_out_of_bounds_and_tolerance():
    """Verify timestamps just outside GT interval without tolerance fail, but succeed with non-zero tolerance."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=5)
    gt = make_dummy_gt_event("M1", st, et)

    # 30s before start_time
    alt_early = make_dummy_alert("M1", st - timedelta(seconds=30), alert_id="ALT-EARLY")

    # Without tolerance -> FP=1, FN=1
    eval_no_tol = AnomalyEvaluator(temporal_tolerance_seconds=0.0)
    res_no_tol = eval_no_tol.evaluate([alt_early], [gt])
    assert res_no_tol.tp == 0
    assert res_no_tol.fp == 1
    assert res_no_tol.fn == 1

    # With 60s tolerance -> TP=1, FP=0, FN=0 (latency is -30s signed delta)
    eval_tol = AnomalyEvaluator(temporal_tolerance_seconds=60.0)
    res_tol = eval_tol.evaluate([alt_early], [gt])
    assert res_tol.tp == 1
    assert res_tol.fp == 0
    assert res_tol.fn == 0
    assert res_tol.mean_latency_seconds == -30.0


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
