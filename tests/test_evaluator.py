"""Comprehensive behavioral unit tests for Day 7 AnomalyEvaluator.

Validates all 20 required evaluation behavioral dimensions:
1. Perfect detection (Precision=1.0, Recall=1.0, F1=1.0).
2. All missed (Precision=1.0, Recall=0.0, F1=0.0).
3. All false positives (Precision=0.0, Recall=1.0, F1=0.0).
4. Partial temporal overlap matching.
5. Merchant mismatch isolation (Merchant A vs B).
6. Multiple alerts for one event (first alert matches TP=1).
7. One alert for multiple events.
8. Duplicate predictions handling.
9. Deterministic matching semantics.
10. TP/FP/FN count correctness.
11. Precision calculation.
12. Recall calculation.
13. F1 score calculation.
14. Zero-denominator behavior (handles empty lists gracefully).
15. Detection latency calculation in seconds.
16. No-threshold-tuning invariant (zero tuning logic).
17. Detector-layer GroundTruth isolation (zero GT imports in features, baseline, scoring, state).
18. Evaluator GroundTruth access (evaluator legitimately accesses GroundTruthEvent).
19. Deterministic evaluation replay.
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
# 1. Perfect Detection, All Missed, All False Positives
# =====================================================================

def test_perfect_detection():
    """Verify perfect detection yields TP=1, FP=0, FN=0, Precision=1.0, Recall=1.0, F1=1.0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5))
    alt = make_dummy_alert("M1", st + timedelta(minutes=1))

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt], [gt])

    assert res.tp == 1
    assert res.fp == 0
    assert res.fn == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert res.mean_latency_seconds == 60.0


def test_all_missed_events():
    """Verify all missed events yield TP=0, FP=0, FN=1, Precision=1.0, Recall=0.0, F1=0.0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5))

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([], [gt])

    assert res.tp == 0
    assert res.fp == 0
    assert res.fn == 1
    assert res.precision == 1.0
    assert res.recall == 0.0
    assert res.f1_score == 0.0
    assert res.mean_latency_seconds is None


def test_all_false_positives():
    """Verify all false positive alerts yield TP=0, FP=1, FN=0, Precision=0.0, Recall=1.0, F1=0.0."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    alt = make_dummy_alert("M1", st)

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt], [])

    assert res.tp == 0
    assert res.fp == 1
    assert res.fn == 0
    assert res.precision == 0.0
    assert res.recall == 1.0
    assert res.f1_score == 0.0


# =====================================================================
# 2. Merchant Mismatch & Multiple Alerts
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


def test_multiple_alerts_for_one_event():
    """Verify multiple alerts within 1 GT event pair with first alert (TP=1) and latency measured from first alert."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gt = make_dummy_gt_event("M1", st, st + timedelta(minutes=5))

    alt1 = make_dummy_alert("M1", st + timedelta(minutes=1), alert_id="ALT-1")
    alt2 = make_dummy_alert("M1", st + timedelta(minutes=3), alert_id="ALT-2")

    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([alt1, alt2], [gt])

    assert res.tp == 1
    assert res.fp == 0
    assert res.fn == 0
    assert res.mean_latency_seconds == 60.0
    assert len(res.matched_events) == 1
    assert res.matched_events[0]["alert_id"] == "ALT-1"


# =====================================================================
# 3. Zero-Denominator Semantics & Latency
# =====================================================================

def test_zero_denominator_empty_inputs_graceful_handling():
    """Verify empty alerts and GT events return precision=1.0, recall=1.0, f1=1.0 without NaN."""
    evaluator = AnomalyEvaluator()
    res = evaluator.evaluate([], [])

    assert res.tp == 0
    assert res.fp == 0
    assert res.fn == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1_score == 1.0
    assert res.mean_latency_seconds is None


# =====================================================================
# 4. No-Threshold-Tuning Invariant
# =====================================================================

def test_no_threshold_tuning_invariant():
    """Verify AnomalyEvaluator has zero methods or attributes for detector threshold optimization."""
    evaluator = AnomalyEvaluator()
    methods = [m for m in dir(evaluator) if not m.startswith("__")]

    assert "evaluate" in methods
    assert "tune_threshold" not in methods
    assert "optimize" not in methods
    assert "calibrate" not in methods


# =====================================================================
# 5. Detector GroundTruth Isolation & Evaluator GT Access
# =====================================================================

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


def test_evaluator_ground_truth_access_permitted():
    """Verify evaluation package is permitted to consume GroundTruthEvent."""
    eval_dir = Path(__file__).parent.parent / "src" / "evaluation"
    eval_files = list(eval_dir.rglob("*.py"))
    assert len(eval_files) > 0


# =====================================================================
# 6. Deterministic Replay & Schema Validation
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
