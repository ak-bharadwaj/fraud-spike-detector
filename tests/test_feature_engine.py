"""Comprehensive behavioral unit tests for Day 3 FeatureEngine.

Validates all 13 required feature extraction dimensions:
1. Exact window assignment ([window_start, window_end) half-open boundary).
2. Transaction counting (volume and velocity).
3. Amount aggregation (total, mean, std, min, max).
4. Customer cardinality (unique_customers).
5. Device cardinality (unique_devices).
6. Categorical distributions (high_risk_country_ratio, prepaid_payment_ratio).
7. Robust statistics (median_amount, mad_amount exact mathematical verification).
8. Empty / sparse windows (EMPTY data quality).
9. Multi-merchant isolation (Merchant A vs B partition stability).
10. Deterministic replay.
11. No future leakage (transactions at or after window_end excluded).
12. GroundTruth isolation (zero ground-truth imports in src/features).
13. FeatureSnapshot Pydantic schema compliance.
"""

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import numpy as np
import pytest

from src.contracts.contracts import Transaction, FeatureSnapshot
from src.features.feature_engine import FeatureEngine
from src.generator.stream_generator import SyntheticStreamGenerator
from src.stream.clock import VirtualClock


# =====================================================================
# 1. Exact Window Assignment ([window_start, window_end) half-open boundary)
# =====================================================================

def test_exact_window_assignment_half_open_boundary():
    """Verify half-open window boundary semantics: window_start <= timestamp < window_end."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=5.0)
    engine = FeatureEngine(window_duration_minutes=5.0)

    tx_start = Transaction(
        transaction_id="TX-START",
        timestamp=st,  # Exact window start -> INCLUDED
        merchant_id="M1",
        customer_id="C1",
        amount=100.0,
        payment_method="CREDIT_CARD",
        country="US",
        device_id="D1",
    )
    tx_mid = Transaction(
        transaction_id="TX-MID",
        timestamp=st + timedelta(minutes=2.5),  # Middle -> INCLUDED
        merchant_id="M1",
        customer_id="C2",
        amount=150.0,
        payment_method="CREDIT_CARD",
        country="US",
        device_id="D2",
    )
    tx_end = Transaction(
        transaction_id="TX-END",
        timestamp=et,  # Exact window end -> EXCLUDED
        merchant_id="M1",
        customer_id="C3",
        amount=200.0,
        payment_method="CREDIT_CARD",
        country="US",
        device_id="D3",
    )

    txs = [tx_start, tx_mid, tx_end]
    snap = engine.extract_snapshot("M1", txs, st, et)

    assert snap.volume == 2.0
    assert snap.amount_statistics["total_amount"] == 250.0


# =====================================================================
# 2. Transaction Counting (volume & velocity)
# =====================================================================

def test_transaction_counting_volume_and_velocity():
    """Verify volume (count) and velocity (rate per minute) over arbitrary window durations."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=10.0)
    engine = FeatureEngine(window_duration_minutes=10.0)

    txs = [
        Transaction(
            transaction_id=f"TX-{i}",
            timestamp=st + timedelta(minutes=i),
            merchant_id="M1",
            customer_id=f"C-{i}",
            amount=50.0,
            payment_method="DEBIT_CARD",
            country="US",
            device_id=f"D-{i}",
        )
        for i in range(8)
    ]

    snap = engine.extract_snapshot("M1", txs, st, et)

    assert snap.volume == 8.0
    assert snap.velocity == 0.8  # 8 txs / 10 minutes = 0.8 tx/min


# =====================================================================
# 3. Amount Aggregation (total, mean, std, min, max)
# =====================================================================

def test_amount_statistics_aggregation():
    """Verify total, mean, std, min, max amount statistics."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=1.0)
    engine = FeatureEngine(window_duration_minutes=1.0)

    amounts = [10.0, 20.0, 30.0, 40.0]
    txs = [
        Transaction(
            transaction_id=f"TX-{i}",
            timestamp=st + timedelta(seconds=i * 10),
            merchant_id="M1",
            customer_id=f"C-{i}",
            amount=amt,
            payment_method="CREDIT_CARD",
            country="US",
            device_id=f"D-{i}",
        )
        for i, amt in enumerate(amounts)
    ]

    snap = engine.extract_snapshot("M1", txs, st, et)

    stats = snap.amount_statistics
    assert stats["total_amount"] == 100.0
    assert stats["mean_amount"] == 25.0
    assert math.isclose(stats["std_amount"], np.std(amounts, ddof=1), abs_tol=1e-4)
    assert stats["min_amount"] == 10.0
    assert stats["max_amount"] == 40.0


# =====================================================================
# 4. Customer Cardinality & 5. Device Cardinality
# =====================================================================

def test_customer_and_device_cardinality():
    """Verify unique customer and unique device count aggregation."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=1.0)
    engine = FeatureEngine(window_duration_minutes=1.0)

    txs = [
        Transaction(transaction_id="TX-1", timestamp=st, merchant_id="M1", customer_id="CUST-A", amount=100.0, payment_method="CREDIT_CARD", country="US", device_id="DEV-X"),
        Transaction(transaction_id="TX-2", timestamp=st + timedelta(seconds=10), merchant_id="M1", customer_id="CUST-A", amount=120.0, payment_method="CREDIT_CARD", country="US", device_id="DEV-Y"),
        Transaction(transaction_id="TX-3", timestamp=st + timedelta(seconds=20), merchant_id="M1", customer_id="CUST-B", amount=140.0, payment_method="CREDIT_CARD", country="US", device_id="DEV-X"),
        Transaction(transaction_id="TX-4", timestamp=st + timedelta(seconds=30), merchant_id="M1", customer_id="CUST-C", amount=160.0, payment_method="CREDIT_CARD", country="US", device_id="DEV-Z"),
    ]

    snap = engine.extract_snapshot("M1", txs, st, et)

    assert snap.unique_customers == 3  # CUST-A, CUST-B, CUST-C
    assert snap.unique_devices == 3    # DEV-X, DEV-Y, DEV-Z
    assert math.isclose(snap.amount_statistics["customer_device_ratio"], 1.0)


# =====================================================================
# 6. Categorical Distributions
# =====================================================================

def test_categorical_distribution_features():
    """Verify country and payment method ratio distribution calculations."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=1.0)
    engine = FeatureEngine(window_duration_minutes=1.0)

    txs = [
        Transaction(transaction_id="TX-1", timestamp=st, merchant_id="M1", customer_id="C1", amount=100.0, payment_method="PREPAID_CARD", country="HIGH_RISK_GEO", device_id="D1"),
        Transaction(transaction_id="TX-2", timestamp=st + timedelta(seconds=10), merchant_id="M1", customer_id="C2", amount=100.0, payment_method="DEBIT_CARD", country="US", device_id="D2"),
        Transaction(transaction_id="TX-3", timestamp=st + timedelta(seconds=20), merchant_id="M1", customer_id="C3", amount=100.0, payment_method="CREDIT_CARD", country="US", device_id="D3"),
        Transaction(transaction_id="TX-4", timestamp=st + timedelta(seconds=30), merchant_id="M1", customer_id="C4", amount=100.0, payment_method="CREDIT_CARD", country="HIGH_RISK_GEO", device_id="D4"),
    ]

    snap = engine.extract_snapshot("M1", txs, st, et)

    stats = snap.amount_statistics
    assert stats["high_risk_country_ratio"] == 0.5  # 2 of 4 HIGH_RISK_GEO
    assert stats["prepaid_payment_ratio"] == 0.25   # 1 of 4 PREPAID_CARD
    assert stats["debit_payment_ratio"] == 0.25     # 1 of 4 DEBIT_CARD
    assert stats["credit_payment_ratio"] == 0.5     # 2 of 4 CREDIT_CARD


# =====================================================================
# 7. Robust Statistics (Median & MAD Exact Mathematical Verification)
# =====================================================================

def test_robust_statistics_median_and_mad():
    """Verify exact mathematical median and MAD (Median Absolute Deviation) statistics."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=1.0)
    engine = FeatureEngine(window_duration_minutes=1.0)

    # Amounts: [100.0, 200.0, 300.0, 400.0, 500.0]
    # Median = 300.0
    # Absolute deviations from median: [200.0, 100.0, 0.0, 100.0, 200.0]
    # Sorted abs deviations: [0.0, 100.0, 100.0, 200.0, 200.0]
    # MAD = median([0, 100, 100, 200, 200]) = 100.0
    amounts = [100.0, 200.0, 300.0, 400.0, 500.0]
    txs = [
        Transaction(transaction_id=f"TX-{i}", timestamp=st + timedelta(seconds=i*5), merchant_id="M1", customer_id=f"C-{i}", amount=amt, payment_method="CREDIT_CARD", country="US", device_id=f"D-{i}")
        for i, amt in enumerate(amounts)
    ]

    snap = engine.extract_snapshot("M1", txs, st, et)

    stats = snap.amount_statistics
    assert stats["median_amount"] == 300.0
    assert stats["mad_amount"] == 100.0


# =====================================================================
# 8. Empty / Sparse Windows
# =====================================================================

def test_empty_sparse_window_semantics():
    """Verify empty window produces volume=0, velocity=0, unique_counts=0, data_quality='EMPTY'."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=5.0)
    engine = FeatureEngine(window_duration_minutes=5.0)

    snap = engine.extract_snapshot("M1", [], st, et)

    assert snap.merchant_id == "M1"
    assert snap.timestamp == et
    assert snap.volume == 0.0
    assert snap.velocity == 0.0
    assert snap.unique_customers == 0
    assert snap.unique_devices == 0
    assert snap.data_quality == "EMPTY"
    assert snap.amount_statistics["total_amount"] == 0.0
    assert snap.amount_statistics["mean_amount"] == 0.0
    assert snap.amount_statistics["median_amount"] == 0.0


# =====================================================================
# 9. Multi-Merchant Isolation
# =====================================================================

def test_multi_merchant_isolation():
    """Verify processing multi-merchant transaction streams strictly partitions features per merchant."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=5.0)
    engine = FeatureEngine(window_duration_minutes=5.0)

    txs_m1 = [Transaction(transaction_id="TX-M1-1", timestamp=st, merchant_id="M1", customer_id="C1", amount=100.0, payment_method="CREDIT_CARD", country="US", device_id="D1")]
    txs_m2 = [
        Transaction(transaction_id="TX-M2-1", timestamp=st, merchant_id="M2", customer_id="C2", amount=200.0, payment_method="CREDIT_CARD", country="US", device_id="D2"),
        Transaction(transaction_id="TX-M2-2", timestamp=st + timedelta(minutes=1), merchant_id="M2", customer_id="C3", amount=300.0, payment_method="CREDIT_CARD", country="US", device_id="D3"),
    ]

    all_txs = txs_m1 + txs_m2

    snapshots = engine.extract_all_merchant_snapshots(["M1", "M2"], all_txs, st, et)

    assert snapshots["M1"].volume == 1.0
    assert snapshots["M1"].amount_statistics["total_amount"] == 100.0

    assert snapshots["M2"].volume == 2.0
    assert snapshots["M2"].amount_statistics["total_amount"] == 500.0


# =====================================================================
# 10. Deterministic Replay
# =====================================================================

def test_deterministic_feature_snapshot_replay():
    """Verify identical inputs produce identical FeatureSnapshot objects across separate runs."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    gen = SyntheticStreamGenerator(42, [{"id": "M1", "archetype": "stable"}], VirtualClock(initial_time=st))

    txs, _ = gen.generate_window(5.0)
    engine = FeatureEngine(window_duration_minutes=5.0)

    snap1 = engine.extract_snapshot("M1", txs, st, st + timedelta(minutes=5.0))
    snap2 = engine.extract_snapshot("M1", txs, st, st + timedelta(minutes=5.0))

    assert snap1 == snap2
    assert snap1.model_dump() == snap2.model_dump()


# =====================================================================
# 11. No Future Leakage
# =====================================================================

def test_no_future_leakage_exclusion_of_future_transactions():
    """Verify transactions occurring at or after window_end are strictly excluded."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    et = st + timedelta(minutes=5.0)
    engine = FeatureEngine(window_duration_minutes=5.0)

    tx_valid = Transaction(transaction_id="TX-VALID", timestamp=st + timedelta(minutes=2.0), merchant_id="M1", customer_id="C1", amount=100.0, payment_method="CREDIT_CARD", country="US", device_id="D1")
    tx_future = Transaction(transaction_id="TX-FUTURE", timestamp=et + timedelta(seconds=1), merchant_id="M1", customer_id="C2", amount=9999.0, payment_method="CREDIT_CARD", country="US", device_id="D2")

    snap = engine.extract_snapshot("M1", [tx_valid, tx_future], st, et)

    assert snap.volume == 1.0
    assert snap.amount_statistics["total_amount"] == 100.0


# =====================================================================
# 12. GroundTruth Isolation (AST Architectural Boundary Check)
# =====================================================================

def test_ground_truth_isolation_in_features_package():
    """Verify src/features code contains zero imports of ground_truth or GroundTruthEvent."""
    features_dir = Path(__file__).parent.parent / "src" / "features"
    py_files = list(features_dir.rglob("*.py"))

    assert len(py_files) > 0

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
# 13. FeatureSnapshot Schema Validation
# =====================================================================

def test_feature_snapshot_pydantic_schema_compliance():
    """Verify emitted FeatureSnapshot validates strictly against FeatureSnapshot Pydantic schema."""
    st = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    engine = FeatureEngine(window_duration_minutes=1.0)

    txs = [Transaction(transaction_id="TX-1", timestamp=st, merchant_id="M1", customer_id="C1", amount=100.0, payment_method="CREDIT_CARD", country="US", device_id="D1")]
    snap = engine.extract_snapshot("M1", txs, st)

    # Validate model serialization and re-instantiation
    dumped = snap.model_dump()
    reconstructed = FeatureSnapshot(**dumped)

    assert reconstructed.merchant_id == "M1"
    assert reconstructed.volume == 1.0
    assert reconstructed.data_quality == "GOOD"
