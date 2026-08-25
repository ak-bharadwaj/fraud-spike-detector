"""FeatureEngine module for deterministic streaming feature extraction.

Transforms transaction streams into per-merchant FeatureSnapshot objects over half-open
time windows [window_start, window_end).

Key Invariants:
- Pure deterministic feature transformer (caller owns virtual time progression).
- Strict half-open window boundary semantics: [window_start, window_end).
- Timezone-aware timestamp policy: rejects naive datetimes with TypeError.
- Native float double-precision preservation (no arbitrary extraction-layer rounding).
- GroundTruth isolation: NO imports of GroundTruthEvent, AnomalySpec, or ground truth metadata.
- Sparse window handling: empty windows return volume=0, velocity=0, data_quality="EMPTY".
- Robust statistics: exact median and MAD (Median Absolute Deviation: median(|x - median(x)|)).
- Multi-merchant isolation: merchant transactions are strictly partitioned.
- Schema compliance: all emitted snapshots validate against FeatureSnapshot contract.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence
import numpy as np

from src.contracts.contracts import Transaction, FeatureSnapshot


class FeatureEngine:
    """Deterministic feature extractor transforming transactions into per-merchant FeatureSnapshots."""

    def __init__(self, window_duration_minutes: float = 1.0):
        if window_duration_minutes <= 0:
            raise ValueError(f"window_duration_minutes must be positive, got {window_duration_minutes}")
        self.window_duration_minutes = float(window_duration_minutes)

    def extract_snapshot(
        self,
        merchant_id: str,
        transactions: Sequence[Transaction],
        window_start: datetime,
        window_end: Optional[datetime] = None,
    ) -> FeatureSnapshot:
        """Extract a FeatureSnapshot for a specific merchant over [window_start, window_end)."""
        if window_start.tzinfo is None:
            raise TypeError(f"window_start must be timezone-aware (got naive datetime {window_start})")

        w_start = window_start

        if window_end is None:
            w_end = w_start + timedelta(minutes=self.window_duration_minutes)
        else:
            if window_end.tzinfo is None:
                raise TypeError(f"window_end must be timezone-aware (got naive datetime {window_end})")
            w_end = window_end

        if w_end <= w_start:
            raise ValueError(f"window_end ({w_end}) must be strictly after window_start ({w_start})")

        duration_min = (w_end - w_start).total_seconds() / 60.0

        # Validate transaction timestamps and apply strict half-open window filtering
        merchant_txs = []
        for t in transactions:
            if t.timestamp.tzinfo is None:
                raise TypeError(f"Transaction '{t.transaction_id}' timestamp must be timezone-aware (got {t.timestamp})")
            if t.merchant_id == merchant_id and w_start <= t.timestamp < w_end:
                merchant_txs.append(t)

        if not merchant_txs:
            # Sparse / Empty window
            return FeatureSnapshot(
                merchant_id=merchant_id,
                timestamp=w_end,
                volume=0.0,
                velocity=0.0,
                amount_statistics={
                    "total_amount": 0.0,
                    "mean_amount": 0.0,
                    "std_amount": 0.0,
                    "median_amount": 0.0,
                    "mad_amount": 0.0,
                    "min_amount": 0.0,
                    "max_amount": 0.0,
                },
                unique_customers=0,
                unique_devices=0,
                data_quality="EMPTY",
            )

        n_txs = len(merchant_txs)
        volume = float(n_txs)
        velocity = volume / duration_min

        amounts = np.array([t.amount for t in merchant_txs], dtype=np.float64)
        total_amt = float(np.sum(amounts))
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts, ddof=1)) if n_txs > 1 else 0.0
        median_amt = float(np.median(amounts))
        mad_amt = float(np.median(np.abs(amounts - median_amt)))
        min_amt = float(np.min(amounts))
        max_amt = float(np.max(amounts))

        unique_custs = len({t.customer_id for t in merchant_txs})
        unique_devs = len({t.device_id for t in merchant_txs})

        amount_stats = {
            "total_amount": total_amt,
            "mean_amount": mean_amt,
            "std_amount": std_amt,
            "median_amount": median_amt,
            "mad_amount": mad_amt,
            "min_amount": min_amt,
            "max_amount": max_amt,
        }

        return FeatureSnapshot(
            merchant_id=merchant_id,
            timestamp=w_end,
            volume=volume,
            velocity=velocity,
            amount_statistics=amount_stats,
            unique_customers=unique_custs,
            unique_devices=unique_devs,
            data_quality="GOOD",
        )

    def extract_all_merchant_snapshots(
        self,
        merchant_ids: Sequence[str],
        transactions: Sequence[Transaction],
        window_start: datetime,
        window_end: Optional[datetime] = None,
    ) -> dict[str, FeatureSnapshot]:
        """Extract FeatureSnapshots for a collection of merchant IDs for the same window."""
        return {
            m_id: self.extract_snapshot(m_id, transactions, window_start, window_end)
            for m_id in merchant_ids
        }
