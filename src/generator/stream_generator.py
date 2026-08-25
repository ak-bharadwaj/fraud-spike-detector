"""SyntheticStreamGenerator implementation.

Generates deterministic transaction streams and ground-truth events using VirtualClock and RNG sub-seeds.

Key Invariants:
- Uses VirtualClock for time management.
- Enforces No Overlapping Active Events per merchant.
- Produces valid Transaction and GroundTruthEvent output lists.
- Completely isolated from detector code.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence
import uuid
import numpy as np

from src.contracts.contracts import Transaction, GroundTruthEvent
from src.generator.archetypes import (
    MerchantProfile,
    create_merchant_profile,
    compute_legitimate_rate,
    sample_legitimate_amount,
)
from src.generator.anomalies import (
    AnomalySpec,
    create_ground_truth_event,
    compute_standardized_magnitude,
    compute_compound_severity,
)
from src.generator.rng import get_merchant_rng
from src.stream.clock import VirtualClock


class OverlapAnomalyError(ValueError):
    """Raised when an anomaly overlaps with an existing active anomaly for the same merchant."""

    pass


class SyntheticStreamGenerator:
    """Deterministic synthetic transaction stream and ground-truth generator."""

    def __init__(
        self,
        global_seed: int,
        merchant_configs: Sequence[dict[str, str]],
        clock: Optional[VirtualClock] = None,
    ):
        self.global_seed = global_seed
        self.clock = clock or VirtualClock()
        self.simulation_start = self.clock.current_time()

        # Build merchant profiles and RNGs
        self.profiles: dict[str, MerchantProfile] = {}
        self.merchant_rngs: dict[str, np.random.Generator] = {}
        self.active_events: dict[str, list[GroundTruthEvent]] = {}

        for cfg in merchant_configs:
            m_id = cfg["id"]
            arch = cfg.get("archetype", "stable")
            self.profiles[m_id] = create_merchant_profile(global_seed, m_id, arch)
            self.merchant_rngs[m_id] = get_merchant_rng(global_seed, f"{m_id}_stream")
            self.active_events[m_id] = []

    def schedule_anomaly(
        self,
        merchant_id: str,
        spec: AnomalySpec,
        event_id: Optional[str] = None,
    ) -> GroundTruthEvent:
        """Schedule an anomaly for a merchant, enforcing the No Overlapping Active Events invariant."""
        if merchant_id not in self.profiles:
            raise KeyError(f"Merchant '{merchant_id}' not configured in generator.")

        st = spec.start_time if spec.start_time.tzinfo else spec.start_time.replace(tzinfo=timezone.utc)
        et = spec.end_time if spec.end_time.tzinfo else spec.end_time.replace(tzinfo=timezone.utc)

        # Enforce No Overlapping Active Events invariant per merchant
        for existing in self.active_events[merchant_id]:
            ex_st = existing.start_time if existing.start_time.tzinfo else existing.start_time.replace(tzinfo=timezone.utc)
            ex_et = existing.end_time if existing.end_time.tzinfo else existing.end_time.replace(tzinfo=timezone.utc)

            # Check interval overlap: max(st, ex_st) < min(et, ex_et)
            if max(st, ex_st) < min(et, ex_et):
                raise OverlapAnomalyError(
                    f"Anomaly overlap rejected for merchant '{merchant_id}': "
                    f"New event [{st} .. {et}] overlaps existing event '{existing.event_id}' [{ex_st} .. {ex_et}]."
                )

        eid = event_id or f"EVT-{uuid.UUID(bytes=self.merchant_rngs[merchant_id].bytes(16)).hex[:8]}"
        gt_event = create_ground_truth_event(eid, merchant_id, spec)
        self.active_events[merchant_id].append(gt_event)
        return gt_event

    def generate_window(
        self,
        duration_minutes: float = 1.0,
        is_surge_active: dict[str, bool] = None,
    ) -> tuple[list[Transaction], list[GroundTruthEvent]]:
        """Generate transactions and active ground-truth events for the current time step."""
        is_surge_active = is_surge_active or {}
        window_start = self.clock.current_time()
        window_end = window_start + timedelta(minutes=duration_minutes)

        all_txs: list[Transaction] = []
        emitted_events: set[str] = set()
        active_gt_events: list[GroundTruthEvent] = []

        for m_id, profile in self.profiles.items():
            rng = self.merchant_rngs[m_id]

            # Find active anomalies for this merchant during [window_start, window_end)
            current_anomalies = [
                e for e in self.active_events[m_id]
                if max(window_start, e.start_time) < min(window_end, e.end_time)
            ]

            for e in current_anomalies:
                if e.event_id not in emitted_events:
                    active_gt_events.append(e)
                    emitted_events.add(e.event_id)

            # Base legitimate rate
            legit_rate = compute_legitimate_rate(
                profile=profile,
                current_time=window_start,
                simulation_start=self.simulation_start,
                is_surge_active=is_surge_active.get(m_id, False),
            )

            # Check if volume or velocity spike anomaly is active
            rate_multiplier = 1.0
            amount_multiplier = 1.0
            override_country: Optional[str] = None
            override_payment: Optional[str] = None
            is_behavioral_spike = False

            for anomaly in current_anomalies:
                atype = anomaly.anomaly_type
                params = anomaly.parameters

                if atype in ("velocity_spike", "volume_spike", "sustained_anomaly"):
                    rate_multiplier *= params.get("rate_multiplier", 3.0)
                elif atype == "amount_spike":
                    amount_multiplier *= params.get("amount_multiplier", 5.0)
                elif atype == "behavioral_shift":
                    is_behavioral_spike = True
                    rate_multiplier *= params.get("rate_multiplier", 2.0)
                elif atype == "attribute_anomaly":
                    override_country = params.get("country", "XX")
                    override_payment = params.get("payment_method", "PREPAID_CARD")
                elif atype == "compound_anomaly":
                    rate_multiplier *= params.get("rate_multiplier", 2.5)
                    amount_multiplier *= params.get("amount_multiplier", 3.0)
                    is_behavioral_spike = True
                    override_country = params.get("country", "HIGH_RISK_GEO")

            effective_rate = legit_rate * rate_multiplier
            # Expected transaction count in duration_minutes window (Poisson process)
            expected_count = int(np.round(effective_rate * duration_minutes))
            tx_count = max(0, int(rng.poisson(lam=max(0.1, expected_count))))

            for i in range(tx_count):
                # Distribute transaction timestamps deterministically within [window_start, window_end)
                offset_sec = float(rng.uniform(0.0, duration_minutes * 60.0))
                tx_time = window_start + timedelta(seconds=offset_sec)

                base_amt = sample_legitimate_amount(profile, rng)
                final_amt = round(max(1.0, base_amt * amount_multiplier), 2)

                tx_id = f"TX-{m_id}-{uuid.UUID(bytes=rng.bytes(16)).hex[:10]}"
                cust_id = f"CUST-{rng.integers(1, 50 if is_behavioral_spike else 5000)}"
                dev_id = f"DEV-{rng.integers(1, 30 if is_behavioral_spike else 3000)}"

                country = override_country or ("US" if rng.random() > 0.1 else "CA")
                payment = override_payment or ("CREDIT_CARD" if rng.random() > 0.2 else "DEBIT_CARD")

                tx = Transaction(
                    transaction_id=tx_id,
                    timestamp=tx_time,
                    merchant_id=m_id,
                    customer_id=cust_id,
                    amount=final_amt,
                    payment_method=payment,
                    country=country,
                    device_id=dev_id,
                )
                all_txs.append(tx)

        # Advance virtual clock
        self.clock.advance(duration_minutes * 60.0)

        # Sort emitted transactions by time
        all_txs.sort(key=lambda x: x.timestamp)
        return all_txs, active_gt_events
