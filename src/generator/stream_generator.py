"""SyntheticStreamGenerator implementation.

Generates deterministic transaction streams and ground-truth events using VirtualClock and RNG sub-seeds.

Key Invariants:
- Uses VirtualClock for time management.
- Enforces No Overlapping Active Events per merchant.
- Derives realized ground-truth magnitude from actual generated transaction stream statistics over the anomaly's exact temporal interval [start_time, end_time).
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

        self.profiles: dict[str, MerchantProfile] = {}
        self.merchant_rngs: dict[str, np.random.Generator] = {}
        self.scheduled_specs: dict[str, list[tuple[str, AnomalySpec]]] = {}
        self.anomaly_tx_history: dict[str, list[Transaction]] = {}
        self.active_events: dict[str, list[GroundTruthEvent]] = {}

        for cfg in merchant_configs:
            m_id = cfg["id"]
            arch = cfg.get("archetype", "stable")
            self.profiles[m_id] = create_merchant_profile(global_seed, m_id, arch)
            self.merchant_rngs[m_id] = get_merchant_rng(global_seed, f"{m_id}_stream")
            self.scheduled_specs[m_id] = []
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
        for _, existing_spec in self.scheduled_specs[merchant_id]:
            ex_st = existing_spec.start_time if existing_spec.start_time.tzinfo else existing_spec.start_time.replace(tzinfo=timezone.utc)
            ex_et = existing_spec.end_time if existing_spec.end_time.tzinfo else existing_spec.end_time.replace(tzinfo=timezone.utc)

            if max(st, ex_st) < min(et, ex_et):
                raise OverlapAnomalyError(
                    f"Anomaly overlap rejected for merchant '{merchant_id}': "
                    f"New spec [{st} .. {et}] overlaps existing spec [{ex_st} .. {ex_et}]."
                )

        eid = event_id or f"EVT-{uuid.UUID(bytes=self.merchant_rngs[merchant_id].bytes(16)).hex[:8]}"
        self.scheduled_specs[merchant_id].append((eid, spec))
        self.anomaly_tx_history[eid] = []

        gt_event = create_ground_truth_event(eid, merchant_id, spec, realized_magnitude=spec.target_magnitude)
        self.active_events[merchant_id].append(gt_event)
        return gt_event

    def generate_window(
        self,
        duration_minutes: float = 1.0,
        is_surge_active: dict[str, bool] = None,
    ) -> tuple[list[Transaction], list[GroundTruthEvent]]:
        """Generate transactions and realized ground-truth events for the current time step."""
        is_surge_active = is_surge_active or {}
        window_start = self.clock.current_time()
        window_end = window_start + timedelta(minutes=duration_minutes)

        all_txs: list[Transaction] = []
        emitted_events: set[str] = set()
        active_gt_events: list[GroundTruthEvent] = []

        for m_id, profile in self.profiles.items():
            rng = self.merchant_rngs[m_id]

            current_specs = [
                (eid, spec) for eid, spec in self.scheduled_specs[m_id]
                if max(window_start, spec.start_time) < min(window_end, spec.end_time)
            ]

            legit_rate = compute_legitimate_rate(
                profile=profile,
                current_time=window_start,
                simulation_start=self.simulation_start,
                is_surge_active=is_surge_active.get(m_id, False),
            )

            rate_multiplier = 1.0
            amount_multiplier = 1.0
            override_country: Optional[str] = None
            override_payment: Optional[str] = None
            is_behavioral_spike = False

            for eid, spec in current_specs:
                atype = spec.anomaly_type
                params = spec.parameters
                tm = spec.target_magnitude

                if atype == "velocity_spike":
                    rate_multiplier *= params.get("rate_multiplier", max(3.0, tm))
                elif atype in ("volume_spike", "sustained_anomaly"):
                    rate_multiplier *= params.get("rate_multiplier", max(2.5, tm))
                elif atype == "amount_spike":
                    amount_multiplier *= params.get("amount_multiplier", max(3.0, tm))
                elif atype == "behavioral_shift":
                    is_behavioral_spike = True
                    rate_multiplier *= params.get("rate_multiplier", 1.8)
                elif atype == "attribute_anomaly":
                    override_country = params.get("country", "HIGH_RISK_GEO")
                    override_payment = params.get("payment_method", "PREPAID_CARD")
                elif atype == "compound_anomaly":
                    rate_multiplier *= params.get("rate_multiplier", max(2.5, tm * 0.8))
                    amount_multiplier *= params.get("amount_multiplier", max(3.0, tm * 0.8))
                    is_behavioral_spike = True
                    override_country = params.get("country", "HIGH_RISK_GEO")

            effective_rate = legit_rate * rate_multiplier
            expected_count = int(np.round(effective_rate * duration_minutes))
            tx_count = max(0, int(rng.poisson(lam=max(0.1, expected_count))))

            window_txs: list[Transaction] = []
            for i in range(tx_count):
                offset_sec = float(rng.uniform(0.0, duration_minutes * 60.0))
                tx_time = window_start + timedelta(seconds=offset_sec)

                base_amt = sample_legitimate_amount(profile, rng)
                final_amt = round(max(1.0, base_amt * amount_multiplier), 2)

                tx_id = f"TX-{m_id}-{uuid.UUID(bytes=rng.bytes(16)).hex[:10]}"
                cust_id = f"CUST-{rng.integers(1, 10 if is_behavioral_spike else 5000)}"
                dev_id = f"DEV-{rng.integers(1, 5 if is_behavioral_spike else 3000)}"

                country = override_country or ("US" if rng.random() > 0.02 else "HIGH_RISK_GEO")
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
                window_txs.append(tx)

            all_txs.extend(window_txs)

            # Record transactions occurring within each anomaly's exact duration [start_time, end_time)
            for eid, spec in current_specs:
                st = spec.start_time if spec.start_time.tzinfo else spec.start_time.replace(tzinfo=timezone.utc)
                et = spec.end_time if spec.end_time.tzinfo else spec.end_time.replace(tzinfo=timezone.utc)

                txs_in_spec = [t for t in window_txs if st <= t.timestamp < et]
                self.anomaly_tx_history[eid].extend(txs_in_spec)

                accumulated_txs = self.anomaly_tx_history[eid]
                total_duration_min = max(0.1, spec.duration_seconds / 60.0)

                # Compute baseline expected rate at spec.start_time to ensure invariance across step sizes
                expected_spec_rate = compute_legitimate_rate(
                    profile=profile,
                    current_time=st,
                    simulation_start=self.simulation_start,
                    is_surge_active=is_surge_active.get(m_id, False),
                )

                # Measure actual stream statistics over elapsed portion of anomaly duration
                elapsed_min = min(total_duration_min, max(0.1, (window_end - st).total_seconds() / 60.0))
                obs_rate = len(accumulated_txs) / elapsed_min
                scale_rate = max(0.5, 0.2 * expected_spec_rate)
                m_rate = compute_standardized_magnitude(obs_rate, expected_spec_rate, scale_rate)

                obs_amounts = [t.amount for t in accumulated_txs] if accumulated_txs else [profile.base_mean_amount]
                obs_mean_amt = float(np.mean(obs_amounts))
                expected_amt = profile.base_mean_amount
                scale_amt = max(1.0, profile.base_std_amount)
                m_amt = compute_standardized_magnitude(obs_mean_amt, expected_amt, scale_amt)

                obs_dev_ratio = (len({t.device_id for t in accumulated_txs}) / max(1, len(accumulated_txs))) if accumulated_txs else profile.expected_device_ratio
                m_dev = compute_standardized_magnitude(obs_dev_ratio, profile.expected_device_ratio, profile.robust_scale_device_ratio)

                high_risk_count = len([t for t in accumulated_txs if t.country == "HIGH_RISK_GEO" or t.payment_method == "PREPAID_CARD"]) if accumulated_txs else 0
                obs_attr_ratio = (high_risk_count / max(1, len(accumulated_txs))) if accumulated_txs else profile.expected_high_risk_country_ratio
                m_attr = compute_standardized_magnitude(obs_attr_ratio, profile.expected_high_risk_country_ratio, profile.robust_scale_country_ratio)

                atype = spec.anomaly_type
                if atype == "compound_anomaly":
                    realized_m = compute_compound_severity([m_rate, m_amt, m_dev])
                elif atype in ("velocity_spike", "volume_spike", "sustained_anomaly"):
                    realized_m = m_rate
                elif atype == "amount_spike":
                    realized_m = m_amt
                elif atype == "behavioral_shift":
                    realized_m = m_dev
                elif atype == "attribute_anomaly":
                    realized_m = m_attr
                else:
                    realized_m = spec.target_magnitude

                realized_gt = create_ground_truth_event(eid, m_id, spec, realized_magnitude=realized_m)

                if eid not in emitted_events:
                    active_gt_events.append(realized_gt)
                    emitted_events.add(eid)

        # Advance virtual clock
        self.clock.advance(duration_minutes * 60.0)

        all_txs.sort(key=lambda x: x.timestamp)
        return all_txs, active_gt_events
