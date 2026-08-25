"""SyntheticStreamGenerator implementation.

Generates deterministic transaction streams and ground-truth events using VirtualClock and RNG sub-seeds.

Key Invariants:
- Uses VirtualClock for time management.
- Enforces No Overlapping Active Events per merchant.
- Derives realized ground-truth magnitude from actual generated transaction stream statistics over the anomaly's exact temporal interval [start_time, end_time).
- Emits finalized GroundTruthEvents ONLY after the complete [start_time, end_time) interval has finished.
- Validates Option A whole-minute anomaly durations (duration_seconds % 60 == 0).
- Guarantees exact window-partitioning identity via minute-indexed RNG states.
- 128-bit SHA-256 deterministic collision-resistant Event IDs.
- Customer population independent of device population.
- Full 3-tier legitimate payment distribution (credit/debit/prepaid).
- Completely isolated from detector code.
"""

from datetime import datetime, timedelta, timezone
import hashlib
from typing import Any, Optional, Sequence
import math
import uuid
import numpy as np

from src.contracts.contracts import Transaction, GroundTruthEvent
from src.generator.archetypes import (
    MerchantProfile,
    create_merchant_profile,
    compute_legitimate_rate,
    sample_legitimate_amount,
    compute_expected_device_ratio,
    compute_robust_scale_device_ratio,
    compute_expected_country_ratio,
    compute_robust_scale_country_ratio,
    compute_expected_payment_ratio,
    compute_robust_scale_payment_ratio,
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
        self.scheduled_specs: dict[str, list[tuple[str, AnomalySpec]]] = {}
        self.anomaly_tx_history: dict[str, list[Transaction]] = {}
        self.finalized_events: dict[str, list[GroundTruthEvent]] = {}
        self.completed_eids: set[str] = set()

        for cfg in merchant_configs:
            m_id = cfg["id"]
            arch = cfg.get("archetype", "stable")
            self.profiles[m_id] = create_merchant_profile(global_seed, m_id, arch)
            self.scheduled_specs[m_id] = []
            self.finalized_events[m_id] = []

    def schedule_anomaly(
        self,
        merchant_id: str,
        spec: AnomalySpec,
        event_id: Optional[str] = None,
    ) -> GroundTruthEvent:
        """Schedule an anomaly for a merchant, enforcing the No Overlapping Active Events invariant."""
        if merchant_id not in self.profiles:
            raise KeyError(f"Merchant '{merchant_id}' not configured in generator.")

        if spec.duration_seconds <= 0 or float(spec.duration_seconds) % 60.0 != 0.0:
            raise ValueError(
                f"Anomaly duration_seconds must be a positive whole number of minutes (multiple of 60s), got {spec.duration_seconds}"
            )

        st = spec.start_time if spec.start_time.tzinfo else spec.start_time.replace(tzinfo=timezone.utc)
        et = spec.end_time if spec.end_time.tzinfo else spec.end_time.replace(tzinfo=timezone.utc)

        # Validate attribute anomaly specification
        if spec.anomaly_type == "attribute_anomaly":
            valid_keys = {"country", "payment_method"}
            if not spec.parameters:
                raise ValueError(
                    f"attribute_anomaly spec for merchant '{merchant_id}' requires at least one supported attribute parameter ('country', 'payment_method')."
                )
            for k in spec.parameters:
                if k not in valid_keys:
                    raise ValueError(f"Unsupported attribute parameter '{k}' for attribute_anomaly.")

        # Enforce No Overlapping Active Events invariant per merchant
        for _, existing_spec in self.scheduled_specs[merchant_id]:
            ex_st = existing_spec.start_time if existing_spec.start_time.tzinfo else existing_spec.start_time.replace(tzinfo=timezone.utc)
            ex_et = existing_spec.end_time if existing_spec.end_time.tzinfo else existing_spec.end_time.replace(tzinfo=timezone.utc)

            if max(st, ex_st) < min(et, ex_et):
                raise OverlapAnomalyError(
                    f"Anomaly overlap rejected for merchant '{merchant_id}': "
                    f"New spec [{st} .. {et}] overlaps existing spec [{ex_st} .. {ex_et}]."
                )

        spec_key = f"{self.global_seed}:{merchant_id}:{spec.anomaly_type}:{st.isoformat()}:{et.isoformat()}:{sorted(spec.parameters.items())}"
        hash_hex = hashlib.sha256(spec_key.encode("utf-8")).hexdigest()[:32]
        eid = event_id or f"EVT-{merchant_id}-{hash_hex}"

        self.scheduled_specs[merchant_id].append((eid, spec))
        self.anomaly_tx_history[eid] = []

        # Return placeholder specification event
        return create_ground_truth_event(eid, merchant_id, spec, realized_magnitude=spec.target_magnitude)

    def generate_window(
        self,
        duration_minutes: float = 1.0,
        is_surge_active: dict[str, bool] = None,
    ) -> tuple[list[Transaction], list[GroundTruthEvent]]:
        """Generate transactions and realized ground-truth events for the current time step."""
        if duration_minutes <= 0 or float(duration_minutes) != float(int(duration_minutes)):
            raise ValueError(f"duration_minutes must be a positive integer, got {duration_minutes}")

        is_surge_active = is_surge_active or {}
        window_start = self.clock.current_time()
        window_end = window_start + timedelta(minutes=duration_minutes)

        all_txs: list[Transaction] = []
        emitted_events: list[GroundTruthEvent] = []

        num_minute_steps = int(duration_minutes)
        for step in range(num_minute_steps):
            step_start = window_start + timedelta(minutes=step)
            step_end = step_start + timedelta(minutes=1.0)
            elapsed_minute_index = int((step_start - self.simulation_start).total_seconds() // 60)

            for m_id, profile in self.profiles.items():
                rng = get_merchant_rng(self.global_seed, f"{m_id}_min_{elapsed_minute_index}")

                current_specs = [
                    (eid, spec) for eid, spec in self.scheduled_specs[m_id]
                    if max(step_start, spec.start_time) < min(step_end, spec.end_time)
                ]

                legit_rate = compute_legitimate_rate(
                    profile=profile,
                    current_time=step_start,
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
                        if "country" in params:
                            override_country = params["country"]
                        if "payment_method" in params:
                            override_payment = params["payment_method"]
                    elif atype == "compound_anomaly":
                        rate_multiplier *= params.get("rate_multiplier", max(2.5, tm * 0.8))
                        amount_multiplier *= params.get("amount_multiplier", max(3.0, tm * 0.8))
                        is_behavioral_spike = True
                        if "country" in params:
                            override_country = params["country"]
                        if "payment_method" in params:
                            override_payment = params["payment_method"]

                effective_rate = legit_rate * rate_multiplier
                expected_count = int(np.round(effective_rate * 1.0))
                tx_count = max(0, int(rng.poisson(lam=max(0.1, expected_count))))

                step_txs: list[Transaction] = []
                for i in range(tx_count):
                    offset_sec = float(rng.uniform(0.0, 60.0))
                    tx_time = step_start + timedelta(seconds=offset_sec)

                    base_amt = sample_legitimate_amount(profile, rng)
                    final_amt = round(max(1.0, base_amt * amount_multiplier), 2)

                    tx_id = f"TX-{m_id}-{uuid.UUID(bytes=rng.bytes(16)).hex[:10]}"
                    dev_pool_size = 5 if is_behavioral_spike else profile.legit_device_pool_size
                    cust_pool_size = profile.legit_customer_pool_size
                    dev_id = f"DEV-{rng.integers(1, dev_pool_size + 1)}"
                    cust_id = f"CUST-{rng.integers(1, cust_pool_size + 1)}"

                    country = override_country or ("HIGH_RISK_GEO" if rng.random() < profile.p_high_risk_country else "US")

                    r_pay = rng.random()
                    if r_pay < profile.p_prepaid_payment:
                        legit_payment = "PREPAID_CARD"
                    elif r_pay < profile.p_prepaid_payment + profile.p_debit_payment:
                        legit_payment = "DEBIT_CARD"
                    else:
                        legit_payment = "CREDIT_CARD"

                    payment = override_payment or legit_payment

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
                    step_txs.append(tx)

                all_txs.extend(step_txs)

                # Record transactions and finalize completed events when step_end reaches end_time
                for eid, spec in current_specs:
                    st = spec.start_time if spec.start_time.tzinfo else spec.start_time.replace(tzinfo=timezone.utc)
                    et = spec.end_time if spec.end_time.tzinfo else spec.end_time.replace(tzinfo=timezone.utc)

                    txs_in_spec = [t for t in step_txs if st <= t.timestamp < et]
                    self.anomaly_tx_history[eid].extend(txs_in_spec)

                    # Finalize realized severity ONLY when the complete [start_time, end_time) interval has finished
                    if step_end >= et and eid not in self.completed_eids:
                        accumulated_txs = self.anomaly_tx_history[eid]
                        total_duration_min_int = max(1, int(round(spec.duration_seconds / 60.0)))
                        total_duration_min = max(0.1, spec.duration_seconds / 60.0)

                        # Integrated expected rate over full anomaly duration [start_time, end_time)
                        expected_total_count = sum(
                            compute_legitimate_rate(
                                profile=profile,
                                current_time=st + timedelta(minutes=m),
                                simulation_start=self.simulation_start,
                                is_surge_active=is_surge_active.get(m_id, False),
                            )
                            for m in range(total_duration_min_int)
                        )
                        expected_spec_rate = expected_total_count / total_duration_min

                        obs_rate = len(accumulated_txs) / total_duration_min
                        scale_rate = max(0.5, 0.2 * expected_spec_rate)
                        m_rate = compute_standardized_magnitude(obs_rate, expected_spec_rate, scale_rate)

                        obs_amounts = [t.amount for t in accumulated_txs] if accumulated_txs else [profile.base_mean_amount]
                        obs_mean_amt = float(np.mean(obs_amounts))
                        expected_amt = profile.base_mean_amount
                        scale_amt = max(1.0, profile.base_std_amount)
                        m_amt = compute_standardized_magnitude(obs_mean_amt, expected_amt, scale_amt)

                        n_txs = max(1, len(accumulated_txs))
                        expected_dev_ratio = compute_expected_device_ratio(n_txs, profile.legit_device_pool_size)
                        scale_dev_ratio = compute_robust_scale_device_ratio(expected_dev_ratio)
                        obs_dev_ratio = len({t.device_id for t in accumulated_txs}) / n_txs
                        m_dev = compute_standardized_magnitude(obs_dev_ratio, expected_dev_ratio, scale_dev_ratio)

                        high_risk_country_count = len([t for t in accumulated_txs if t.country == "HIGH_RISK_GEO"])
                        obs_country_ratio = high_risk_country_count / n_txs
                        expected_country_ratio = compute_expected_country_ratio(profile.p_high_risk_country)
                        scale_country_ratio = compute_robust_scale_country_ratio(profile.p_high_risk_country, n_txs)
                        m_country = compute_standardized_magnitude(obs_country_ratio, expected_country_ratio, scale_country_ratio)

                        prepaid_count = len([t for t in accumulated_txs if t.payment_method == "PREPAID_CARD"])
                        obs_payment_ratio = prepaid_count / n_txs
                        expected_payment_ratio = compute_expected_payment_ratio(profile.p_prepaid_payment)
                        scale_payment_ratio = compute_robust_scale_payment_ratio(profile.p_prepaid_payment, n_txs)
                        m_payment = compute_standardized_magnitude(obs_payment_ratio, expected_payment_ratio, scale_payment_ratio)

                        atype = spec.anomaly_type
                        params = spec.parameters

                        if atype == "compound_anomaly":
                            active_signals = [m_rate, m_amt, m_dev]
                            if "country" in params:
                                active_signals.append(m_country)
                            if "payment_method" in params:
                                active_signals.append(m_payment)
                            realized_m = compute_compound_severity(active_signals)
                        elif atype in ("velocity_spike", "volume_spike", "sustained_anomaly"):
                            realized_m = m_rate
                        elif atype == "amount_spike":
                            realized_m = m_amt
                        elif atype == "behavioral_shift":
                            realized_m = m_dev
                        elif atype == "attribute_anomaly":
                            attr_signals = []
                            if "country" in params:
                                attr_signals.append(m_country)
                            if "payment_method" in params:
                                attr_signals.append(m_payment)
                            realized_m = compute_compound_severity(attr_signals or [m_country])
                        else:
                            realized_m = spec.target_magnitude

                        finalized_gt = create_ground_truth_event(eid, m_id, spec, realized_magnitude=realized_m)
                        self.finalized_events[m_id].append(finalized_gt)
                        self.completed_eids.add(eid)
                        emitted_events.append(finalized_gt)

        # Advance virtual clock
        self.clock.advance(duration_minutes * 60.0)

        all_txs.sort(key=lambda x: x.timestamp)
        return all_txs, emitted_events
