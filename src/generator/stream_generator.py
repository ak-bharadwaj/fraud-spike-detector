"""SyntheticStreamGenerator implementation.

Generates deterministic transaction streams and ground-truth events using VirtualClock and RNG sub-seeds.

Key Invariants:
- Uses VirtualClock for time management.
- Enforces No Overlapping Active Events per merchant.
- Derives realized ground-truth magnitude from actual generated transaction stream statistics over the anomaly's exact temporal interval [start_time, end_time).
- Guarantees exact window-partitioning identity (sev1 == sev2) via minute-indexed RNG states.
- Completely isolated from detector code.
"""

from datetime import datetime, timedelta, timezone
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
        self.active_events: dict[str, list[GroundTruthEvent]] = {}

        for cfg in merchant_configs:
            m_id = cfg["id"]
            arch = cfg.get("archetype", "stable")
            self.profiles[m_id] = create_merchant_profile(global_seed, m_id, arch)
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

        eid = event_id or f"EVT-{m_id}-{int(st.timestamp())}"
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

        num_minute_steps = int(np.round(duration_minutes))
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
                        override_country = params.get("country", "HIGH_RISK_GEO")
                        override_payment = params.get("payment_method", "PREPAID_CARD")
                    elif atype == "compound_anomaly":
                        rate_multiplier *= params.get("rate_multiplier", max(2.5, tm * 0.8))
                        amount_multiplier *= params.get("amount_multiplier", max(3.0, tm * 0.8))
                        is_behavioral_spike = True
                        override_country = params.get("country", "HIGH_RISK_GEO")

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
                    dev_id = f"DEV-{rng.integers(1, dev_pool_size)}"
                    cust_id = f"CUST-{rng.integers(1, dev_pool_size)}"

                    country = override_country or ("US" if rng.random() > profile.p_high_risk_country else "HIGH_RISK_GEO")
                    payment = override_payment or ("CREDIT_CARD" if rng.random() > profile.p_prepaid_payment else "PREPAID_CARD")

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

                # Record transactions for scheduled anomalies during step
                for eid, spec in current_specs:
                    st = spec.start_time if spec.start_time.tzinfo else spec.start_time.replace(tzinfo=timezone.utc)
                    et = spec.end_time if spec.end_time.tzinfo else spec.end_time.replace(tzinfo=timezone.utc)

                    txs_in_spec = [t for t in step_txs if st <= t.timestamp < et]
                    self.anomaly_tx_history[eid].extend(txs_in_spec)

                    accumulated_txs = self.anomaly_tx_history[eid]
                    total_duration_min = max(0.1, spec.duration_seconds / 60.0)

                    expected_spec_rate = compute_legitimate_rate(
                        profile=profile,
                        current_time=st,
                        simulation_start=self.simulation_start,
                        is_surge_active=is_surge_active.get(m_id, False),
                    )

                    elapsed_min = min(total_duration_min, max(0.1, (step_end - st).total_seconds() / 60.0))
                    obs_rate = len(accumulated_txs) / elapsed_min
                    scale_rate = max(0.5, 0.2 * expected_spec_rate)
                    m_rate = compute_standardized_magnitude(obs_rate, expected_spec_rate, scale_rate)

                    obs_amounts = [t.amount for t in accumulated_txs] if accumulated_txs else [profile.base_mean_amount]
                    obs_mean_amt = float(np.mean(obs_amounts))
                    expected_amt = profile.base_mean_amount
                    scale_amt = max(1.0, profile.base_std_amount)
                    m_amt = compute_standardized_magnitude(obs_mean_amt, expected_amt, scale_amt)

                    # Device ratio deviation derived from legitimate occupancy baseline
                    n_txs = max(1, len(accumulated_txs))
                    expected_dev_ratio = compute_expected_device_ratio(n_txs, profile.legit_device_pool_size)
                    scale_dev_ratio = compute_robust_scale_device_ratio(expected_dev_ratio)
                    obs_dev_ratio = len({t.device_id for t in accumulated_txs}) / n_txs
                    m_dev = compute_standardized_magnitude(obs_dev_ratio, expected_dev_ratio, scale_dev_ratio)

                    # Country ratio deviation
                    high_risk_country_count = len([t for t in accumulated_txs if t.country == "HIGH_RISK_GEO"])
                    obs_country_ratio = high_risk_country_count / n_txs
                    expected_country_ratio = compute_expected_country_ratio(profile.p_high_risk_country)
                    scale_country_ratio = compute_robust_scale_country_ratio(profile.p_high_risk_country, n_txs)
                    m_country = compute_standardized_magnitude(obs_country_ratio, expected_country_ratio, scale_country_ratio)

                    # Payment method ratio deviation
                    prepaid_count = len([t for t in accumulated_txs if t.payment_method == "PREPAID_CARD"])
                    obs_payment_ratio = prepaid_count / n_txs
                    expected_payment_ratio = compute_expected_payment_ratio(profile.p_prepaid_payment)
                    scale_payment_ratio = compute_robust_scale_payment_ratio(profile.p_prepaid_payment, n_txs)
                    m_payment = compute_standardized_magnitude(obs_payment_ratio, expected_payment_ratio, scale_payment_ratio)

                    atype = spec.anomaly_type
                    if atype == "compound_anomaly":
                        # Section 14 Compound Severity Rule: mean absolute standardized deviation across active signals
                        active_signals = [m_rate, m_amt, m_dev, m_country]
                        realized_m = compute_compound_severity(active_signals)
                    elif atype in ("velocity_spike", "volume_spike", "sustained_anomaly"):
                        realized_m = m_rate
                    elif atype == "amount_spike":
                        realized_m = m_amt
                    elif atype == "behavioral_shift":
                        realized_m = m_dev
                    elif atype == "attribute_anomaly":
                        realized_m = compute_compound_severity([m_country, m_payment])
                    else:
                        realized_m = spec.target_magnitude

                    realized_gt = create_ground_truth_event(eid, m_id, spec, realized_magnitude=realized_m)

                    # Update active_events list for merchant with realized_gt
                    for idx, ex_gt in enumerate(self.active_events[m_id]):
                        if ex_gt.event_id == eid:
                            self.active_events[m_id][idx] = realized_gt

        # Collect active ground-truth events for the window
        active_gt_events: list[GroundTruthEvent] = []
        for m_id in self.profiles:
            for gt in self.active_events[m_id]:
                if max(window_start, gt.start_time) < min(window_end, gt.end_time):
                    if gt.event_id not in emitted_events:
                        active_gt_events.append(gt)
                        emitted_events.add(gt.event_id)

        # Advance virtual clock
        self.clock.advance(duration_minutes * 60.0)

        all_txs.sort(key=lambda x: x.timestamp)
        return all_txs, active_gt_events
