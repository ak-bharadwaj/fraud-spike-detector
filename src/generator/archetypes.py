"""Merchant archetypes and legitimate traffic generation.

Six required archetypes:
1. stable: steady transaction volume, low variance.
2. seasonal: diurnal (time-of-day) and weekly cyclic patterns.
3. growing: organic baseline growth over time.
4. volatile: high transaction rate and amount variance.
5. sparse: low transaction frequency with extended inter-arrival gaps.
6. mixed: combination of seasonality, growth, and moderate variance.

Legitimate promotional surges generate realistic volume spikes without ground-truth fraud events.
All RNG behavior uses deterministic per-merchant Generators derived from (global_seed, merchant_id).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import numpy as np


@dataclass
class MerchantProfile:
    merchant_id: str
    archetype: str
    base_rate_per_min: float
    base_mean_amount: float
    base_std_amount: float
    expected_device_ratio: float = 0.90
    robust_scale_device_ratio: float = 0.10
    expected_high_risk_country_ratio: float = 0.02
    robust_scale_country_ratio: float = 0.05


def create_merchant_profile(global_seed: int, merchant_id: str, archetype: str) -> MerchantProfile:
    """Create a deterministic merchant profile based on archetype and sub-seed."""
    from src.generator.rng import get_merchant_rng
    rng = get_merchant_rng(global_seed, f"{merchant_id}_profile")

    arch = archetype.lower()
    if arch == "sparse":
        rate = float(rng.uniform(0.2, 0.8))
        mean_amt = float(rng.uniform(15.0, 40.0))
        std_amt = float(mean_amt * 0.2)
    elif arch == "stable":
        rate = float(rng.uniform(8.0, 15.0))
        mean_amt = float(rng.uniform(40.0, 80.0))
        std_amt = float(mean_amt * 0.15)
    elif arch == "seasonal":
        rate = float(rng.uniform(10.0, 20.0))
        mean_amt = float(rng.uniform(50.0, 100.0))
        std_amt = float(mean_amt * 0.25)
    elif arch == "growing":
        rate = float(rng.uniform(5.0, 10.0))
        mean_amt = float(rng.uniform(30.0, 70.0))
        std_amt = float(mean_amt * 0.2)
    elif arch == "volatile":
        rate = float(rng.uniform(12.0, 25.0))
        mean_amt = float(rng.uniform(60.0, 150.0))
        std_amt = float(mean_amt * 0.6)
    elif arch == "mixed":
        rate = float(rng.uniform(8.0, 18.0))
        mean_amt = float(rng.uniform(45.0, 110.0))
        std_amt = float(mean_amt * 0.3)
    else:
        rate = 10.0
        mean_amt = 50.0
        std_amt = 10.0

    return MerchantProfile(
        merchant_id=merchant_id,
        archetype=arch,
        base_rate_per_min=rate,
        base_mean_amount=mean_amt,
        base_std_amount=std_amt,
    )


def compute_legitimate_rate(
    profile: MerchantProfile,
    current_time: datetime,
    simulation_start: datetime,
    is_surge_active: bool = False,
    surge_multiplier: float = 2.5,
) -> float:
    """Compute the expected legitimate transaction rate (tx/min) for a merchant at a specific time."""
    rate = profile.base_rate_per_min
    dt_utc = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
    start_utc = simulation_start if simulation_start.tzinfo else simulation_start.replace(tzinfo=timezone.utc)

    # 1. Diurnal seasonality (time of day)
    hour = dt_utc.hour + dt_utc.minute / 60.0
    diurnal_mult = 1.0 + 0.4 * math.sin(2.0 * math.pi * (hour - 6.0) / 24.0)

    # 2. Weekly seasonality
    day_of_week = dt_utc.weekday()  # 0=Mon, 6=Sun
    weekly_mult = 1.2 if day_of_week in (5, 6) else 0.95

    # 3. Organic growth
    elapsed_days = (dt_utc - start_utc).total_seconds() / 86400.0
    growth_mult = 1.0 + 0.02 * elapsed_days

    arch = profile.archetype
    if arch == "stable":
        effective_rate = rate
    elif arch == "seasonal":
        effective_rate = rate * diurnal_mult * weekly_mult
    elif arch == "growing":
        effective_rate = rate * growth_mult
    elif arch == "volatile":
        noise = 1.0 + 0.3 * math.sin(2.0 * math.pi * elapsed_days * 3.0)
        effective_rate = rate * noise
    elif arch == "sparse":
        effective_rate = rate
    elif arch == "mixed":
        effective_rate = rate * diurnal_mult * growth_mult
    else:
        effective_rate = rate

    if is_surge_active:
        effective_rate *= surge_multiplier

    return max(0.1, float(effective_rate))


def sample_legitimate_amount(profile: MerchantProfile, rng: np.random.Generator) -> float:
    """Sample a legitimate transaction amount based on merchant profile."""
    if profile.archetype == "volatile":
        sigma = 0.5
        mu = math.log(max(1.0, profile.base_mean_amount)) - 0.5 * (sigma ** 2)
        val = rng.lognormal(mean=mu, sigma=sigma)
    else:
        val = rng.normal(loc=profile.base_mean_amount, scale=profile.base_std_amount)

    return max(1.0, round(float(val), 2))
