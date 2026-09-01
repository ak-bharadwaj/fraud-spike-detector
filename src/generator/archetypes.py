"""Merchant archetypes and legitimate traffic generation.

Nine required merchant archetypes (M1 through M9):
1. M1 (small / low_volume / sparse): low transaction frequency with extended inter-arrival gaps (0.2 <= rate <= 0.8).
2. M2 (medium / stable): steady transaction volume, low variance (8.0 <= rate <= 15.0).
3. M3 (high_volume / large): high transaction throughput (30.0 <= rate <= 60.0).
4. M4 (seasonal): diurnal (time-of-day) and weekly cyclic patterns.
5. M5 (weekend_heavy): significantly higher transaction rates on weekend days (Saturday/Sunday).
6. M6 (highly_variable / volatile): high transaction rate variance and dispersion.
7. M7 (night_heavy): peak transaction activity concentrated during night hours (22:00-05:00 UTC).
8. M8 (mixed): combination of seasonality, growth, and moderate variance.
9. M9 (growing / organic_growth): progressive organic baseline growth over time.

All RNG behavior uses deterministic per-merchant Generators derived from (global_seed, merchant_id).
Legitimate baselines for device, customer, country, and payment method are mathematically derived from sampling probabilities.
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
    p_high_risk_country: float = 0.02
    p_prepaid_payment: float = 0.05
    p_debit_payment: float = 0.15
    legit_device_pool_size: int = 5000
    legit_customer_pool_size: int = 5000


def create_merchant_profile(global_seed: int, merchant_id: str, archetype: str) -> MerchantProfile:
    """Create a deterministic merchant profile based on archetype and sub-seed."""
    from src.generator.rng import get_merchant_rng
    rng = get_merchant_rng(global_seed, f"{merchant_id}_profile")

    arch = archetype.lower().strip()
    if arch in ("sparse", "low_volume", "small", "m1"):
        rate = float(rng.uniform(0.2, 0.8))
        mean_amt = float(rng.uniform(15.0, 40.0))
        std_amt = float(mean_amt * 0.2)
        canonical_arch = "sparse"
    elif arch in ("stable", "medium", "m2"):
        rate = float(rng.uniform(8.0, 15.0))
        mean_amt = float(rng.uniform(40.0, 80.0))
        std_amt = float(mean_amt * 0.15)
        canonical_arch = "stable"
    elif arch in ("high_volume", "high", "large", "m3"):
        rate = float(rng.uniform(30.0, 60.0))
        mean_amt = float(rng.uniform(50.0, 100.0))
        std_amt = float(mean_amt * 0.2)
        canonical_arch = "high_volume"
    elif arch in ("seasonal", "m4"):
        rate = float(rng.uniform(10.0, 20.0))
        mean_amt = float(rng.uniform(50.0, 100.0))
        std_amt = float(mean_amt * 0.25)
        canonical_arch = "seasonal"
    elif arch in ("weekend_heavy", "weekend", "m5"):
        rate = float(rng.uniform(8.0, 16.0))
        mean_amt = float(rng.uniform(40.0, 90.0))
        std_amt = float(mean_amt * 0.2)
        canonical_arch = "weekend_heavy"
    elif arch in ("volatile", "highly_variable", "m6"):
        rate = float(rng.uniform(12.0, 25.0))
        mean_amt = float(rng.uniform(60.0, 150.0))
        std_amt = float(mean_amt * 0.6)
        canonical_arch = "volatile"
    elif arch in ("night_heavy", "night", "m7"):
        rate = float(rng.uniform(6.0, 14.0))
        mean_amt = float(rng.uniform(35.0, 85.0))
        std_amt = float(mean_amt * 0.25)
        canonical_arch = "night_heavy"
    elif arch in ("mixed", "m8"):
        rate = float(rng.uniform(8.0, 18.0))
        mean_amt = float(rng.uniform(45.0, 110.0))
        std_amt = float(mean_amt * 0.3)
        canonical_arch = "mixed"
    elif arch in ("growing", "organic_growth", "growth", "m9"):
        rate = float(rng.uniform(5.0, 10.0))
        mean_amt = float(rng.uniform(30.0, 70.0))
        std_amt = float(mean_amt * 0.2)
        canonical_arch = "growing"
    else:
        rate = 10.0
        mean_amt = 50.0
        std_amt = 10.0
        canonical_arch = arch

    return MerchantProfile(
        merchant_id=merchant_id,
        archetype=canonical_arch,
        base_rate_per_min=rate,
        base_mean_amount=mean_amt,
        base_std_amount=std_amt,
    )


def compute_expected_device_ratio(sample_size: int, pool_size: int) -> float:
    """Compute exact expected unique device ratio for N transactions sampled from pool size P."""
    if sample_size <= 0:
        return 1.0
    p = max(1, pool_size)
    n = sample_size
    expected_unique = p * (1.0 - math.pow(1.0 - 1.0 / p, n))
    return expected_unique / n


def compute_robust_scale_device_ratio(expected_ratio: float) -> float:
    """Compute robust scale for device ratio."""
    return max(0.02, 0.15 * expected_ratio)


def compute_expected_country_ratio(p_high_risk: float) -> float:
    """Compute expected legitimate high-risk country ratio."""
    return p_high_risk


def compute_robust_scale_country_ratio(p_high_risk: float, sample_size: int) -> float:
    """Compute robust scale for high-risk country ratio using Binomial standard error."""
    n = max(1, sample_size)
    se = math.sqrt(p_high_risk * (1.0 - p_high_risk) / n)
    return max(0.01, se)


def compute_expected_payment_ratio(p_prepaid: float) -> float:
    """Compute expected legitimate prepaid card ratio."""
    return p_prepaid


def compute_robust_scale_payment_ratio(p_prepaid: float, sample_size: int) -> float:
    """Compute robust scale for prepaid payment ratio using Binomial standard error."""
    n = max(1, sample_size)
    se = math.sqrt(p_prepaid * (1.0 - p_prepaid) / n)
    return max(0.01, se)


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
    day_of_week = dt_utc.weekday()
    weekly_mult = 1.2 if day_of_week in (5, 6) else 0.95

    # 3. Organic growth
    elapsed_days = (dt_utc - start_utc).total_seconds() / 86400.0
    growth_mult = 1.0 + 0.02 * elapsed_days

    arch = profile.archetype
    if arch == "stable":
        effective_rate = rate
    elif arch == "high_volume":
        effective_rate = rate
    elif arch == "sparse":
        effective_rate = rate
    elif arch == "seasonal":
        effective_rate = rate * diurnal_mult * weekly_mult
    elif arch == "weekend_heavy":
        # Saturday (5) & Sunday (6) have 3x traffic compared to weekdays
        weekend_factor = 2.5 if day_of_week in (5, 6) else 0.6
        effective_rate = rate * weekend_factor
    elif arch == "volatile":
        noise = 1.0 + 0.3 * math.sin(2.0 * math.pi * elapsed_days * 3.0)
        effective_rate = rate * noise
    elif arch == "night_heavy":
        # Night peak 22:00 to 05:00 UTC
        if hour >= 22.0 or hour < 5.0:
            night_factor = 2.4
        else:
            night_factor = 0.5
        effective_rate = rate * night_factor
    elif arch == "growing":
        effective_rate = rate * growth_mult
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
