"""Generator package exports."""

from src.generator.rng import (
    get_merchant_seed,
    get_merchant_rng,
    seed_python_random,
)
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
from src.generator.stream_generator import (
    SyntheticStreamGenerator,
    OverlapAnomalyError,
)

__all__ = [
    "get_merchant_seed",
    "get_merchant_rng",
    "seed_python_random",
    "MerchantProfile",
    "create_merchant_profile",
    "compute_legitimate_rate",
    "sample_legitimate_amount",
    "AnomalySpec",
    "create_ground_truth_event",
    "compute_standardized_magnitude",
    "compute_compound_severity",
    "SyntheticStreamGenerator",
    "OverlapAnomalyError",
]
