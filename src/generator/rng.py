"""Deterministic random number generator foundation.

Rule 10 & Day 1 Checklist:
- Global seed is supplied once.
- Each merchant receives a deterministic sub-seed derived from (global_seed, merchant_id).
- Adding a merchant/scenario must not reshuffle unrelated merchants.
- Explicitly seed Python random and NumPy RNG.
"""

import hashlib
import random
import numpy as np


def get_merchant_seed(global_seed: int, merchant_id: str) -> int:
    """Derive a deterministic sub-seed for a specific merchant from global_seed and merchant_id."""
    key = f"{global_seed}:{merchant_id}".encode("utf-8")
    hash_bytes = hashlib.sha256(key).digest()
    # Convert first 4 bytes to unsigned 32-bit integer
    sub_seed = int.from_bytes(hash_bytes[:4], byteorder="big", signed=False)
    return sub_seed


def get_merchant_rng(global_seed: int, merchant_id: str) -> np.random.Generator:
    """Get a dedicated NumPy random Generator initialized with merchant-specific sub-seed."""
    sub_seed = get_merchant_seed(global_seed, merchant_id)
    return np.random.default_rng(sub_seed)


def seed_python_random(seed: int) -> None:
    """Seed Python built-in random module explicitly."""
    random.seed(seed)
