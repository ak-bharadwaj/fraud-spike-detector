"""Tests for VirtualClock determinism and RNG sub-seed stability."""

from datetime import datetime, timezone
import pytest

from src.stream.clock import VirtualClock
from src.generator.rng import get_merchant_seed, get_merchant_rng


def test_virtual_clock_operations():
    init_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    clock = VirtualClock(initial_time=init_time)
    assert clock.current_time() == init_time

    new_time = clock.advance(60)
    assert new_time == datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    assert clock.current_time() == new_time

    with pytest.raises(ValueError):
        clock.advance(-10)


def test_deterministic_subseeding():
    global_seed = 42
    seed_m1_a = get_merchant_seed(global_seed, "M1")
    seed_m1_b = get_merchant_seed(global_seed, "M1")
    seed_m2 = get_merchant_seed(global_seed, "M2")

    assert seed_m1_a == seed_m1_b, "Sub-seed for M1 must be deterministic across calls."
    assert seed_m1_a != seed_m2, "Sub-seeds for different merchants must differ."


def test_merchant_rng_isolation():
    global_seed = 100
    rng_m1 = get_merchant_rng(global_seed, "M1")
    vals_m1_a = [rng_m1.random() for _ in range(5)]

    # Interleave creation of M2
    rng_m2 = get_merchant_rng(global_seed, "M2")
    _ = [rng_m2.random() for _ in range(5)]

    # Re-instantiate M1 RNG and verify exact stream matching
    rng_m1_fresh = get_merchant_rng(global_seed, "M1")
    vals_m1_b = [rng_m1_fresh.random() for _ in range(5)]

    assert vals_m1_a == vals_m1_b, "M1 random sequence must be independent of M2 creation."
