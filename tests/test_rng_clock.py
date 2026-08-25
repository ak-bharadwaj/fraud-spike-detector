"""Tests for VirtualClock invariants and RNG compositional reproducibility."""

from datetime import datetime, timezone, timedelta
import pytest

from src.stream.clock import VirtualClock
from src.generator.rng import get_merchant_seed, get_merchant_rng


def test_virtual_clock_initial_state_and_timezone():
    """Test VirtualClock initial state and UTC timezone awareness."""
    clock_default = VirtualClock()
    assert clock_default.current_time().tzinfo == timezone.utc
    assert clock_default.current_time() == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    naive_dt = datetime(2026, 6, 15, 10, 30, 0)
    clock_custom = VirtualClock(initial_time=naive_dt)
    assert clock_custom.current_time().tzinfo == timezone.utc
    assert clock_custom.current_time() == datetime(2026, 6, 15, 10, 30, 0, tzinfo=timezone.utc)


def test_virtual_clock_advancement_and_monotonicity():
    """Test forward advancement, zero advancement, and monotonicity."""
    clock = VirtualClock(initial_time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))

    t0 = clock.current_time()
    t1 = clock.advance(0)
    assert t1 == t0, "Zero advancement should leave clock unchanged."

    t2 = clock.advance(45.5)
    assert t2 > t1, "Clock advance must be strictly monotonic for seconds > 0."
    assert t2 == datetime(2026, 1, 1, 0, 0, 45, 500000, tzinfo=timezone.utc)


def test_virtual_clock_rejection_of_negative_advancement():
    """Test that negative advancement is rejected with ValueError."""
    clock = VirtualClock()
    with pytest.raises(ValueError, match="cannot move backward"):
        clock.advance(-1.0)


def test_virtual_clock_set_time():
    """Test explicit set_time behavior."""
    clock = VirtualClock()
    target = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    clock.set_time(target)
    assert clock.current_time() == target


def test_deterministic_subseeding_stability():
    """Test that per-merchant sub-seeds are deterministic and key-independent."""
    global_seed = 42
    seed_m1 = get_merchant_seed(global_seed, "M1")
    seed_m2 = get_merchant_seed(global_seed, "M2")

    assert seed_m1 == get_merchant_seed(global_seed, "M1")
    assert seed_m2 == get_merchant_seed(global_seed, "M2")
    assert seed_m1 != seed_m2


def test_rng_compositional_reproducibility():
    """Test that merchant_A sequence is identical whether merchant_B exists or not."""
    global_seed = 1000

    # Scenario 1: Generate M1 sequence alone
    rng_m1_alone = get_merchant_rng(global_seed, "M1")
    seq_m1_alone = [rng_m1_alone.normal(0, 1) for _ in range(100)]

    # Scenario 2: Generate M1 sequence after instantiating M2, M3, M4
    _ = get_merchant_rng(global_seed, "M2")
    _ = get_merchant_rng(global_seed, "M3")
    _ = get_merchant_rng(global_seed, "M4")
    rng_m1_after = get_merchant_rng(global_seed, "M1")
    seq_m1_after = [rng_m1_after.normal(0, 1) for _ in range(100)]

    # Verify compositional reproducibility invariant
    assert seq_m1_alone == seq_m1_after, (
        "Compositional reproducibility violation: M1 RNG sequence changed when other merchants were introduced!"
    )
