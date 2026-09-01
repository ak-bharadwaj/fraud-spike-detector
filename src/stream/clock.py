"""VirtualClock implementation for deterministic time management.

Rule 10: All detector time behavior uses VirtualClock.current_time().
No datetime.now(), datetime.utcnow(), or time.time() outside clock.py.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional


class VirtualClock:
    """Deterministic virtual clock for streaming execution enforcing monotonic time progression."""

    def __init__(self, initial_time: Optional[datetime] = None):
        if initial_time is None:
            # Default epoch reference for synthetic streams
            self._current_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        else:
            self._current_time = initial_time if initial_time.tzinfo else initial_time.replace(tzinfo=timezone.utc)

    def current_time(self) -> datetime:
        """Return the current virtual time."""
        return self._current_time

    def set_time(self, new_time: datetime) -> None:
        """Set the current virtual time, strictly rejecting backward time movement."""
        dt_utc = new_time if new_time.tzinfo else new_time.replace(tzinfo=timezone.utc)
        if dt_utc < self._current_time:
            raise ValueError(
                f"VirtualClock cannot move backward in time: attempted to set {dt_utc} < current {self._current_time}."
            )
        self._current_time = dt_utc

    def advance(self, seconds: float) -> datetime:
        """Advance the virtual clock by a given number of seconds."""
        if seconds < 0:
            raise ValueError("VirtualClock cannot move backward in time.")
        self._current_time += timedelta(seconds=seconds)
        return self._current_time
