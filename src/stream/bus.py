"""TimeOrderedEventBus module for deterministic event-driven transaction streaming.

Key Invariants:
- Deterministic chronological ordering by (timestamp, merchant_id, transaction_id).
- Deterministic tie-breaking across events sharing identical timestamps.
- Transaction-only detector-facing payload (zero ground-truth leakage).
- VirtualClock remains sole detector time authority; clock advances monotonically as stream drains.
- Replay determinism: Replaying identical stream yields identical ordered events.
- Merchant compositionality: Adding a new merchant does not alter existing merchant event ordering.
- Zero wall-clock calls (no datetime.now() or time.time()).
"""

from typing import List, Optional, Callable, Sequence
from datetime import datetime

from src.contracts.contracts import Transaction
from src.stream.clock import VirtualClock


class TimeOrderedEventBus:
    """Bus for queuing, sorting, and dispatching transaction events strictly ordered by timestamp."""

    def __init__(self, clock: Optional[VirtualClock] = None):
        """Initialize bus with VirtualClock time authority."""
        self.clock = clock or VirtualClock()
        self._events: List[Transaction] = []

    def publish(self, transaction: Transaction) -> None:
        """Publish a single transaction event onto the bus."""
        if not isinstance(transaction, Transaction):
            raise TypeError(
                f"TimeOrderedEventBus accepts Transaction instances only, got {type(transaction).__name__}"
            )
        self._events.append(transaction)

    def publish_batch(self, transactions: Sequence[Transaction]) -> None:
        """Publish a batch of transactions onto the bus."""
        for tx in transactions:
            self.publish(tx)

    def get_ordered_events(self) -> List[Transaction]:
        """Return all queued transactions sorted deterministically by (timestamp, merchant_id, transaction_id)."""
        return sorted(
            self._events,
            key=lambda t: (t.timestamp, t.merchant_id, t.transaction_id),
        )

    def drain(self, handler: Optional[Callable[[Transaction], None]] = None) -> List[Transaction]:
        """Sort queued events, advance clock to event timestamp, and dispatch to handler in order."""
        ordered = self.get_ordered_events()
        self._events.clear()

        for tx in ordered:
            self.clock.set_time(tx.timestamp)
            if handler is not None:
                handler(tx)

        return ordered

    def clear(self) -> None:
        """Clear all queued events from the bus."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)
