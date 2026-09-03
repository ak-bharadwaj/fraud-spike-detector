"""AlertStateMachine module for managing alert state transitions and persistence gating.

Key Invariants:
- State model & lifecycle: NORMAL -> CANDIDATE (SUSPICIOUS) -> ALERT -> COOLDOWN -> NORMAL.
- CANDIDATE state is identical to SUSPICIOUS state (§10 intermediate persistence accumulation state).
- ALERT state consistency: when persistence is reached in window w, process_score returns ("ALERT", alert) and get_merchant_state() returns "ALERT".
- COOLDOWN transition: in subsequent windows (w+1..w+5 for cooldown_windows=5), state transitions to "COOLDOWN", suppressing new alerts.
- Normal transition: after cooldown_windows finish, state transitions back to "NORMAL" (at window w+6).
- Persistence (P=2): Requires score >= static_threshold for 2 consecutive qualifying windows.
- Threshold operator: score >= static_threshold (exact >= operator).
- Persistence reset: score < static_threshold or evidence_state == "INSUFFICIENT" resets persistence counter to 0.
- DEGRADED evidence: qualifies for persistence if score >= static_threshold (carries confidence=0.5).
- SUFFICIENT evidence: qualifies for persistence if score >= static_threshold (carries confidence=1.0).
- Merchant isolation: state context is strictly maintained per merchant_id.
- GroundTruth & Holdout isolation: NO imports of ground truth or holdout code.
"""

from datetime import datetime, timezone
import hashlib
from typing import Dict, Optional, Tuple

from src.contracts.contracts import RiskScore, Alert
from src.contracts.config_schemas import DetectorConfig


class MerchantStateContext:
    """Per-merchant state machine tracking context."""

    def __init__(self):
        self.state: str = "NORMAL"  # NORMAL, CANDIDATE, ALERT, COOLDOWN
        self.persistence_counter: int = 0
        self.cooldown_counter: int = 0


class AlertStateMachine:
    """State machine converting RiskScore observations into Alert emissions and state transitions."""

    def __init__(self, persistence: int = 1, cooldown_windows: int = 5, static_threshold: float = 5.0, detector_version: str = "1.1.0"):
        if persistence <= 0:
            raise ValueError(f"persistence must be positive, got {persistence}")
        if cooldown_windows < 0:
            raise ValueError(f"cooldown_windows must be non-negative, got {cooldown_windows}")
        if static_threshold <= 0.0:
            raise ValueError(f"static_threshold must be positive, got {static_threshold}")

        self.persistence = persistence
        self.cooldown_windows = cooldown_windows
        self.static_threshold = float(static_threshold)
        self.detector_version = detector_version

        # Per-merchant state tracking
        self._merchant_states: Dict[str, MerchantStateContext] = {}

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "AlertStateMachine":
        """Construct AlertStateMachine from DetectorConfig."""
        v = getattr(config, "detector_version", getattr(config, "version", "1.1.0"))
        return cls(
            persistence=config.scorer.persistence,
            cooldown_windows=config.state_machine.cooldown_windows,
            static_threshold=config.scorer.static_threshold,
            detector_version=v,
        )

    def process_score(
        self,
        merchant_id: str,
        timestamp: datetime,
        risk_score: RiskScore,
    ) -> Tuple[str, Optional[Alert]]:
        """Process a RiskScore observation for a merchant and return (new_state, Optional[Alert])."""
        if timestamp.tzinfo is None:
            raise TypeError(f"timestamp must be timezone-aware (got naive datetime {timestamp})")

        if merchant_id not in self._merchant_states:
            self._merchant_states[merchant_id] = MerchantStateContext()

        ctx = self._merchant_states[merchant_id]
        score_val = risk_score.score

        # 1. Handle transition from ALERT to COOLDOWN (or NORMAL) in window following ALERT
        if ctx.state == "ALERT":
            if self.cooldown_windows > 0:
                ctx.state = "COOLDOWN"
                ctx.cooldown_counter = self.cooldown_windows
                ctx.persistence_counter = 0
                return ("COOLDOWN", None)
            else:
                ctx.state = "NORMAL"
                ctx.persistence_counter = 0

        # 2. Handle COOLDOWN state progression
        elif ctx.state == "COOLDOWN":
            if score_val is not None and score_val >= self.static_threshold:
                ctx.cooldown_counter = self.cooldown_windows
                return ("COOLDOWN", None)
            else:
                ctx.cooldown_counter -= 1
                if ctx.cooldown_counter <= 0:
                    ctx.state = "NORMAL"
                    ctx.persistence_counter = 0
                    ctx.cooldown_counter = 0
                    return ("NORMAL", None)
                return ("COOLDOWN", None)

        # 3. Check if risk_score is None (INSUFFICIENT evidence) or below static_threshold
        if score_val is None or score_val < self.static_threshold:
            ctx.state = "NORMAL"
            ctx.persistence_counter = 0
            return ("NORMAL", None)

        # 4. Qualifying score breach: score >= static_threshold
        ctx.persistence_counter += 1

        if ctx.persistence_counter < self.persistence:
            ctx.state = "CANDIDATE"
            return ("CANDIDATE", None)

        # 5. Persistence threshold reached (persistence_counter >= P) -> Transition to ALERT
        ctx.state = "ALERT"

        # Generate deterministic alert_id
        ts_iso = timestamp.isoformat()
        sig_str = ":".join(sorted(risk_score.triggered_signals))
        spec_str = f"{merchant_id}:{ts_iso}:{score_val:.4f}:{sig_str}"
        alert_id = f"ALT-{hashlib.sha256(spec_str.encode('utf-8')).hexdigest()[:32]}"

        reason = (
            f"Risk score {score_val:.2f} breached threshold {self.static_threshold:.2f} "
            f"for {self.persistence} consecutive windows"
        )

        alert_obj = Alert(
            alert_id=alert_id,
            merchant_id=merchant_id,
            timestamp=timestamp,
            risk_score=float(score_val),
            confidence=risk_score.confidence,
            reason=reason,
            triggered_signals=risk_score.triggered_signals,
            detector_version=self.detector_version,
        )

        return ("ALERT", alert_obj)

    def get_merchant_state(self, merchant_id: str) -> str:
        """Return the current state name for a merchant."""
        if merchant_id not in self._merchant_states:
            return "NORMAL"
        return self._merchant_states[merchant_id].state

    def reset(self, merchant_id: Optional[str] = None) -> None:
        """Reset state machine context for a specific merchant or all merchants."""
        if merchant_id is not None:
            self._merchant_states.pop(merchant_id, None)
        else:
            self._merchant_states.clear()
