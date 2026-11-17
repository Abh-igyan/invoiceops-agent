"""Configurable fault injection (timeouts, HTTP 429s, stale reads) for the mock ERP."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

FaultDecision = Literal["ok", "timeout", "rate_limit_429", "stale_read"]


@dataclass
class FlakySettings:
    """Runtime fault-injection probabilities and simulation parameters."""

    enabled: bool = False
    timeout_rate: float = 0.12
    rate_limit_429_rate: float = 0.15
    stale_read_rate: float = 0.10
    timeout_sleep_seconds: float = 6.0
    seed: int = 42


@dataclass
class FlakyInjector:
    """Seeded fault injector with deterministic queue override for unit tests."""

    settings: FlakySettings = field(default_factory=FlakySettings)
    _rng: random.Random = field(init=False)
    _scripted: deque[FaultDecision] = field(default_factory=deque)
    total_calls: int = 0
    injected_timeouts: int = 0
    injected_429s: int = 0
    injected_stale_reads: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.settings.seed)

    def enqueue_scripted(self, *decisions: FaultDecision) -> None:
        """Queue deterministic fault outcomes for unit testing."""
        self._scripted.extend(decisions)

    def next_http_fault(self) -> FaultDecision:
        """Sample transport-level fault (`timeout` or `rate_limit_429`) before tool dispatch."""
        self.total_calls += 1
        if self._scripted and self._scripted[0] in {"timeout", "rate_limit_429", "ok"}:
            decision = self._scripted.popleft()
            self._record(decision)
            return decision

        if not self.settings.enabled:
            return "ok"

        roll = self._rng.random()
        if roll < self.settings.timeout_rate:
            self._record("timeout")
            return "timeout"
        if roll < self.settings.timeout_rate + self.settings.rate_limit_429_rate:
            self._record("rate_limit_429")
            return "rate_limit_429"
        return "ok"

    def should_return_stale_read(self) -> bool:
        """Sample application-level stale replica read for read-only lookups."""
        if self._scripted and self._scripted[0] == "stale_read":
            self._scripted.popleft()
            self._record("stale_read")
            return True
        if not self.settings.enabled:
            return False
        if self._rng.random() < self.settings.stale_read_rate:
            self._record("stale_read")
            return True
        return False

    def _record(self, decision: FaultDecision) -> None:
        if decision == "timeout":
            self.injected_timeouts += 1
        elif decision == "rate_limit_429":
            self.injected_429s += 1
        elif decision == "stale_read":
            self.injected_stale_reads += 1

    def stats(self) -> dict[str, int | bool]:
        """Return fault-injection counters for observability and assertions."""
        return {
            "enabled": self.settings.enabled,
            "total_calls": self.total_calls,
            "injected_timeouts": self.injected_timeouts,
            "injected_429s": self.injected_429s,
            "injected_stale_reads": self.injected_stale_reads,
        }
