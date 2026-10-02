# Copyright 2026 Jan Kovář
# Copyright 2026 Acmark s.r.o.
# SPDX-License-Identifier: Apache-2.0
"""Per-tenant isolation for CRM tool calls.

~30 Coripo instances share one gateway worker pool. A slow or dead CRM must not eat the
shared HTTP pool or make every turn of that tenant wait for timeouts, and it must never
affect other tenants. Each tenant gets its own concurrency limit and circuit breaker.
State is per worker process (no coordination needed at this scale).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class CircuitOpenError(Exception):
    """The tenant's CRM failed repeatedly; calls fail fast until the cool-down ends."""


@dataclass
class CircuitBreaker:
    """closed → (N consecutive transport failures) → open → (cool-down) → half-open:
    one trial call; success closes, failure re-opens."""

    failure_threshold: int = 5
    reset_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    _trial_in_flight: bool = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.reset_seconds:
            return "half_open"
        return "open"

    def before_call(self) -> None:
        state = self.state
        if state == "open" or (state == "half_open" and self._trial_in_flight):
            raise CircuitOpenError(f"CRM unavailable (circuit {state})")
        if state == "half_open":
            self._trial_in_flight = True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self._trial_in_flight = False
        self.failures += 1
        if self.opened_at is not None or self.failures >= self.failure_threshold:
            self.opened_at = self.clock()

    def release_trial(self) -> None:
        """A half-open trial ended without a transport verdict (e.g. cancelled)."""
        self._trial_in_flight = False


@dataclass
class TenantGuard:
    semaphore: asyncio.Semaphore
    breaker: CircuitBreaker


@dataclass
class TenantGuards:
    max_concurrency: int = 8
    failure_threshold: int = 5
    reset_seconds: float = 30.0
    _guards: dict[str, TenantGuard] = field(default_factory=dict)

    def get(self, tenant_id: str) -> TenantGuard:
        guard = self._guards.get(tenant_id)
        if guard is None:
            guard = TenantGuard(
                semaphore=asyncio.Semaphore(self.max_concurrency),
                breaker=CircuitBreaker(failure_threshold=self.failure_threshold, reset_seconds=self.reset_seconds),
            )
            self._guards[tenant_id] = guard
        return guard

    def state(self, tenant_id: str) -> dict[str, object]:
        guard = self._guards.get(tenant_id)
        if guard is None:
            return {"breaker": "closed", "consecutive_failures": 0}
        return {"breaker": guard.breaker.state, "consecutive_failures": guard.breaker.failures}
