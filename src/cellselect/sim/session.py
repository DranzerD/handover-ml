"""Data-session reconnection with exponential back-off (component: session.reconnect)."""

from __future__ import annotations

from dataclasses import dataclass

from cellselect.sim.context import SimContext

BACKOFF_BASE_S = 0.2
BACKOFF_CAP_S = 8.0
DETECTION_DELAY_S = 0.15


@dataclass
class OutageResult:
    recovery_times_s: list[float]

    @property
    def max_recovery_s(self) -> float:
        return max(self.recovery_times_s, default=0.0)


def ideal_recovery_s(outage_s: float) -> float:
    """Recovery time of a correct back-off implementation for a single outage (reference oracle)."""
    t, attempt = DETECTION_DELAY_S, 0
    while t < outage_s:
        t += min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S)
        attempt += 1
    return t - outage_s


def simulate_outages(ctx: SimContext, outage_s: float, count: int, gap_s: float = 20.0) -> OutageResult:
    """Network becomes unavailable `count` times; the UE retries with exponential back-off."""
    attempt = 0
    recoveries: list[float] = []
    for _ in range(count):
        duration = outage_s * float(ctx.rng.uniform(0.95, 1.05))
        if not ctx.has("F_BACKOFF"):
            attempt = 0
        t = DETECTION_DELAY_S
        while t < duration:
            t += min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S)
            attempt += 1
        setup = 0.06 + float(ctx.rng.exponential(0.02))
        recoveries.append(t - duration + setup)
        _ = gap_s
    return OutageResult(recoveries)
