"""Transport-level models: latency and buffering (components: transport.latency, transport.buffer)."""

from __future__ import annotations

import numpy as np

from cellselect.sim.context import SimContext

CORE_LATENCY_MS = 8.0


def latency_samples_ms(
    ctx: SimContext,
    load: float,
    expected_tx: np.ndarray,
    slot_ms: float,
    harq_rtt_ms: float,
    n_packets: int,
) -> np.ndarray:
    """One-way user-plane latency samples: core + air + HARQ retransmissions + M/M/1 queueing."""
    if not 0.0 <= load < 1.0:
        raise ValueError("load must be in [0, 1)")
    tx = ctx.rng.choice(expected_tx, size=n_packets)
    retx = ctx.rng.poisson(np.maximum(tx - 1.0, 0.0))
    service_ms = 2.0 * slot_ms
    mean_wait = service_ms * load / (1.0 - load)
    if ctx.has("F_LATENCY_QUEUE") and load > 0.7:
        mean_wait *= 3.0
    queue = ctx.rng.exponential(mean_wait, size=n_packets) if mean_wait > 0 else np.zeros(n_packets)
    jitter = ctx.rng.normal(0.0, 0.3, size=n_packets)
    return CORE_LATENCY_MS + slot_ms + retx * harq_rtt_ms + queue + np.abs(jitter)


def buffer_capacity_packets(ctx: SimContext) -> int:
    return 128 if ctx.has("F_BUFFER_SIZE") else 512


def simulate_bursty_queue(
    ctx: SimContext,
    n_steps: int,
    burst_packets: int,
    burst_period_steps: int,
    service_per_step: float,
) -> tuple[int, int]:
    """Discrete-time queue with periodic bursts; returns (arrived, dropped) packet counts."""
    capacity = buffer_capacity_packets(ctx)
    queue = 0.0
    arrived = dropped = 0
    phase = int(ctx.rng.integers(0, burst_period_steps))
    for t in range(n_steps):
        if (t + phase) % burst_period_steps == 0:
            n = int(ctx.rng.poisson(burst_packets))
            arrived += n
            space = capacity - queue
            accepted = min(n, max(int(space), 0))
            dropped += n - accepted
            queue += accepted
        queue = max(0.0, queue - service_per_step * ctx.rng.uniform(0.8, 1.2))
    return arrived, dropped


def memory_growth_mb(ctx: SimContext, duration_s: float, sessions_per_hour: float) -> float:
    """Resident-memory growth of the bearer manager over a long run."""
    hours = duration_s / 3600.0
    sessions = ctx.rng.poisson(sessions_per_hour * hours)
    growth = abs(ctx.rng.normal(0.4, 0.3)) * hours  # allocator fragmentation, benign
    if ctx.has("F_MEMORY_LEAK"):
        growth += 0.03 * sessions
    return float(growth)
