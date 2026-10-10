"""MAC scheduler models (component: mac.scheduler)."""

from __future__ import annotations

import numpy as np

from cellselect.sim.context import SimContext
from cellselect.sim.radio import ar1_trace, resource_rate_re_per_s, select_cqi, spectral_efficiency

# Maximum transmission bandwidth configuration (PRBs) per SCS and channel bandwidth (TS 38.101-1 style).
PRB_TABLE: dict[int, dict[int, int]] = {
    15: {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 40: 216, 50: 270},
    30: {10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106, 50: 133, 60: 162, 80: 217, 100: 273},
}
LTE_PRB = {5: 25, 10: 50, 15: 75, 20: 100}


def n_prb(bandwidth_mhz: int, scs_khz: int, rat: str, ctx: SimContext) -> int:
    if rat == "LTE":
        return LTE_PRB[bandwidth_mhz]
    table = PRB_TABLE[scs_khz]
    if ctx.has("F_SCHED_PRB") and scs_khz == 30 and bandwidth_mhz == 100:
        return table[80]  # off-by-one row
    return table[bandwidth_mhz]


def proportional_fair(
    ctx: SimContext,
    mean_sinrs_db: np.ndarray,
    n_steps: int,
    n_prb_total: int,
    scs_khz: int,
    rat: str = "NR",
    demand_mbps: np.ndarray | None = None,
) -> np.ndarray:
    """Single-carrier proportional-fair scheduler; returns the average throughput (Mbps) per UE."""
    n_users = len(mean_sinrs_db)
    sinr = (
        np.stack([m + ar1_trace(ctx.rng, n_steps, 3.0, 0.8) for m in mean_sinrs_db]) - ctx.env.sinr_penalty_db
    )
    rate = (
        resource_rate_re_per_s(n_prb_total, scs_khz)
        * spectral_efficiency(select_cqi(sinr, ctx), rat)
        * 0.9
        / 1e6
    )
    if demand_mbps is not None:
        rate = np.minimum(rate, demand_mbps[:, None] * n_users)
    avg = np.full(n_users, 1e-3)
    served = np.zeros(n_users)
    tracked = min(n_users, 4) if ctx.has("F_SCHED_MULTIUSER") else n_users
    beta = 0.05
    for t in range(n_steps):
        metric = rate[:tracked, t] / avg[:tracked]
        if demand_mbps is not None:
            metric = np.where(served[:tracked] / max(t, 1) >= demand_mbps[:tracked], metric * 0.01, metric)
        k = int(np.argmax(metric))
        alloc = np.zeros(n_users)
        alloc[k] = rate[k, t]
        served += alloc
        avg = (1 - beta) * avg + beta * alloc
    return served / n_steps


def jain_fairness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    denom = len(x) * float(np.sum(x * x))
    return float(np.sum(x) ** 2 / denom) if denom > 0 else 0.0
