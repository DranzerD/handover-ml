"""Uplink open/closed-loop power control (component: power.control)."""

from __future__ import annotations

import numpy as np

from cellselect.sim.context import SimContext
from cellselect.sim.radio import BS_ANTENNA_GAIN_DB

P_MAX_DBM = 23.0
P0_DBM = -96.0
ALPHA = 0.8
TARGET_SINR_DB = 10.0
BS_NOISE_FIGURE_DB = 5.0


def ul_noise_per_prb_dbm(scs_khz: int) -> float:
    return -174.0 + 10.0 * np.log10(12 * scs_khz * 1e3) + BS_NOISE_FIGURE_DB


def simulate_ul_sinr(
    ctx: SimContext,
    pathloss_db: float,
    n_prb_alloc: int,
    scs_khz: int,
    pl_estimation_error_db: float,
    n_steps: int,
) -> np.ndarray:
    """Per-step UL SINR (dB) at the base station with TPC closed loop driving SINR to target."""
    noise = ul_noise_per_prb_dbm(scs_khz) + ctx.env.sinr_penalty_db
    lo, hi = (-20.0, 0.0) if ctx.has("F_POWER_CTRL") else (-20.0, 20.0)
    f_cl = 0.0
    out = np.empty(n_steps)
    fading = ctx.rng.normal(0.0, 1.0, n_steps)
    for t in range(n_steps):
        pl_est = pathloss_db + pl_estimation_error_db
        p_tx = min(P_MAX_DBM, P0_DBM + 10 * np.log10(n_prb_alloc) + ALPHA * pl_est + f_cl)
        sinr = p_tx - 10 * np.log10(n_prb_alloc) + BS_ANTENNA_GAIN_DB - pathloss_db - noise + fading[t]
        out[t] = sinr
        tpc = float(np.clip(np.round(TARGET_SINR_DB - sinr), -1.0, 3.0))
        f_cl = float(np.clip(f_cl + tpc * 0.5, lo, hi))
    return out
