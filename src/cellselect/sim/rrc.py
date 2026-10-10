"""RRC procedures: connection setup, radio link monitoring and re-establishment.

Components: rrc.connection, rrc.timers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cellselect.sim.context import SimContext

# Nominal one-shot latency (ms) of each setup step and the extra delay of one retransmission.
SETUP_STEPS_MS = np.array([10.0, 6.0, 8.0, 8.0, 15.0, 9.0])  # preamble, RAR, msg3, msg4, setup, complete
RETX_PENALTY_MS = np.array([20.0, 8.0, 8.0, 8.0, 10.0, 8.0])
PREAMBLE_TRANS_MAX = 10
Q_OUT_DB = -8.0
Q_IN_DB = -6.0


@dataclass(frozen=True)
class RlmConfig:
    t310_ms: float = 1000.0
    n310: int = 2
    n311: int = 1


def t300_ms(ctx: SimContext) -> float:
    return 100.0 if ctx.has("F_RRC_T300") else 1000.0


def robust_bler(sinr_db: np.ndarray | float) -> np.ndarray:
    """Error rate of robust (low-MCS) control-plane messages."""
    return 1.0 / (1.0 + np.exp(np.asarray(sinr_db, dtype=float) + 5.0))


@dataclass
class SetupResult:
    success: np.ndarray
    setup_time_ms: np.ndarray

    @property
    def success_rate(self) -> float:
        return float(self.success.mean())

    @property
    def mean_time_ms(self) -> float:
        ok = self.setup_time_ms[self.success]
        return float(ok.mean()) if ok.size else float("inf")


def connection_setup(ctx: SimContext, sinr_db: float, n_trials: int) -> SetupResult:
    """Monte-Carlo RRC connection establishment (random access + RRC setup)."""
    sinr = sinr_db - ctx.env.sinr_penalty_db + ctx.rng.normal(0.0, 1.0, size=n_trials)
    p_fail = np.clip(robust_bler(sinr), 1e-9, 0.999)
    total = np.zeros(n_trials)
    ok = np.ones(n_trials, dtype=bool)
    for step, (base, penalty) in enumerate(zip(SETUP_STEPS_MS, RETX_PENALTY_MS)):
        attempts = ctx.rng.geometric(1.0 - p_fail)
        if step == 0:
            ok &= attempts <= PREAMBLE_TRANS_MAX
        total += base + (attempts - 1) * (base + penalty)
    ok &= total <= t300_ms(ctx)
    return SetupResult(ok, total)


class RadioLinkMonitor:
    """Incremental N310/T310/N311 radio link monitoring state machine."""

    def __init__(self, ctx: SimContext, cfg: RlmConfig | None = None) -> None:
        cfg = cfg or RlmConfig()
        self.t310 = cfg.t310_ms / 10.0 if ctx.has("F_T310_SCALE") else cfg.t310_ms
        self.n310 = cfg.n310
        self.n311 = 20 if ctx.has("F_N311") else cfg.n311
        self.reset()

    def reset(self) -> None:
        self.oos_run = 0
        self.is_run = 0
        self.t310_start: float | None = None

    def step(self, sinr_db: float, now_ms: float) -> bool:
        """Feed one in-sync/out-of-sync indication; return True if RLF is declared."""
        if sinr_db < Q_OUT_DB:
            self.oos_run += 1
            self.is_run = 0
        elif sinr_db > Q_IN_DB:
            self.is_run += 1
            self.oos_run = 0
        if self.t310_start is None:
            if self.oos_run >= self.n310:
                self.t310_start = now_ms
            return False
        if self.is_run >= self.n311:
            self.t310_start = None
            return False
        if now_ms - self.t310_start >= self.t310:
            self.reset()
            return True
        return False


def radio_link_monitor(
    ctx: SimContext, sinr_db: np.ndarray, dt_ms: float, cfg: RlmConfig | None = None
) -> list[int]:
    """Return sample indices at which radio link failure (T310 expiry) is declared."""
    rlm = RadioLinkMonitor(ctx, cfg)
    return [i for i, s in enumerate(sinr_db) if rlm.step(float(s), i * dt_ms)]


def reestablishment_time_ms(ctx: SimContext, after_rlf: bool, sinr_db: float) -> float:
    """Time from link loss to restored RRC connection."""
    base = 60.0 + float(ctx.rng.exponential(15.0))
    if after_rlf and ctx.has("F_RRC_STATE"):
        # Re-establishment rejected (stale AS context) -> cell reselection + full setup.
        return base + 350.0 + float(connection_setup(ctx, sinr_db, 1).setup_time_ms[0]) + 250.0
    return base + float(ctx.rng.exponential(10.0))
