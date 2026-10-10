"""Physical-layer behavioural models.

Components: phy.pathloss, phy.link_adaptation, mac.harq, mobility.measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cellselect.sim.context import SimContext

# Spectral efficiency (bits/RE) per CQI index; NR 256QAM table and LTE 64QAM table (3GPP-style).
CQI_SE_NR = np.array(
    [
        0.0,
        0.1523,
        0.3770,
        0.8770,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,
        6.2266,
        6.9141,
        7.4063,
    ]
)
CQI_SE_LTE = np.array(
    [
        0.0,
        0.1523,
        0.2344,
        0.3770,
        0.6016,
        0.8770,
        1.1758,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,
    ]
)
CQI_BACKOFF_DB = 1.0
HARQ_COMBINING_GAIN_DB = 3.0
INDOOR_PENETRATION_DB = 12.0
BS_ANTENNA_GAIN_DB = 17.0

FADING_PROFILES: dict[str, tuple[float, float]] = {
    # name: (std dB, AR(1) correlation per step)
    "static": (0.8, 0.98),
    "pedestrian": (2.5, 0.90),
    "vehicular": (4.0, 0.60),
}


def ar1_trace(rng: np.random.Generator, n: int, std: float, rho: float) -> np.ndarray:
    """Stationary AR(1) Gaussian process with the given marginal std and lag-1 correlation."""
    noise = rng.normal(0.0, std, size=n)
    out = np.empty(n)
    out[0] = noise[0]
    scale = np.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + scale * noise[i]
    return out


# ---------------------------------------------------------------- phy.pathloss
def pathloss_db(
    freq_mhz: float, dist_m: np.ndarray | float, ctx: SimContext, indoor: bool = False
) -> np.ndarray:
    """Simplified urban-micro NLOS path loss (dB)."""
    d = np.maximum(np.asarray(dist_m, dtype=float), 10.0)
    pl = 32.4 + 20.0 * np.log10(freq_mhz / 1000.0) + 31.9 * np.log10(d)
    if indoor:
        pl = pl + INDOOR_PENETRATION_DB
        if ctx.has("F_PL_INDOOR"):
            pl = pl + INDOOR_PENETRATION_DB  # penetration loss applied twice
    return pl


def rsrp_dbm(tx_power_dbm: float, n_prb: int, pl_db: np.ndarray | float) -> np.ndarray:
    return tx_power_dbm - 10.0 * np.log10(12 * n_prb) + BS_ANTENNA_GAIN_DB - np.asarray(pl_db)


def noise_per_re_dbm(scs_khz: int, noise_figure_db: float) -> float:
    return -174.0 + 10.0 * np.log10(scs_khz * 1e3) + noise_figure_db


# ---------------------------------------------------------------- phy.link_adaptation
def sinr_required_db(cqi: np.ndarray) -> np.ndarray:
    return -7.0 + 2.0 * (np.asarray(cqi, dtype=float) - 1.0)


def select_cqi(sinr_est_db: np.ndarray, ctx: SimContext) -> np.ndarray:
    est = np.asarray(sinr_est_db, dtype=float)
    cqi = np.clip(np.floor((est - CQI_BACKOFF_DB + 7.0) / 2.0) + 1, 0, 15).astype(int)
    if ctx.has("F_CQI_OFFSET"):
        cqi = np.where(est > 18.0, np.minimum(cqi + 2, 15), cqi)
    return cqi


def bler(sinr_db: np.ndarray, cqi: np.ndarray, ctx: SimContext, rat: str) -> np.ndarray:
    """First-transmission block error rate as a logistic function of SINR margin."""
    cqi = np.asarray(cqi)
    margin = np.asarray(sinr_db, dtype=float) - sinr_required_db(np.maximum(cqi, 1))
    if ctx.has("F_BLER_CURVE") and rat == "NR":
        margin = np.where(cqi >= 12, margin - 3.0, margin)
    out = 1.0 / (1.0 + np.exp(np.clip(2.0 * margin, -50, 50)))
    return np.where(cqi == 0, 1.0, out)


def spectral_efficiency(cqi: np.ndarray, rat: str) -> np.ndarray:
    table = CQI_SE_NR if rat == "NR" else CQI_SE_LTE
    return table[np.asarray(cqi, dtype=int)]


# ---------------------------------------------------------------- mac.harq
def harq_max_retx(ctx: SimContext) -> int:
    return 1 if ctx.has("F_HARQ_MAXRETX") else 3


def harq_outcome(
    sinr_db: np.ndarray, cqi: np.ndarray, ctx: SimContext, rat: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (residual loss after HARQ, expected number of transmissions) per transport block."""
    max_retx = harq_max_retx(ctx)
    survive = np.ones_like(np.asarray(sinr_db, dtype=float))
    expected_tx = np.zeros_like(survive)
    for k in range(max_retx + 1):
        expected_tx += survive
        survive = survive * bler(sinr_db + HARQ_COMBINING_GAIN_DB * k, cqi, ctx, rat)
    idle = np.asarray(cqi) == 0
    return np.where(idle, 0.0, survive), np.where(idle, 1.0, expected_tx)


@dataclass
class LinkResult:
    sinr_db: np.ndarray
    cqi: np.ndarray
    bler: np.ndarray
    residual_loss: np.ndarray
    expected_tx: np.ndarray
    tput_mbps: np.ndarray

    @property
    def mean_tput(self) -> float:
        return float(self.tput_mbps.mean())


def resource_rate_re_per_s(n_prb: int, scs_khz: int) -> float:
    """Data resource elements per second (14 symbols/slot, 25 % overhead)."""
    return n_prb * 12 * 14 * 1000.0 * (scs_khz / 15.0) * 0.75


def simulate_link(
    ctx: SimContext,
    mean_sinr_db: float,
    n_steps: int,
    fading: str,
    rat: str,
    n_prb: int,
    scs_khz: int,
    layers: int = 2,
    dl_fraction: float = 1.0,
    extra_interference_db: float = 0.0,
) -> LinkResult:
    """Simulate a link for n_steps scheduling intervals with fading and one-step CQI delay."""
    std, rho = FADING_PROFILES[fading]
    sinr = (
        mean_sinr_db - ctx.env.sinr_penalty_db - extra_interference_db + ar1_trace(ctx.rng, n_steps, std, rho)
    )
    sinr_est = np.concatenate(([sinr[0]], sinr[:-1])) + ctx.rng.normal(0.0, 0.5, n_steps)
    cqi = select_cqi(sinr_est, ctx)
    b0 = bler(sinr, cqi, ctx, rat)
    residual, exp_tx = harq_outcome(sinr, cqi, ctx, rat)
    raw = resource_rate_re_per_s(n_prb, scs_khz) * spectral_efficiency(cqi, rat) * layers * dl_fraction
    tput = raw * (1.0 - residual) / exp_tx / 1e6
    return LinkResult(sinr, cqi, b0, residual, exp_tx, tput)


# ---------------------------------------------------------------- mobility.measurement
def l3_filter(samples: np.ndarray, ctx: SimContext, k: int = 8) -> np.ndarray:
    """Layer-3 exponential filter F_n = (1-a) F_{n-1} + a M_n with a = 1/2^(k/4)."""
    a = 1.0 if ctx.has("F_L3_FILTER") else 1.0 / 2 ** (k / 4.0)
    out = np.empty_like(samples, dtype=float)
    acc = samples[0]
    for i, m in enumerate(samples):
        acc = (1.0 - a) * acc + a * m
        out[i] = acc
    return out
