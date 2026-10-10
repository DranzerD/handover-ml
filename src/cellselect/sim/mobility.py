"""Two-cell mobility model with A3-event handover (components: mobility.handover, mobility.measurement).

Cell A sits at x=0 and cell B at x=ISD on a straight road. The caller supplies the UE trajectory.
Per step we compute true RSRP (path loss + distance-correlated shadowing + temporally-correlated
fast fading), L3-filtered measurements, A3 entering condition with time-to-trigger, handover
execution with interruption, ping-pong detection and radio link monitoring on the serving link.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cellselect.sim.context import SimContext
from cellselect.sim.radio import ar1_trace, l3_filter, noise_per_re_dbm, pathloss_db, rsrp_dbm
from cellselect.sim.rrc import RadioLinkMonitor

PING_PONG_WINDOW_S = 1.0


@dataclass
class DriveResult:
    handovers: list[tuple[float, int, int]] = field(default_factory=list)
    ping_pongs: int = 0
    rlf_count: int = 0
    interruptions_ms: list[float] = field(default_factory=list)
    min_serving_sinr_db: float = float("inf")

    @property
    def max_interruption_ms(self) -> float:
        return max(self.interruptions_ms, default=0.0)


def linear_path(start_m: float, end_m: float, speed_kmh: float, dt_s: float) -> np.ndarray:
    v = speed_kmh / 3.6
    n = max(2, int(abs(end_m - start_m) / (v * dt_s)) + 1)
    return np.linspace(start_m, end_m, n)


def oscillating_path(
    center_m: float, amplitude_m: float, speed_kmh: float, duration_s: float, dt_s: float
) -> np.ndarray:
    v = speed_kmh / 3.6
    t = np.arange(0.0, duration_s, dt_s)
    period = 4.0 * amplitude_m / v
    phase = (t % period) / period
    tri = np.where(phase < 0.5, 4 * phase - 1, 3 - 4 * phase)  # triangle wave in [-1, 1]
    return center_m + amplitude_m * tri


def _shadowing(ctx: SimContext, length_m: int, std_db: float, decorr_m: float = 50.0) -> np.ndarray:
    return ar1_trace(ctx.rng, length_m + 1, std_db, math.exp(-1.0 / decorr_m))


def simulate_drive(
    ctx: SimContext,
    positions_m: np.ndarray,
    dt_s: float,
    speed_kmh: float,
    freq_mhz: tuple[float, float] = (3550.0, 3550.0),
    n_prb: tuple[int, int] = (106, 106),
    scs_khz: int = 30,
    isd_m: float = 500.0,
    fading_std_db: float = 1.5,
    shadow_std_db: float = 3.0,
    a3_offset_db: float = 2.0,
    hysteresis_db: float = 1.0,
    ttt_ms: float = 160.0,
) -> DriveResult:
    inter_freq = freq_mhz[0] != freq_mhz[1]
    n = len(positions_m)
    x = np.clip(positions_m, 1.0, isd_m - 1.0)
    rho_ff = max(0.2, math.exp(-(speed_kmh / 3.6) * dt_s / 2.0))
    rsrp = np.empty((2, n))
    for c, (dist, f, prb) in enumerate(((x, freq_mhz[0], n_prb[0]), (isd_m - x, freq_mhz[1], n_prb[1]))):
        shadow = _shadowing(ctx, int(isd_m), shadow_std_db)[np.round(x).astype(int)]
        ff = ar1_trace(ctx.rng, n, fading_std_db, rho_ff)
        rsrp[c] = rsrp_dbm(46.0, prb, pathloss_db(f, dist, ctx)) + shadow + ff
    meas = rsrp + ctx.rng.normal(0.0, 0.7, size=rsrp.shape)  # measurement error
    filt = np.stack([l3_filter(meas[0], ctx), l3_filter(meas[1], ctx)])

    noise_mw = 10 ** (noise_per_re_dbm(scs_khz, ctx.env.noise_figure_db) / 10.0)
    far_interference_mw = 10 ** (-118.0 / 10.0)
    if inter_freq:
        interf = np.full((2, n), noise_mw + far_interference_mw)
    else:
        # Neighbour cell at 50 % load is the dominant interferer on the same carrier.
        interf = (
            np.stack([10 ** ((rsrp[1] - 3.0) / 10.0), 10 ** ((rsrp[0] - 3.0) / 10.0)])
            + noise_mw
            + far_interference_mw
        )
    sinr = rsrp - 10.0 * np.log10(interf) - ctx.env.sinr_penalty_db

    tick_ms = 40.0 if ctx.has("F_TTT_UNITS") else 4.0  # TTT read in 4 ms ticks; bug uses 40 ms steps
    ttt_steps = max(1, math.ceil(ttt_ms / 4.0 * tick_ms / (dt_s * 1000.0)))
    if ctx.has("F_A3_HYST_SIGN"):
        margin = a3_offset_db - hysteresis_db
    else:
        margin = a3_offset_db + hysteresis_db

    res = DriveResult()
    rlm = RadioLinkMonitor(ctx)
    serving = 0 if x[0] < isd_m / 2 else 1
    counter = 0
    last_ho_time: float | None = None
    last_ho_from: int | None = None
    blocked_until = -1
    for i in range(n):
        now_s = i * dt_s
        target = 1 - serving
        s_now = float(sinr[serving, i])
        res.min_serving_sinr_db = min(res.min_serving_sinr_db, s_now)
        if i < blocked_until:
            continue
        if rlm.step(s_now, now_s * 1000.0):
            res.rlf_count += 1
            serving = int(np.argmax(filt[:, i]))  # re-establish on the strongest cell
            counter = 0
            continue
        if filt[target, i] > filt[serving, i] + margin:
            counter += 1
        else:
            counter = 0
        if counter >= ttt_steps:
            if (
                last_ho_from == target
                and last_ho_time is not None
                and now_s - last_ho_time <= PING_PONG_WINDOW_S
            ):
                res.ping_pongs += 1
            interruption = (40.0 if inter_freq else 28.0) + float(ctx.rng.uniform(0.0, 10.0))
            if inter_freq and ctx.has("F_HO_INTERRUPT"):
                interruption += 110.0
            res.interruptions_ms.append(interruption)
            res.handovers.append((now_s, serving, target))
            last_ho_from, last_ho_time = serving, now_s
            serving = target
            counter = 0
            rlm.reset()
            blocked_until = i + max(1, math.ceil(interruption / (dt_s * 1000.0)))
    return res
