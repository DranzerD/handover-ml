"""Independent reference oracles ("spec values") used by the validation checks.

These deliberately do NOT call the simulator code under test, so an injected fault in the
simulator cannot silently change the expected value it is compared against.
"""

from __future__ import annotations

import math

import numpy as np

REF_PRB = {
    ("NR", 15): {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 40: 216, 50: 270},
    ("NR", 30): {10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 40: 106, 50: 133, 60: 162, 80: 217, 100: 273},
    ("LTE", 15): {5: 25, 10: 50, 15: 75, 20: 100},
}
# band: (DL frequency MHz, SCS kHz, RAT, DL fraction, UL fraction, default bandwidth MHz)
REF_BANDS = {
    "b13": (751.0, 15, "LTE", 1.0, 1.0, 10),
    "n3": (1842.5, 15, "NR", 1.0, 1.0, 20),
    "n41": (2593.0, 30, "NR", 0.74, 0.26, 100),
    "n78": (3550.0, 30, "NR", 0.74, 0.26, 100),
}
REF_TDD_DL = {"DDDSU": 0.74, "DDSUU": 0.54, "DDDDDDDSUU": 0.77}
REF_SE = {
    "NR": [
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
    ],
    "LTE": [
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
    ],
}


def ref_prb(band: str, bw: int) -> int:
    _, scs, rat, *_ = REF_BANDS[band]
    return REF_PRB[(rat, scs)][bw]


def ref_cqi(sinr_db: float) -> int:
    return int(min(15, max(0, math.floor((sinr_db - 1.0 + 7.0) / 2.0) + 1)))


def ref_se(sinr_db: float, rat: str) -> float:
    return REF_SE[rat][ref_cqi(sinr_db)]


def ref_rate_mbps(n_prb: int, scs_khz: int, se: float, layers: int, fraction: float) -> float:
    """Reference goodput at the 10 % BLER operating point."""
    return n_prb * 12 * 14 * 1000.0 * (scs_khz / 15.0) * 0.75 * se * layers * fraction * 0.9 / 1e6


def ref_dl_tput_mbps(
    band: str, bw: int, sinr_db: float, layers: int = 2, tdd_pattern: str = "DDDSU"
) -> float:
    _, scs, rat, dl, _, _ = REF_BANDS[band]
    if dl < 1.0:
        dl = REF_TDD_DL[tdd_pattern]
    return ref_rate_mbps(ref_prb(band, bw), scs, ref_se(sinr_db, rat), layers, dl)


def ref_pathloss_db(freq_mhz: float, dist_m: float, indoor: bool = False) -> float:
    return (
        32.4
        + 20 * math.log10(freq_mhz / 1000.0)
        + 31.9 * math.log10(max(dist_m, 10.0))
        + (12.0 if indoor else 0.0)
    )


def ref_rsrp_dbm(band: str, dist_m: float, indoor: bool = False) -> float:
    freq, *_, bw = REF_BANDS[band]
    return 46.0 - 10 * math.log10(12 * ref_prb(band, bw)) + 17.0 - ref_pathloss_db(freq, dist_m, indoor)


def ref_ul_sinr_db(pathloss_db: float, n_prb_alloc: int, scs_khz: int, target_db: float = 10.0) -> float:
    noise = -174.0 + 10 * math.log10(12 * scs_khz * 1e3) + 5.0
    limited = 23.0 - 10 * math.log10(n_prb_alloc) + 17.0 - pathloss_db - noise
    return min(target_db, limited)


def ref_latency_p95_ms(load: float, scs_khz: int) -> float:
    slot = 15.0 / scs_khz
    rtt = 8.0 if scs_khz == 15 else 4.0
    mean_wait = 2.0 * slot * load / (1.0 - load)
    return 8.0 + slot + rtt + (-math.log(0.05)) * mean_wait * 1.25 + 2.0


def ref_backoff_recovery_s(
    outage_s: float, base: float = 0.2, cap: float = 8.0, detect: float = 0.15
) -> float:
    t, k = detect, 0
    while t < outage_s:
        t += min(base * 2**k, cap)
        k += 1
    return t - outage_s


def jain(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    d = len(x) * float(np.sum(x * x))
    return float(np.sum(x) ** 2 / d) if d > 0 else 0.0
