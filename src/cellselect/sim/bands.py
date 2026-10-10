"""Band capability table (component: config.band)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from cellselect.sim.context import SimContext

TDD_PATTERN_DL_FRACTION = {"DDDSU": 0.74, "DDSUU": 0.54, "DDDDDDDSUU": 0.77}


@dataclass(frozen=True)
class BandInfo:
    name: str
    rat: str
    dl_freq_mhz: float
    duplex: str
    max_bw_mhz: int
    scs_khz: int
    dl_fraction: float
    ul_fraction: float


_BANDS: dict[str, BandInfo] = {
    "b13": BandInfo("b13", "LTE", 751.0, "FDD", 10, 15, 1.0, 1.0),
    "n3": BandInfo("n3", "NR", 1842.5, "FDD", 40, 15, 1.0, 1.0),
    "n41": BandInfo("n41", "NR", 2593.0, "TDD", 100, 30, 0.74, 0.26),
    "n78": BandInfo("n78", "NR", 3550.0, "TDD", 100, 30, 0.74, 0.26),
}

KNOWN_BANDS: tuple[str, ...] = tuple(_BANDS)


def band_info(name: str, ctx: SimContext) -> BandInfo:
    """Look up a band. Raises KeyError for unknown bands."""
    info = _BANDS[name]
    if ctx.has("F_BAND_TABLE") and name == "n41":
        # Wrong TDD pattern row (DDSUU) copied into the n41 entry.
        info = replace(info, dl_fraction=0.54, ul_fraction=0.46)
    return info
