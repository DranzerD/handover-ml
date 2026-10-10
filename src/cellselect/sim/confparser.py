"""Radio configuration parsing, validation and change application (component: config.parser)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from cellselect.sim.bands import KNOWN_BANDS, TDD_PATTERN_DL_FRACTION, band_info
from cellselect.sim.context import SimContext

ALLOWED_BW_MHZ = (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100)


class ConfigError(ValueError):
    """Raised when a radio configuration is invalid."""


@dataclass(frozen=True)
class RadioConfig:
    band: str
    bandwidth_mhz: int
    mtu: int = 1500
    tdd_pattern: str = "DDDSU"
    tx_power_dbm: float = 46.0
    effective_mtu: int = 1500


def parse_config(raw: dict, ctx: SimContext) -> RadioConfig:
    """Validate a raw configuration dictionary and return a RadioConfig."""
    band = raw.get("band")
    if band not in KNOWN_BANDS:
        raise ConfigError(f"unknown band {band!r}")
    info = band_info(band, ctx)
    try:
        bw = int(raw["bandwidth_mhz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError("bandwidth_mhz missing or not an integer") from exc
    if bw not in ALLOWED_BW_MHZ:
        raise ConfigError(f"bandwidth {bw} MHz is not a standard channel bandwidth")
    within = bw < info.max_bw_mhz if ctx.has("F_CONFIG_RANGE") else bw <= info.max_bw_mhz
    if not within:
        raise ConfigError(f"bandwidth {bw} MHz exceeds band {band} maximum {info.max_bw_mhz} MHz")
    mtu = int(raw.get("mtu", 1500))
    if not 576 <= mtu <= 9216:
        raise ConfigError(f"mtu {mtu} out of range [576, 9216]")
    pattern = raw.get("tdd_pattern", "DDDSU")
    if info.duplex == "TDD" and pattern not in TDD_PATTERN_DL_FRACTION:
        raise ConfigError(f"unsupported TDD pattern {pattern!r}")
    power = float(raw.get("tx_power_dbm", 46.0))
    if not 20.0 <= power <= 49.0:
        raise ConfigError(f"tx_power_dbm {power} out of range [20, 49]")
    return RadioConfig(band, bw, mtu, pattern, power, effective_mtu=mtu)


def apply_config_change(current: RadioConfig, change: dict, ctx: SimContext) -> RadioConfig:
    """Apply a partial configuration change; the result is re-validated as a whole."""
    raw = {
        "band": current.band,
        "bandwidth_mhz": current.bandwidth_mhz,
        "mtu": current.mtu,
        "tdd_pattern": current.tdd_pattern,
        "tx_power_dbm": current.tx_power_dbm,
    }
    raw.update(change)
    new = parse_config(raw, ctx)
    if ctx.has("F_MTU") and "mtu" in change and new.mtu > 1500:
        # PDCP keeps the old SDU size: jumbo MTU accepted by RRC but never propagated.
        new = replace(new, effective_mtu=current.effective_mtu)
    return new


def dl_fraction(cfg: RadioConfig, ctx: SimContext) -> float:
    info = band_info(cfg.band, ctx)
    if info.duplex == "FDD":
        return 1.0
    if cfg.tdd_pattern != "DDDSU":
        return TDD_PATTERN_DL_FRACTION[cfg.tdd_pattern]
    return info.dl_fraction
