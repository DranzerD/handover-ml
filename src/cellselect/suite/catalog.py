"""Test catalog: unique IDs, categories, coverage tags, parameters and nominal lab durations.

Durations are *nominal lab execution times* (seconds) for the equivalent test on real
equipment (drive routes, soak tests, conducted RF measurements). They are the cost model used
for budgets and runtime-reduction metrics. The simulated checks themselves run in milliseconds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

CATEGORIES: tuple[str, ...] = (
    "smoke",
    "signal_quality",
    "throughput",
    "latency_loss",
    "connection",
    "handover",
    "interruption",
    "configuration",
    "regression",
    "stability",
)


@dataclass(frozen=True)
class TestSpec:
    __test__ = False  # not a pytest test class

    test_id: str
    template: str
    category: str
    tags: tuple[str, ...]
    duration_s: float
    params: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    critical: bool = False
    description: str = ""

    @property
    def band(self) -> str:
        return str(self.params.get("band", "na"))

    @property
    def coverage_items(self) -> frozenset[str]:
        items = set(self.tags) | {f"cat:{self.category}"}
        if self.band != "na":
            items.add(f"band:{self.band}")
        return frozenset(items)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["coverage_items"] = sorted(self.coverage_items)
        return d


def _ho_duration(path_m: float, speed_kmh: float) -> float:
    return round(path_m / (speed_kmh / 3.6) + 30.0, 1)


@lru_cache(maxsize=1)
def build_catalog() -> tuple[TestSpec, ...]:
    """Deterministically construct the full validation catalog (160 tests)."""
    specs: list[TestSpec] = []

    def add(
        prefix: str,
        template: str,
        category: str,
        tags: tuple[str, ...],
        duration: float,
        params: dict[str, Any],
        critical: bool = False,
        description: str = "",
    ) -> None:
        n = sum(1 for s in specs if s.test_id.startswith(prefix + "-")) + 1
        specs.append(
            TestSpec(
                f"{prefix}-{n:03d}", template, category, tags, float(duration), params, critical, description
            )
        )

    # --- smoke (mandatory / critical) ------------------------------------------------------
    add(
        "SMOKE",
        "smoke_attach",
        "smoke",
        ("rrc.connection",),
        20,
        {"band": "n78", "sinr_db": 10.0},
        True,
        "UE attaches at good SINR",
    )
    add(
        "SMOKE",
        "smoke_dl",
        "smoke",
        ("phy.link_adaptation", "mac.scheduler"),
        40,
        {"band": "n78", "bw": 40, "sinr_db": 15.0},
        True,
        "Basic DL throughput sanity",
    )
    add(
        "SMOKE",
        "smoke_latency",
        "smoke",
        ("transport.latency",),
        30,
        {"band": "n78", "load": 0.3},
        True,
        "Ping latency at light load",
    )
    add(
        "SMOKE",
        "smoke_handover",
        "smoke",
        ("mobility.handover",),
        60,
        {"band": "n78", "speed_kmh": 30.0, "ttt_ms": 160.0},
        True,
        "Single intra-frequency handover",
    )
    add(
        "SMOKE",
        "smoke_config",
        "smoke",
        ("config.parser",),
        10,
        {"band": "n78", "bw": 40},
        True,
        "Default configuration parses",
    )

    # --- signal quality ----------------------------------------------------------------------
    for band in ("b13", "n3", "n41", "n78"):
        for loc in ("outdoor", "indoor"):
            add(
                "SIG-RSRP",
                "rsrp_measurement",
                "signal_quality",
                ("phy.pathloss", "mobility.measurement", "config.band"),
                45,
                {"band": band, "location": loc},
                description="RSRP reporting accuracy and link budget",
            )
    for band in ("b13", "n78"):
        for sinr in (0.0, 10.0, 20.0, 25.0):
            add(
                "SIG-CQI",
                "cqi_link_adaptation",
                "signal_quality",
                ("phy.link_adaptation", "mac.harq"),
                30,
                {"band": band, "sinr_db": sinr},
                description="CQI selection and BLER target",
            )
    for band in ("n3", "n78"):
        for profile in ("gradual", "step", "deep_fade"):
            add(
                "SIG-DEG",
                "link_degradation",
                "signal_quality",
                ("phy.link_adaptation", "mac.harq", "rrc.timers"),
                90,
                {"band": band, "profile": profile},
                description="Link-quality degradation and recovery",
            )

    # --- throughput ----------------------------------------------------------------------------
    for band, bw in (("b13", 10), ("n3", 20), ("n3", 40), ("n41", 100), ("n78", 40), ("n78", 100)):
        for sinr in (6.0, 13.0, 20.0):
            add(
                "TPUT-DL",
                "dl_throughput",
                "throughput",
                ("phy.link_adaptation", "mac.harq", "mac.scheduler", "config.band"),
                60,
                {"band": band, "bw": bw, "sinr_db": sinr},
                description="DL throughput vs reference",
            )
    for band in ("n3", "n78"):
        for dist in (50.0, 200.0, 350.0):
            for err in (0.0, -4.0):
                add(
                    "TPUT-UL",
                    "ul_throughput",
                    "throughput",
                    ("power.control", "phy.pathloss", "phy.link_adaptation"),
                    60,
                    {"band": band, "distance_m": dist, "pl_error_db": err},
                    description="UL throughput / power control",
                )
    for users in (2, 4, 8, 16):
        for traffic in ("equal", "mixed"):
            add(
                "TPUT-MU",
                "multi_user",
                "throughput",
                ("mac.scheduler",),
                120,
                {"band": "n78", "users": users, "traffic": traffic},
                description="Multi-UE scheduling fairness",
            )

    # --- latency / loss -----------------------------------------------------------------------
    for band in ("n3", "n78"):
        for load in (0.3, 0.6, 0.8, 0.9):
            add(
                "LAT-RTT",
                "latency_load",
                "latency_loss",
                ("transport.latency", "mac.harq", "mac.scheduler"),
                60,
                {"band": band, "load": load},
                description="User-plane latency under load",
            )
    for mob in ("static", "vehicular"):
        for interf in (0.0, 4.0, 8.0):
            add(
                "LAT-LOSS",
                "packet_loss",
                "latency_loss",
                ("mac.harq", "phy.link_adaptation"),
                60,
                {"band": "n78", "mobility": mob, "interference_db": interf},
                description="Residual loss under interference",
            )
    for burst in (64, 200, 400):
        for period in (10, 20):
            add(
                "LAT-BURST",
                "burst_buffer",
                "latency_loss",
                ("transport.buffer", "transport.latency"),
                45,
                {"burst_packets": burst, "period_steps": period},
                description="Bursty traffic buffering",
            )

    # --- connection ---------------------------------------------------------------------------
    for band in ("n3", "n78"):
        for sinr in (-2.0, 0.0, 5.0, 15.0):
            add(
                "CONN-SETUP",
                "rrc_setup",
                "connection",
                ("rrc.connection", "phy.link_adaptation"),
                50,
                {"band": band, "sinr_db": sinr},
                description="RRC connection establishment",
            )
    for cause in ("rlf", "release"):
        for cycles in (1, 3, 5):
            add(
                "CONN-RECON",
                "reconnection",
                "connection",
                ("rrc.connection", "rrc.timers", "session.reconnect"),
                90,
                {"cause": cause, "cycles": cycles},
                description="Reconnection after link loss",
            )

    # --- handover -----------------------------------------------------------------------------
    for speed in (3.0, 30.0, 60.0, 120.0):
        for ttt in (40.0, 160.0, 320.0):
            add(
                "HO-INTRA",
                "intra_handover",
                "handover",
                ("mobility.handover", "mobility.measurement", "rrc.timers"),
                _ho_duration(260, speed),
                {"band": "n78", "speed_kmh": speed, "ttt_ms": ttt},
                description="Intra-frequency handover drive",
            )
    for speed in (30.0, 120.0):
        add(
            "HO-INTRA",
            "intra_handover",
            "handover",
            ("mobility.handover", "mobility.measurement", "rrc.timers"),
            _ho_duration(260, speed),
            {"band": "n3", "speed_kmh": speed, "ttt_ms": 160.0},
            description="Intra-frequency handover drive",
        )
    for src, dst in (("n3", "n78"), ("n78", "n41")):
        for speed in (30.0, 120.0):
            add(
                "HO-INTER",
                "inter_handover",
                "handover",
                ("mobility.handover", "mobility.measurement", "config.band"),
                _ho_duration(400, speed),
                {"band": src, "target_band": dst, "speed_kmh": speed},
                description="Inter-frequency handover drive",
            )
    for hyst in (1.0, 2.0, 3.0):
        for fading in ("low", "high"):
            add(
                "HO-PP",
                "ping_pong",
                "handover",
                ("mobility.handover", "mobility.measurement"),
                300,
                {"band": "n78", "hysteresis_db": hyst, "fading": fading},
                description="Border loitering ping-pong",
            )

    # --- network interruptions ----------------------------------------------------------------
    for outage in (0.5, 2.0, 10.0):
        for count in (1, 3):
            add(
                "INT-OUT",
                "outage_recovery",
                "interruption",
                ("session.reconnect", "rrc.timers"),
                count * (outage + 20.0) + 30.0,
                {"outage_s": outage, "count": count},
                description="Network outage detection and recovery",
            )
    for fade in (200, 500, 850, 1500, 3000):
        add(
            "INT-RLM",
            "rlm_fade",
            "interruption",
            ("rrc.timers",),
            60,
            {"fade_ms": fade},
            description="Radio link monitoring vs. fade duration",
        )

    # --- configuration ------------------------------------------------------------------------
    cfg_cases: list[tuple[dict[str, Any], tuple[str, ...]]] = [
        (
            {"band": "n3", "from": {"bw": 20}, "change": {"bandwidth_mhz": 40}, "expect": "accept"},
            ("config.parser", "config.band", "mac.scheduler"),
        ),
        (
            {"band": "n78", "from": {"bw": 40}, "change": {"bandwidth_mhz": 100}, "expect": "accept"},
            ("config.parser", "config.band", "mac.scheduler"),
        ),
        (
            {"band": "n41", "from": {"bw": 40}, "change": {"bandwidth_mhz": 100}, "expect": "accept"},
            ("config.parser", "config.band", "mac.scheduler"),
        ),
        (
            {
                "band": "n3",
                "from": {"bw": 20},
                "change": {"band": "n78", "bandwidth_mhz": 40},
                "expect": "accept",
            },
            ("config.parser", "config.band", "phy.pathloss"),
        ),
        (
            {"band": "n78", "from": {"bw": 40}, "change": {"mtu": 1280}, "expect": "accept"},
            ("config.parser", "transport.buffer"),
        ),
        (
            {"band": "n78", "from": {"bw": 40}, "change": {"mtu": 9000}, "expect": "accept"},
            ("config.parser", "transport.buffer"),
        ),
        (
            {"band": "n78", "from": {"bw": 40}, "change": {"tdd_pattern": "DDSUU"}, "expect": "accept"},
            ("config.parser", "mac.scheduler"),
        ),
        (
            {"band": "n3", "from": {"bw": 20}, "change": {"bandwidth_mhz": 100}, "expect": "reject"},
            ("config.parser", "config.band"),
        ),
        (
            {"band": "n3", "from": {"bw": 20}, "change": {"band": "n99"}, "expect": "reject"},
            ("config.parser",),
        ),
        (
            {"band": "b13", "from": {"bw": 5}, "change": {"bandwidth_mhz": 10}, "expect": "accept"},
            ("config.parser", "config.band"),
        ),
    ]
    for params, tags in cfg_cases:
        add(
            "CFG-APPLY",
            "config_apply",
            "configuration",
            tags,
            25,
            params,
            description="Apply configuration change",
        )
    valid_cases = [
        ({"band": "n78", "bandwidth_mhz": 40}, "accept"),
        ({"band": "n78", "bandwidth_mhz": 100}, "accept"),
        ({"band": "b13", "bandwidth_mhz": 5}, "accept"),
        ({"band": "n78", "bandwidth_mhz": 40, "mtu": 10000}, "reject"),
        ({"band": "n41", "bandwidth_mhz": 40, "tdd_pattern": "DDXU"}, "reject"),
        ({"band": "n3", "bandwidth_mhz": 20, "tx_power_dbm": 55}, "reject"),
    ]
    for i, (raw, expect) in enumerate(valid_cases):
        add(
            "CFG-VALID",
            "config_validation",
            "configuration",
            ("config.parser", "config.band"),
            5,
            {"band": raw["band"], "raw": raw, "expect": expect},
            critical=(i == 0),
            description="Configuration schema validation",
        )

    # --- regression ---------------------------------------------------------------------------
    for profile, tags in (
        (
            "urban",
            (
                "phy.link_adaptation",
                "mac.scheduler",
                "transport.latency",
                "mobility.handover",
                "rrc.connection",
            ),
        ),
        (
            "suburban",
            ("phy.link_adaptation", "mac.harq", "mac.scheduler", "mobility.handover", "rrc.connection"),
        ),
        (
            "highway",
            ("phy.link_adaptation", "mac.harq", "mobility.handover", "mobility.measurement", "rrc.timers"),
        ),
        ("indoor", ("phy.pathloss", "phy.link_adaptation", "transport.latency", "rrc.connection")),
    ):
        add(
            "REG-KPI",
            "kpi_regression",
            "regression",
            tags,
            300,
            {"band": "n78", "profile": profile},
            description="Composite KPI regression with tight tolerances",
        )

    # --- stability ----------------------------------------------------------------------------
    for dur in (900, 1800, 3600):
        for traffic in ("steady", "bursty"):
            add(
                "STAB-LONG",
                "stability",
                "stability",
                ("transport.buffer", "rrc.timers", "session.reconnect", "rrc.connection"),
                dur,
                {"band": "n78", "soak_s": dur, "traffic": traffic},
                description="Long-duration soak test",
            )
    return tuple(specs)


def catalog_by_id() -> dict[str, TestSpec]:
    return {s.test_id: s for s in build_catalog()}


def catalog_frame():
    """Catalog as a pandas DataFrame (one row per test)."""
    import pandas as pd

    rows = []
    for s in build_catalog():
        rows.append(
            {
                "test_id": s.test_id,
                "template": s.template,
                "category": s.category,
                "band": s.band,
                "tags": ";".join(s.tags),
                "n_tags": len(s.tags),
                "duration_s": s.duration_s,
                "critical": s.critical,
                "description": s.description,
            }
        )
    return pd.DataFrame(rows)


def specs_for_template(template: str) -> list[TestSpec]:
    specs = [s for s in build_catalog() if s.template == template]
    if not specs:
        raise KeyError(f"no tests for template {template!r}")
    return specs
