"""Catalog of injectable faults and the simulator components they live in.

Each fault is a deliberately realistic software bug (off-by-one, unit error, wrong table entry,
state not cleared, ...). Faults only change behaviour under specific conditions, so whether a
test detects a fault depends on the test's parameters, not on a random label.
"""

from __future__ import annotations

from dataclasses import dataclass

# Simulator components = coverage tags used by the test catalog and by change scenarios.
COMPONENTS: tuple[str, ...] = (
    "phy.pathloss",
    "phy.link_adaptation",
    "mac.harq",
    "mac.scheduler",
    "transport.latency",
    "transport.buffer",
    "rrc.connection",
    "rrc.timers",
    "mobility.measurement",
    "mobility.handover",
    "session.reconnect",
    "power.control",
    "config.parser",
    "config.band",
)

# Static dependency graph between components (used for "neighbouring change" features and for
# collateral faults that land outside the components a change declares).
COMPONENT_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "phy.pathloss": ("phy.link_adaptation", "mobility.measurement", "config.band"),
    "phy.link_adaptation": ("phy.pathloss", "mac.harq", "mac.scheduler"),
    "mac.harq": ("phy.link_adaptation", "transport.latency"),
    "mac.scheduler": ("phy.link_adaptation", "transport.latency", "config.band"),
    "transport.latency": ("mac.harq", "mac.scheduler", "transport.buffer"),
    "transport.buffer": ("transport.latency", "session.reconnect"),
    "rrc.connection": ("rrc.timers", "session.reconnect"),
    "rrc.timers": ("rrc.connection", "mobility.handover", "session.reconnect"),
    "mobility.measurement": ("mobility.handover", "phy.pathloss"),
    "mobility.handover": ("mobility.measurement", "rrc.timers", "rrc.connection"),
    "session.reconnect": ("rrc.connection", "rrc.timers", "transport.buffer"),
    "power.control": ("phy.pathloss", "phy.link_adaptation"),
    "config.parser": ("config.band", "mac.scheduler"),
    "config.band": ("config.parser", "phy.pathloss", "mac.scheduler"),
}


@dataclass(frozen=True)
class Fault:
    fault_id: str
    component: str
    description: str
    manifests_when: str


FAULTS: dict[str, Fault] = {
    f.fault_id: f
    for f in (
        Fault("F_PL_INDOOR", "phy.pathloss", "Indoor penetration loss applied twice", "indoor scenarios"),
        Fault(
            "F_CQI_OFFSET", "phy.link_adaptation", "CQI reported two steps too high", "estimated SINR > 18 dB"
        ),
        Fault("F_BLER_CURVE", "phy.link_adaptation", "256QAM BLER curve shifted by 3 dB", "NR and CQI >= 12"),
        Fault(
            "F_HARQ_MAXRETX",
            "mac.harq",
            "Max HARQ retransmissions read as 1 instead of 3",
            "fading / low margin",
        ),
        Fault(
            "F_SCHED_PRB",
            "mac.scheduler",
            "PRB table lookup off by one row for 100 MHz",
            "30 kHz SCS, 100 MHz",
        ),
        Fault(
            "F_SCHED_MULTIUSER", "mac.scheduler", "PF metric array sized for 4 UEs", "more than 4 active UEs"
        ),
        Fault(
            "F_LATENCY_QUEUE",
            "transport.latency",
            "Queue service time triple-counted at high load",
            "cell load > 0.7",
        ),
        Fault(
            "F_BUFFER_SIZE",
            "transport.buffer",
            "RLC buffer allocated in KB instead of packets",
            "bursts > 128 packets",
        ),
        Fault(
            "F_MEMORY_LEAK",
            "transport.buffer",
            "Bearer context leaked on session teardown",
            "many sessions / long runs",
        ),
        Fault(
            "F_RRC_T300",
            "rrc.connection",
            "T300 configured as 100 ms instead of 1000 ms",
            "setup needs retransmissions",
        ),
        Fault(
            "F_RRC_STATE",
            "rrc.connection",
            "Stale AS context after RLF breaks re-establishment",
            "reconnect after RLF",
        ),
        Fault("F_T310_SCALE", "rrc.timers", "T310 scaled by 1/10", "fades longer than ~100 ms"),
        Fault("F_N311", "rrc.timers", "N311 in-sync counter requires 20 indications", "fades close to T310"),
        Fault(
            "F_L3_FILTER",
            "mobility.measurement",
            "L3 filter coefficient ignored",
            "fast fading near cell border",
        ),
        Fault(
            "F_A3_HYST_SIGN",
            "mobility.handover",
            "A3 hysteresis applied with wrong sign",
            "UE lingering near border",
        ),
        Fault(
            "F_TTT_UNITS",
            "mobility.handover",
            "TTT ticks counted as 40 ms measurement periods",
            "high UE speed",
        ),
        Fault(
            "F_HO_INTERRUPT",
            "mobility.handover",
            "Inter-frequency HO waits for full SI read",
            "inter-frequency handover",
        ),
        Fault(
            "F_BACKOFF",
            "session.reconnect",
            "Reconnect back-off not reset after success",
            "repeated interruptions",
        ),
        Fault(
            "F_POWER_CTRL",
            "power.control",
            "Closed-loop TPC accumulator cannot go positive",
            "path-loss estimation error",
        ),
        Fault(
            "F_CONFIG_RANGE",
            "config.parser",
            "Bandwidth validation uses < instead of <=",
            "bandwidth == band maximum",
        ),
        Fault("F_MTU", "config.parser", "Jumbo MTU change not propagated to PDCP", "MTU change > 1500"),
        Fault("F_BAND_TABLE", "config.band", "n41 TDD pattern entry wrong", "band n41"),
    )
}


def faults_for_component(component: str) -> list[str]:
    return [f.fault_id for f in FAULTS.values() if f.component == component]
