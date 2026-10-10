"""Simulated code/configuration changes with hidden injected faults.

A ``ChangeScenario`` has two parts:

* ``visible`` (:class:`ChangeContext`) - what a real CI system knows *before* testing: change
  type, touched components, size, lab environment. Selection strategies only ever see this.
* hidden ground truth - the faults actually active in the build (newly injected by this change
  plus unfixed faults carried over from earlier changes). Only the simulator/executor sees it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from cellselect.sim.context import Environment
from cellselect.sim.faults import COMPONENT_NEIGHBOURS, FAULTS, faults_for_component

CHANGE_AREAS: dict[str, tuple[str, ...]] = {
    "phy": ("phy.pathloss", "phy.link_adaptation"),
    "mac": ("mac.harq", "mac.scheduler"),
    "transport": ("transport.latency", "transport.buffer"),
    "rrc": ("rrc.connection", "rrc.timers"),
    "mobility": ("mobility.measurement", "mobility.handover"),
    "session": ("session.reconnect",),
    "power": ("power.control",),
    "config": ("config.parser", "config.band"),
}
AREA_WEIGHTS = {
    "phy": 0.15,
    "mac": 0.15,
    "transport": 0.12,
    "rrc": 0.14,
    "mobility": 0.16,
    "session": 0.07,
    "power": 0.06,
    "config": 0.15,
}
# change type: (probability, base per-component fault probability, median lines changed)
CHANGE_TYPES: dict[str, tuple[float, float, float]] = {
    "feature": (0.35, 0.32, 220.0),
    "bugfix": (0.30, 0.14, 40.0),
    "refactor": (0.20, 0.26, 400.0),
    "config_update": (0.15, 0.22, 25.0),
}
COLLATERAL_FAULT_PROB = 0.07


@dataclass(frozen=True)
class ChangeContext:
    """Information available to test selection before any test runs."""

    change_id: int
    change_type: str
    area: str
    changed_components: tuple[str, ...]
    lines_changed: int
    files_changed: int
    env: Environment = field(default_factory=Environment)
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["changed_components"] = list(self.changed_components)
        d["env"] = self.env.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ChangeContext:
        return cls(
            change_id=int(d["change_id"]),
            change_type=str(d["change_type"]),
            area=str(d["area"]),
            changed_components=tuple(d["changed_components"]),
            lines_changed=int(d["lines_changed"]),
            files_changed=int(d["files_changed"]),
            env=Environment.from_dict(d.get("env", {})),
            seed=int(d.get("seed", 0)),
        )


@dataclass(frozen=True)
class ChangeScenario:
    visible: ChangeContext
    new_faults: tuple[str, ...] = ()
    carried_faults: tuple[str, ...] = ()

    @property
    def active_faults(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.new_faults) | set(self.carried_faults)))

    def to_dict(self) -> dict:
        return {
            "visible": self.visible.to_dict(),
            "ground_truth": {
                "new_faults": list(self.new_faults),
                "carried_faults": list(self.carried_faults),
                "active_faults": list(self.active_faults),
                "fault_details": {f: asdict(FAULTS[f]) for f in self.active_faults},
                "_note": "Hidden ground truth. Never read by selection strategies; used only by the executor.",
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChangeScenario:
        gt = d.get("ground_truth", {})
        return cls(
            ChangeContext.from_dict(d["visible"]),
            tuple(gt.get("new_faults", ())),
            tuple(gt.get("carried_faults", ())),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ChangeScenario:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def random_environment(rng: np.random.Generator) -> Environment:
    return Environment(
        device=str(rng.choice(["devA", "devB", "devC"], p=[0.5, 0.3, 0.2])),
        lab_interference_db=round(float(rng.uniform(0.0, 3.0)), 2),
        temperature_c=round(float(rng.uniform(20.0, 45.0)), 1),
        firmware_branch=str(rng.choice(["main", "release"], p=[0.7, 0.3])),
    )


def generate_change(
    rng: np.random.Generator,
    change_id: int,
    carried_faults: tuple[str, ...] = (),
    area: str | None = None,
    change_type: str | None = None,
    force_faults: tuple[str, ...] | None = None,
) -> ChangeScenario:
    """Draw one change. Fault injection probability depends on change type, size and touched components."""
    areas = list(AREA_WEIGHTS)
    if area is None:
        area = str(rng.choice(areas, p=np.array([AREA_WEIGHTS[a] for a in areas])))
    elif area not in CHANGE_AREAS:
        raise ValueError(f"unknown change area {area!r}; choose from {sorted(CHANGE_AREAS)}")
    types = list(CHANGE_TYPES)
    if change_type is None:
        change_type = str(rng.choice(types, p=np.array([CHANGE_TYPES[t][0] for t in types])))
    elif change_type not in CHANGE_TYPES:
        raise ValueError(f"unknown change type {change_type!r}; choose from {types}")
    if change_type == "config_update":
        area_components = list(CHANGE_AREAS["config"]) if area != "config" else list(CHANGE_AREAS[area])
        area = "config"
    else:
        area_components = list(CHANGE_AREAS[area])
    k = int(rng.integers(1, len(area_components) + 1))
    comps = set(rng.choice(area_components, size=k, replace=False).tolist())
    if rng.random() < 0.3:
        neighbours = sorted({n for c in comps for n in COMPONENT_NEIGHBOURS[c]} - comps)
        if neighbours:
            comps.add(str(rng.choice(neighbours)))
    _, base_p, median_lines = CHANGE_TYPES[change_type]
    lines = max(1, int(rng.lognormal(math.log(median_lines), 0.8)))
    files = max(1, int(rng.poisson(1 + lines / 120)))
    size_factor = 0.6 + 0.4 * min(math.log10(lines) / 2.5, 1.6)

    if force_faults is not None:
        new_faults = tuple(sorted(force_faults))
    else:
        injected: set[str] = set()
        for comp in sorted(comps):
            candidates = faults_for_component(comp)
            if candidates and rng.random() < min(0.75, base_p * size_factor):
                injected.add(str(rng.choice(candidates)))
        if rng.random() < COLLATERAL_FAULT_PROB:
            neighbours = sorted({n for c in comps for n in COMPONENT_NEIGHBOURS[c]} - comps)
            pool = [f for n in neighbours for f in faults_for_component(n)]
            if pool:
                injected.add(str(rng.choice(pool)))
        new_faults = tuple(sorted(injected - set(carried_faults)))
    ctx = ChangeContext(
        change_id,
        change_type,
        area,
        tuple(sorted(comps)),
        lines,
        files,
        random_environment(rng),
        int(rng.integers(0, 2**31 - 1)),
    )
    return ChangeScenario(ctx, new_faults, tuple(sorted(carried_faults)))
