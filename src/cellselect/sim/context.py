"""Simulation context: environment metadata, active faults and a deterministic RNG."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np

DEVICE_NOISE_FIGURE_DB = {"devA": 7.0, "devB": 7.6, "devC": 8.3}


def stable_seed(*parts: object) -> int:
    """Platform-independent 32-bit seed derived from arbitrary parts (unlike ``hash()``)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class Environment:
    """Lab/environment metadata for one execution of the suite."""

    device: str = "devA"
    lab_interference_db: float = 0.0
    temperature_c: float = 25.0
    firmware_branch: str = "main"

    @property
    def noise_figure_db(self) -> float:
        base = DEVICE_NOISE_FIGURE_DB.get(self.device, 7.5)
        return base + max(0.0, self.temperature_c - 25.0) * 0.03

    @property
    def sinr_penalty_db(self) -> float:
        """Effective SINR loss caused by uncontrolled lab interference and device noise figure."""
        return 0.5 * self.lab_interference_db + (self.noise_figure_db - 7.0) * 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Environment:
        return cls(
            **{
                k: data[k]
                for k in ("device", "lab_interference_db", "temperature_c", "firmware_branch")
                if k in data
            }
        )


@dataclass
class SimContext:
    """Everything a simulated test needs: active faults, environment and a seeded RNG."""

    faults: frozenset[str] = field(default_factory=frozenset)
    env: Environment = field(default_factory=Environment)
    seed: int = 0

    def __post_init__(self) -> None:
        self.faults = frozenset(self.faults)
        self.rng = np.random.default_rng(self.seed)

    def has(self, fault_id: str) -> bool:
        return fault_id in self.faults

    @classmethod
    def for_test(
        cls, test_id: str, scenario_seed: int, faults: Iterable[str], env: Environment
    ) -> SimContext:
        """Context for one test execution; the seed depends only on (scenario, test), never on faults,
        so the same test sees identical noise with and without faults (enables fault attribution)."""
        return cls(faults=frozenset(faults), env=env, seed=stable_seed(scenario_seed, test_id))
