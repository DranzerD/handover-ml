"""In-process execution of validation checks (same code path the pytest suite calls).

Used for bulk history generation and benchmarking, where launching pytest thousands of times
would be wasteful. ``tests/test_runner_consistency.py`` verifies that pytest and this runner
produce identical outcomes for the same scenario.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from cellselect.sim.context import Environment, SimContext
from cellselect.suite.catalog import TestSpec
from cellselect.suite.checks import run_check

log = logging.getLogger(__name__)


@dataclass
class TestOutcome:
    __test__ = False

    test_id: str
    passed: bool
    message: str = ""
    kpis: dict[str, float] = field(default_factory=dict)
    wall_s: float = 0.0
    error: bool = False

    @property
    def outcome(self) -> str:
        return "passed" if self.passed else ("error" if self.error else "failed")


def run_test(spec: TestSpec, scenario_seed: int, faults: Iterable[str], env: Environment) -> TestOutcome:
    ctx = SimContext.for_test(spec.test_id, scenario_seed, faults, env)
    t0 = time.perf_counter()
    try:
        kpis = run_check(spec, ctx)
        return TestOutcome(spec.test_id, True, "", kpis, time.perf_counter() - t0)
    except AssertionError as exc:
        return TestOutcome(spec.test_id, False, str(exc), {}, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - a crashing test is a failing test
        log.debug("test %s raised %r", spec.test_id, exc)
        return TestOutcome(
            spec.test_id, False, f"{type(exc).__name__}: {exc}", {}, time.perf_counter() - t0, error=True
        )


def run_suite(
    specs: Sequence[TestSpec], scenario_seed: int, faults: Iterable[str], env: Environment
) -> dict[str, TestOutcome]:
    faults = frozenset(faults)
    return {s.test_id: run_test(s, scenario_seed, faults, env) for s in specs}


def attribute_faults(
    specs: Sequence[TestSpec],
    scenario_seed: int,
    faults: Iterable[str],
    env: Environment,
    combined: dict[str, TestOutcome],
) -> tuple[dict[str, set[str]], set[str]]:
    """Determine which tests detect which injected fault.

    A test *detects* fault f if it fails with f active alone **and** passes with no faults
    under the identical seed/environment. Tests failing without any fault are "noise"
    failures (environment-sensitive / flaky) and never count as detections.

    Returns (fault_id -> detecting test_ids, noise-failing test_ids). Only tests that failed in the
    combined run are re-executed, which keeps attribution cheap.
    """
    failing = [s for s in specs if not combined[s.test_id].passed]
    detections: dict[str, set[str]] = {f: set() for f in faults}
    if not failing:
        return detections, set()
    clean = run_suite(failing, scenario_seed, (), env)
    noise = {tid for tid, o in clean.items() if not o.passed}
    candidates = [s for s in failing if s.test_id not in noise]
    for f in detections:
        single = run_suite(candidates, scenario_seed, (f,), env)
        detections[f] = {tid for tid, o in single.items() if not o.passed}
    return detections, noise
