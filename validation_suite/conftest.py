"""pytest plugin for the simulated cellular validation suite.

Options
-------
--cs-scenario PATH     scenario JSON (faults + environment + seed). Default: fault-free, seed 0.
--cs-selection PATH    selection JSON; only listed tests run, in the listed priority order.
--cs-results PATH      write per-test results (outcome, KPIs, wall time) as JSON.
--cs-time-scale X      sleep nominal_duration * X per test to emulate lab time (default 0).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from cellselect.data.scenarios import ChangeScenario
from cellselect.sim.context import Environment, SimContext
from cellselect.suite.catalog import TestSpec

_RESULTS: dict[str, dict] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("cellselect")
    g.addoption("--cs-scenario", default=os.environ.get("CELLSELECT_SCENARIO"), help="scenario JSON")
    g.addoption("--cs-selection", default=None, help="selection JSON (ordered test IDs)")
    g.addoption("--cs-results", default=None, help="results JSON output path")
    g.addoption(
        "--cs-time-scale",
        type=float,
        default=float(os.environ.get("CELLSELECT_TIME_SCALE", "0")),
        help="emulate lab time: sleep duration_s * scale",
    )


def _spec_of(item: pytest.Item) -> TestSpec | None:
    callspec = getattr(item, "callspec", None)
    spec = callspec.params.get("spec") if callspec else None
    return spec if isinstance(spec, TestSpec) else None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cellular: simulated cellular validation test")
    path = config.getoption("--cs-scenario")
    config._cs_scenario = ChangeScenario.load(path) if path else None  # type: ignore[attr-defined]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    path = config.getoption("--cs-selection")
    if not path:
        return
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    order = list(data["selected"] if isinstance(data, dict) else data)
    rank = {t: i for i, t in enumerate(order)}
    keep, drop = [], []
    for item in items:
        spec = _spec_of(item)
        (keep if spec is not None and spec.test_id in rank else drop).append(item)
    keep.sort(key=lambda it: rank[_spec_of(it).test_id])  # type: ignore[union-attr]
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep


@pytest.fixture
def cs_ctx(request: pytest.FixtureRequest) -> SimContext:
    """SimContext for the current test: scenario faults/environment and a per-test seed."""
    spec = request.node.callspec.params["spec"]
    scenario: ChangeScenario | None = request.config._cs_scenario  # type: ignore[attr-defined]
    if scenario is None:
        return SimContext.for_test(spec.test_id, 0, (), Environment())
    vis = scenario.visible
    return SimContext.for_test(spec.test_id, vis.seed, scenario.active_faults, vis.env)


@pytest.fixture
def cs_lab_time(request: pytest.FixtureRequest):
    """Emulate the nominal lab duration of the test (scaled) after the check runs."""
    yield
    scale = request.config.getoption("--cs-time-scale")
    if scale > 0:
        time.sleep(request.node.callspec.params["spec"].duration_s * scale)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    spec = _spec_of(item)
    if spec is None:
        return
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        entry = _RESULTS.setdefault(
            spec.test_id,
            {
                "test_id": spec.test_id,
                "nodeid": item.nodeid,
                "nominal_duration_s": spec.duration_s,
                "wall_s": 0.0,
            },
        )
        entry["outcome"] = report.outcome if report.when == "call" else "error"
        entry["wall_s"] += report.duration
        entry["kpis"] = getattr(item, "cs_kpis", {})
        if report.failed:
            msg = str(call.excinfo.value) if call.excinfo else str(report.longrepr)
            entry["message"] = msg.splitlines()[0][:300] if msg else ""


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = session.config.getoption("--cs-results")
    if not path:
        return
    scenario = session.config._cs_scenario  # type: ignore[attr-defined]
    payload = {
        "scenario_change_id": scenario.visible.change_id if scenario else None,
        "exit_status": int(exitstatus),
        "results": list(_RESULTS.values()),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
