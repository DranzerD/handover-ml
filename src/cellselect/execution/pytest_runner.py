"""Execute (a selection of) the validation suite through pytest and collect genuine results."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_DIR = REPO_ROOT / "validation_suite"


@dataclass
class PytestRun:
    exit_code: int
    wall_s: float
    results: dict[str, dict] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)

    @property
    def failed(self) -> set[str]:
        return {t for t, r in self.results.items() if r.get("outcome") != "passed"}

    @property
    def passed(self) -> set[str]:
        return {t for t, r in self.results.items() if r.get("outcome") == "passed"}

    @property
    def nominal_s(self) -> float:
        return float(sum(r.get("nominal_duration_s", 0.0) for r in self.results.values()))

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "wall_s": self.wall_s,
            "nominal_s": self.nominal_s,
            "n_run": len(self.results),
            "n_failed": len(self.failed),
            "command": self.command,
            "results": list(self.results.values()),
        }


def run_pytest(
    scenario_path: str | Path | None,
    selection: list[str] | None = None,
    time_scale: float = 0.0,
    results_path: str | Path | None = None,
    suite_dir: str | Path | None = None,
) -> PytestRun:
    """Run pytest on the validation suite. ``selection=None`` runs the full suite (in catalog order)."""
    suite = Path(suite_dir) if suite_dir else SUITE_DIR
    if not suite.exists():
        raise FileNotFoundError(f"validation suite not found at {suite}")
    with tempfile.TemporaryDirectory(prefix="cellselect_") as tmp:
        res_path = Path(results_path) if results_path else Path(tmp) / "results.json"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(suite),
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
            f"--cs-results={res_path}",
            f"--cs-time-scale={time_scale}",
            "-o",
            "addopts=",
        ]
        if scenario_path:
            cmd.append(f"--cs-scenario={scenario_path}")
        if selection is not None:
            sel_path = Path(tmp) / "selection.json"
            sel_path.write_text(json.dumps({"selected": selection}), encoding="utf-8")
            cmd.append(f"--cs-selection={sel_path}")
        log.info("running pytest on %s tests", "all" if selection is None else len(selection))
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if not res_path.exists():
            raise RuntimeError(
                f"pytest produced no results (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        payload = json.loads(res_path.read_text(encoding="utf-8"))
    results = {r["test_id"]: r for r in payload["results"]}
    if selection is not None:
        missing = set(selection) - set(results)
        if missing:
            raise RuntimeError(f"pytest did not execute selected tests: {sorted(missing)[:10]}")
    # pytest exit codes: 0 all passed, 1 some failed - both are valid test runs.
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"pytest error (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return PytestRun(proc.returncode, wall, results, cmd)
