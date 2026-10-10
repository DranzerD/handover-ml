"""Integration / smoke tests: pytest execution parity and the CLI workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cellselect.cli import main
from cellselect.data.scenarios import generate_change
from cellselect.execution.pytest_runner import run_pytest
from cellselect.suite.catalog import build_catalog
from cellselect.suite.runner import run_suite


def test_pytest_and_in_process_runner_agree(tmp_path: Path) -> None:
    scen = generate_change(
        np.random.default_rng(0), 7, force_faults=("F_TTT_UNITS", "F_BUFFER_SIZE", "F_RRC_T300")
    )
    path = tmp_path / "scenario.json"
    scen.save(path)
    run = run_pytest(path, None)
    inproc = run_suite(build_catalog(), scen.visible.seed, scen.active_faults, scen.visible.env)
    assert set(run.results) == set(inproc)
    mismatch = {t for t, o in inproc.items() if (run.results[t]["outcome"] == "passed") != o.passed}
    assert not mismatch
    assert run.failed  # the injected faults are visible through pytest


def test_pytest_selection_order_and_subset(tmp_path: Path) -> None:
    scen = generate_change(np.random.default_rng(1), 8, force_faults=())
    path = tmp_path / "scenario.json"
    scen.save(path)
    order = ["HO-INTRA-003", "SMOKE-001", "CFG-VALID-004"]
    run = run_pytest(path, order, results_path=tmp_path / "r.json")
    assert list(run.results) == order
    assert json.loads((tmp_path / "r.json").read_text())["results"][0]["test_id"] == "HO-INTRA-003"


def test_cli_end_to_end(tmp_path: Path, small_dataset: Path, small_models: Path) -> None:
    scen = tmp_path / "s.json"
    assert (
        main(["scenario", "--seed", "5", "--area", "mobility", "--type", "feature", "--out", str(scen)]) == 0
    )
    sel = tmp_path / "sel.json"
    assert (
        main(
            [
                "select",
                "--scenario",
                str(scen),
                "--data",
                str(small_dataset),
                "--models",
                str(small_models),
                "--budget-fraction",
                "0.2",
                "--out",
                str(sel),
            ]
        )
        == 0
    )
    assert json.loads(sel.read_text())["n_selected"] > 0
    assert (
        main(
            ["run", "--scenario", str(scen), "--selection", str(sel), "--results", str(tmp_path / "res.json")]
        )
        == 0
    )
    out = tmp_path / "cmp"
    assert (
        main(
            [
                "compare",
                "--scenario",
                str(scen),
                "--data",
                str(small_dataset),
                "--models",
                str(small_models),
                "--budget-fraction",
                "0.2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads((out / "run_report.json").read_text())
    assert payload["execution"]["determinism_check"]["ok"]
    assert {c["strategy"] for c in payload["baseline_comparison"]} == {
        "full",
        "random",
        "history",
        "coverage",
        "ml",
    }
    assert (out / "run_report.html").stat().st_size > 10_000


def test_cli_reports_missing_dataset(tmp_path: Path) -> None:
    scen = tmp_path / "s.json"
    main(["scenario", "--out", str(scen)])
    assert main(["select", "--scenario", str(scen), "--data", str(tmp_path / "nope")]) == 2
