"""Application workflows used by the CLI: select, execute and compare against the full suite."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cellselect.benchmark.metrics import evaluate_selection, runtime_reduction
from cellselect.data.features import HistoryState
from cellselect.data.generate import load_changes, replay_history
from cellselect.data.scenarios import ChangeScenario
from cellselect.execution.pytest_runner import PytestRun, run_pytest
from cellselect.selection.engine import (
    STRATEGY_NAMES,
    SelectionPolicy,
    SelectionResult,
    load_predictor_safely,
    make_strategy,
    run_selection,
)
from cellselect.sim.faults import FAULTS
from cellselect.suite.catalog import build_catalog
from cellselect.suite.runner import TestOutcome, attribute_faults

log = logging.getLogger(__name__)


def load_history(data_dir: str | Path) -> HistoryState:
    d = Path(data_dir)
    if not (d / "executions.csv").exists():
        raise FileNotFoundError(f"no dataset in {d}; run `cellselect generate` first")
    return replay_history(pd.read_csv(d / "executions.csv"), load_changes(d / "changes.jsonl"))


def select_for_scenario(
    scenario: ChangeScenario,
    data_dir: str | Path,
    models_dir: str | Path | None,
    strategy: str,
    policy: SelectionPolicy,
    history: HistoryState | None = None,
) -> SelectionResult:
    history = history or load_history(data_dir)
    predictor, err = load_predictor_safely(str(models_dir) if models_dir else None)
    if strategy == "ml" and predictor is None:
        log.warning("ML model unavailable (%s); the ML strategy will use its safe fallback", err)
    result = run_selection(
        make_strategy(strategy, policy, predictor), scenario.visible, build_catalog(), history, policy
    )
    if strategy == "ml" and predictor is None and err:
        result.fallbacks.append(f"model load error: {err}")
    return result


def selection_preview(result: SelectionResult, limit: int | None = None) -> str:
    lines = [
        f"Strategy: {result.strategy} (effective: {result.effective_strategy or result.strategy})"
        f"{' model=' + result.model_name if result.model_name else ''}",
        f"Budget: {result.budget_s:.0f}s of {result.full_duration_s:.0f}s full suite "
        f"({result.budget_s / result.full_duration_s:.0%}); selected {len(result.selected)} tests, "
        f"{result.selected_duration_s:.0f}s ({result.selected_duration_s / result.full_duration_s:.1%}); "
        f"overhead {result.overhead_s * 1000:.0f} ms",
        "",
    ]
    lines.append(f"{'rank':>4}  {'test_id':14s} {'dur s':>7} {'p_fail':>7}  reason")
    for t in result.selected[:limit]:
        d = result.decisions[t]
        p = f"{d.predicted_p:.3f}" if d.predicted_p is not None else "   -  "
        lines.append(f"{d.rank:>4}  {t:14s} {d.duration_s:7.0f} {p:>7}  {d.reasons[-1] if d.reasons else ''}")
    if limit and len(result.selected) > limit:
        lines.append(f"  ... {len(result.selected) - limit} more selected")
    for w in result.warnings:
        lines.append(f"WARNING: {w}")
    for f in result.fallbacks:
        lines.append(f"FALLBACK: {f}")
    return "\n".join(lines)


def _outcomes_from_run(run: PytestRun) -> dict[str, TestOutcome]:
    return {
        t: TestOutcome(
            t, r.get("outcome") == "passed", r.get("message", ""), r.get("kpis", {}), r.get("wall_s", 0.0)
        )
        for t, r in run.results.items()
    }


def compare_run(
    scenario_path: str | Path,
    data_dir: str | Path,
    models_dir: str | Path | None,
    strategy: str,
    policy: SelectionPolicy,
    out_dir: str | Path,
    time_scale: float = 0.0,
) -> dict:
    """Select -> execute selection via pytest -> execute full suite via pytest -> attribute -> report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario = ChangeScenario.load(scenario_path)
    history = load_history(data_dir)
    catalog = list(build_catalog())
    result = select_for_scenario(scenario, data_dir, models_dir, strategy, policy, history)
    (out / "selection.json").write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")

    sel_run = run_pytest(scenario_path, result.selected, time_scale, out / "selected_results.json")
    full_run = run_pytest(scenario_path, None, time_scale, out / "full_results.json")
    mismatches = sorted(
        t
        for t in result.selected
        if sel_run.results[t]["outcome"] != full_run.results.get(t, {}).get("outcome")
    )
    full_outcomes = _outcomes_from_run(full_run)
    vis = scenario.visible
    detections, noise = attribute_faults(catalog, vis.seed, scenario.active_faults, vis.env, full_outcomes)
    failed_full = {t: not o.passed for t, o in full_outcomes.items()}
    wall = {t: r.get("wall_s", 0.0) for t, r in full_run.results.items()}

    comparisons = []
    for name in STRATEGY_NAMES:
        if name == strategy:
            res = result
        else:
            p = replace(policy, budget_fraction=1.0, budget_seconds=None) if name == "full" else policy
            res = select_for_scenario(scenario, data_dir, models_dir, name, p, history)
        m = evaluate_selection(res, vis, catalog, failed_full, detections, wall)
        m["executed_via_pytest"] = name in (strategy, "full")
        comparisons.append(m)
    main = next(c for c in comparisons if c["strategy"] == strategy)
    payload = {
        "synthetic": True,
        "scenario": {"visible": vis.to_dict()},
        "strategy": strategy,
        "selection": result.to_dict(),
        "execution": {
            "selected": {
                "wall_s": sel_run.wall_s,
                "nominal_lab_s": sel_run.nominal_s,
                "n_run": len(sel_run.results),
                "n_failed": len(sel_run.failed),
                "failed": sorted(sel_run.failed),
            },
            "full": {
                "wall_s": full_run.wall_s,
                "nominal_lab_s": full_run.nominal_s,
                "n_run": len(full_run.results),
                "n_failed": len(full_run.failed),
                "failed": sorted(full_run.failed),
            },
            "time_scale": time_scale,
            "determinism_check": {"mismatched_outcomes": mismatches, "ok": not mismatches},
            "selected_results": list(sel_run.results.values()),
        },
        "runtime": {
            "full_nominal_s": full_run.nominal_s,
            "selected_nominal_s": sel_run.nominal_s,
            "selection_overhead_s": result.overhead_s,
            "runtime_reduction_nominal": runtime_reduction(full_run.nominal_s, sel_run.nominal_s),
            "e2e_runtime_reduction_nominal": runtime_reduction(
                full_run.nominal_s, sel_run.nominal_s + result.overhead_s
            ),
            "measured_wall_full_s": full_run.wall_s,
            "measured_wall_selected_s": sel_run.wall_s,
            "measured_wall_reduction": runtime_reduction(full_run.wall_s, sel_run.wall_s + result.overhead_s),
        },
        "fault_detection": {
            "faults_detected_full": main["faults_detected_full"],
            "faults_detected_selected": main["faults_detected_selected"],
            "faults_missed": main["faults_missed"],
            "fault_retention": main["fault_retention"],
            "coverage_retained": main["coverage_retained"],
            "changed_component_coverage": main["changed_component_coverage"],
        },
        "ground_truth_revealed_after_execution": {
            "active_faults": list(scenario.active_faults),
            "fault_descriptions": {f: FAULTS[f].description for f in scenario.active_faults},
            "detecting_tests": {f: sorted(ts) for f, ts in detections.items()},
            "noise_failures": sorted(noise),
        },
        "baseline_comparison": comparisons,
        "warnings": result.warnings,
        "fallbacks": result.fallbacks,
    }
    (out / "run_report.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    from cellselect.reporting.run_report import build_run_report

    payload["report_html"] = str(build_run_report(payload, out))
    return payload
