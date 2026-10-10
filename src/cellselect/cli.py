"""Command-line interface: ``cellselect <command>`` (or ``python -m cellselect <command>``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from cellselect.config import load_config
from cellselect.logging_utils import setup_logging

log = logging.getLogger("cellselect")

DEFAULT_DATA = "data/history"
DEFAULT_MODELS = "artifacts/models"


def _policy(args: argparse.Namespace):
    cfg = load_config(args.config)
    policy = cfg.selection
    if getattr(args, "budget_seconds", None) is not None:
        return replace(policy, budget_seconds=args.budget_seconds, budget_fraction=None)
    if getattr(args, "budget_fraction", None) is not None:
        return replace(policy, budget_fraction=args.budget_fraction, budget_seconds=None)
    return policy


def cmd_catalog(args: argparse.Namespace) -> int:
    from cellselect.suite.catalog import build_catalog, catalog_frame

    df = catalog_frame()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "test_catalog.csv", index=False)
    (out / "test_catalog.json").write_text(
        json.dumps([s.to_dict() for s in build_catalog()], indent=1), encoding="utf-8"
    )
    print(
        df.groupby("category").agg(tests=("test_id", "count"), lab_time_s=("duration_s", "sum")).to_string()
    )
    print(
        f"\n{len(df)} tests, full-suite nominal lab time {df.duration_s.sum():.0f}s "
        f"({df.duration_s.sum() / 3600:.2f} h); critical: {int(df.critical.sum())}. Written to {out}"
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from cellselect.data.generate import generate_history

    cfg = load_config(args.config).generation
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.n_changes is not None:
        cfg = replace(cfg, n_changes=args.n_changes)
    paths = generate_history(cfg, args.out)
    print(Path(paths["metadata"]).read_text())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import pandas as pd

    from cellselect.ml.train import train_models

    cfg = load_config(args.config).training
    if args.fast:
        cfg = replace(cfg, fast=True)
    executions = pd.read_csv(Path(args.data) / "executions.csv")
    card = train_models(executions, cfg, args.out)
    print(f"\nbest model (forward-chaining CV PR-AUC): {card['best_model']}")
    print(
        f"{'model':28s} {'CV PR-AUC':>9} {'test PR-AUC':>11} {'ROC-AUC':>8} {'prec':>6} {'recall':>6} {'F1':>6}"
    )
    for name, m in card["test"].items():
        cv = card["cv"].get(name, {}).get("pr_auc")
        print(
            f"{name:28s} {cv if cv is None else round(cv, 3)!s:>9} {m['pr_auc']:>11.3f} {m['roc_auc']:>8.3f} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
        )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from cellselect.ml.evaluate import evaluate_models

    summary = evaluate_models(args.data, args.models, args.out)
    print(json.dumps({k: summary[k] for k in ("best_model", "metrics", "plots")}, indent=1, default=str))
    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    from cellselect.data.scenarios import generate_change

    rng = np.random.default_rng(args.seed)
    forced = tuple(args.faults.split(",")) if args.faults else None
    if forced:
        from cellselect.sim.faults import FAULTS

        unknown = [f for f in forced if f not in FAULTS]
        if unknown:
            raise SystemExit(f"unknown fault(s) {unknown}; available: {sorted(FAULTS)}")
    scen = generate_change(rng, args.change_id, area=args.area, change_type=args.type, force_faults=forced)
    scen.save(args.out)
    vis = scen.visible
    print(f"scenario written to {args.out}")
    print(
        f"  change {vis.change_id}: {vis.change_type} in '{vis.area}', components {list(vis.changed_components)}, "
        f"{vis.lines_changed} lines, env {vis.env.to_dict()}"
    )
    print(
        f"  injected faults: {list(scen.active_faults) if args.reveal else '(hidden; use --reveal to print)'}"
    )
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    from cellselect.data.scenarios import ChangeScenario
    from cellselect.orchestrate import select_for_scenario, selection_preview

    scen = ChangeScenario.load(args.scenario)
    result = select_for_scenario(scen, args.data, args.models, args.strategy, _policy(args))
    print(selection_preview(result, limit=args.limit))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
        print(f"\nselection written to {args.out} (preview only - nothing executed)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from cellselect.execution.pytest_runner import run_pytest

    selection = None
    if args.selection:
        data = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        selection = data["selected"]
    run = run_pytest(args.scenario, selection, args.time_scale, args.results)
    print(
        f"pytest executed {len(run.results)} tests in {run.wall_s:.2f}s wall "
        f"(nominal lab time {run.nominal_s:.0f}s); failed: {sorted(run.failed) or 'none'}"
    )
    print(f"results: {args.results}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from cellselect.orchestrate import compare_run

    payload = compare_run(
        args.scenario, args.data, args.models, args.strategy, _policy(args), args.out, args.time_scale
    )
    rt, fd = payload["runtime"], payload["fault_detection"]
    print(
        f"\nselected {payload['selection']['n_selected']}/{payload['selection']['n_total']} tests; "
        f"nominal runtime reduction (e2e) {rt['e2e_runtime_reduction_nominal']:.1%}; measured pytest wall "
        f"{rt['measured_wall_selected_s']:.2f}s vs {rt['measured_wall_full_s']:.2f}s full"
    )
    ret = fd["fault_retention"]
    print(
        f"faults detected: selected {fd['faults_detected_selected']} / full {fd['faults_detected_full']} "
        f"(missed {fd['faults_missed']}; retention {'n/a (no detectable faults)' if ret is None else f'{ret:.1%}'})"
    )
    print(f"{'strategy':10s} {'tests':>5} {'time%':>6} {'detected':>8} {'missed':>6}")
    for c in payload["baseline_comparison"]:
        print(
            f"{c['strategy']:10s} {c['n_tests_selected']:>5} {c['selected_fraction']:>6.1%} "
            f"{c['faults_detected_selected']:>8} {c['faults_missed']:>6}"
        )
    for w in payload["warnings"]:
        print("WARNING:", w)
    for f in payload["fallbacks"]:
        print("FALLBACK:", f)
    print(f"\nreport: {payload['report_html']}\njson:   {Path(args.out) / 'run_report.json'}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from cellselect.benchmark.run import run_benchmark
    from cellselect.ml.evaluate import evaluate_models
    from cellselect.reporting.benchmark_report import build_benchmark_report

    cfg = load_config(args.config)
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",")]
    if args.jobs is not None:
        cfg.jobs = args.jobs
    out = Path(args.out)
    summary = run_benchmark(cfg, out, reuse=not args.no_reuse)
    s0 = cfg.seeds[0]
    evaluate_models(out / f"seed_{s0}" / "data", out / f"seed_{s0}" / "models", out / "evaluation")
    report = build_benchmark_report(out)
    print(f"\nbenchmark complete in {summary['elapsed_s']}s")
    print(f"{'strategy':10s} {'budget':>6} {'e2e red.':>8} {'retention':>9} {'missed':>7} {'coverage':>8}")
    for r in summary["strategies"]:
        print(
            f"{r['strategy']:10s} {r['budget_fraction']:>6.0%} {r['e2e_runtime_reduction_mean__mean']:>8.1%} "
            f"{r['fault_retention_pooled__mean']:>9.1%} {r['faults_missed_total__mean']:>7.1f} "
            f"{r['coverage_retained_mean__mean']:>8.1%}"
        )
    print(f"\nreport: {report}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from cellselect.reporting.benchmark_report import build_benchmark_report

    print(build_benchmark_report(args.benchmark_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cellselect", description="ML-driven cellular validation test selection")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--config", default=None, help="experiment config JSON (default: built-in defaults)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("catalog", help="export the validation test catalog")
    s.add_argument("--out", default="data/catalog")
    s.set_defaults(func=cmd_catalog)

    s = sub.add_parser("generate", help="generate the synthetic historical dataset")
    s.add_argument("--out", default=DEFAULT_DATA)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--n-changes", type=int, default=None)
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("train", help="train and compare failure-prediction models")
    s.add_argument("--data", default=DEFAULT_DATA)
    s.add_argument("--out", default=DEFAULT_MODELS)
    s.add_argument("--fast", action="store_true", help="smaller ensembles (CI)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="evaluate trained models (metrics, PR curves, importance)")
    s.add_argument("--data", default=DEFAULT_DATA)
    s.add_argument("--models", default=DEFAULT_MODELS)
    s.add_argument("--out", default="reports/evaluation")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("scenario", help="generate a simulated change / regression scenario")
    s.add_argument("--out", default="scenarios/scenario.json")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--change-id", type=int, default=10_000)
    s.add_argument("--area", default=None, help="phy|mac|transport|rrc|mobility|session|power|config")
    s.add_argument("--type", default=None, help="feature|bugfix|refactor|config_update")
    s.add_argument("--faults", default=None, help="comma-separated fault IDs to force (e.g. F_TTT_UNITS)")
    s.add_argument("--reveal", action="store_true", help="print the hidden injected faults")
    s.set_defaults(func=cmd_scenario)

    for name, helptext in (
        ("select", "select & prioritise tests (preview, no execution)"),
        ("compare", "select, execute via pytest, compare with the full suite, write report"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--scenario", required=True)
        s.add_argument("--data", default=DEFAULT_DATA)
        s.add_argument("--models", default=DEFAULT_MODELS)
        s.add_argument("--strategy", default="ml", choices=["full", "random", "history", "coverage", "ml"])
        g = s.add_mutually_exclusive_group()
        g.add_argument("--budget-fraction", type=float, default=None)
        g.add_argument("--budget-seconds", type=float, default=None)
        if name == "select":
            s.add_argument("--out", default=None, help="write selection JSON")
            s.add_argument("--limit", type=int, default=40, help="rows to print")
            s.set_defaults(func=cmd_select)
        else:
            s.add_argument("--out", default="reports/run")
            s.add_argument(
                "--time-scale", type=float, default=0.0, help="emulate lab time (sleep duration*scale)"
            )
            s.set_defaults(func=cmd_compare)

    s = sub.add_parser("run", help="execute a selection (or the full suite) via pytest")
    s.add_argument("--scenario", required=True)
    s.add_argument(
        "--selection", default=None, help="selection JSON from `select --out`; omit for full suite"
    )
    s.add_argument("--results", default="reports/run/results.json")
    s.add_argument("--time-scale", type=float, default=0.0)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("benchmark", help="multi-seed benchmark of all strategies + HTML report")
    s.add_argument("--out", default="results/benchmark")
    s.add_argument("--seeds", default=None, help="override seeds, e.g. 0,1,2")
    s.add_argument("--jobs", type=int, default=None, help="parallel seed workers")
    s.add_argument("--no-reuse", action="store_true", help="regenerate datasets even if present")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("report", help="rebuild the benchmark HTML report")
    s.add_argument("--benchmark-dir", default="results/benchmark")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
