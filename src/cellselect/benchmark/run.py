"""Multi-seed benchmark: generate -> train -> replay held-out changes -> compare strategies.

For each seed the held-out (temporal test-period) changes are replayed in chronological order.
For every change, every strategy and every budget, a selection is computed from information
available *before* the change is tested; it is then scored against the full-suite outcomes and
the single-fault attribution of that same change. History is then updated with the full-suite
outcomes (assumption: a nightly full run keeps history complete; documented in the README).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cellselect.benchmark.metrics import evaluate_selection
from cellselect.config import ExperimentConfig
from cellselect.data.generate import generate_history, load_changes, replay_history
from cellselect.data.scenarios import ChangeContext
from cellselect.ml.predictor import FailurePredictor
from cellselect.ml.train import train_models
from cellselect.selection.engine import make_strategy, run_selection
from cellselect.suite.catalog import catalog_by_id

log = logging.getLogger(__name__)


def _seed_dirs(out: Path, seed: int) -> tuple[Path, Path]:
    return out / f"seed_{seed}" / "data", out / f"seed_{seed}" / "models"


def run_seed(cfg: ExperimentConfig, seed: int, out_dir: str | Path, reuse: bool = True) -> pd.DataFrame:
    out = Path(out_dir)
    data_dir, models_dir = _seed_dirs(out, seed)
    gen_cfg = replace(cfg.generation, seed=seed)
    if not (
        reuse
        and (data_dir / "metadata.json").exists()
        and json.loads((data_dir / "metadata.json").read_text())["config"] == _jsonable(gen_cfg)
    ):
        log.info("[seed %d] generating %d changes", seed, gen_cfg.n_changes)
        generate_history(gen_cfg, data_dir)
    executions = pd.read_csv(data_dir / "executions.csv")
    changes = load_changes(data_dir / "changes.jsonl")
    detections_df = pd.read_csv(data_dir / "fault_detection.csv")
    train_cfg = replace(cfg.training, seed=seed)
    log.info("[seed %d] training models", seed)
    card = train_models(executions, train_cfg, models_dir)
    predictor = FailurePredictor.load(models_dir)
    cutoff = card["cutoff_change_id"]

    state = replay_history(executions, changes, upto_change=cutoff)
    by_id = catalog_by_id()
    by_change = {cid: g for cid, g in executions.groupby("change_id")}
    det_by_change: dict[int, dict[str, set[str]]] = {}
    for r in detections_df.itertuples():
        det_by_change.setdefault(int(r.change_id), {}).setdefault(r.fault_id, set()).add(r.test_id)

    test_changes = [c for c in changes if c["visible"]["change_id"] >= cutoff]
    if cfg.benchmark.max_test_changes:
        test_changes = test_changes[: cfg.benchmark.max_test_changes]
    rows: list[dict] = []
    t0 = time.perf_counter()
    for k, row in enumerate(test_changes):
        ctx = ChangeContext.from_dict(row["visible"])
        g = by_change[ctx.change_id]
        catalog = [by_id[t] for t in g["test_id"]]
        failed = dict(zip(g["test_id"], g["failed"].astype(bool)))
        wall = dict(zip(g["test_id"], g["wall_s"].astype(float)))
        detections = {f: set() for f in row["ground_truth"]["active_faults"]}
        detections.update(det_by_change.get(ctx.change_id, {}))
        base = {
            "seed": seed,
            "n_new_faults": len(row["ground_truth"]["new_faults"]),
            "n_carried_faults": len(row["ground_truth"]["carried_faults"]),
        }
        for strat_name in cfg.benchmark.strategies:
            budgets = [1.0] if strat_name == "full" else cfg.benchmark.budgets
            reps = cfg.benchmark.random_repeats if strat_name == "random" else 1
            for b in budgets:
                for rep in range(reps):
                    policy = replace(cfg.selection, budget_fraction=b, budget_seconds=None, random_seed=rep)
                    strategy = make_strategy(strat_name, policy, predictor)
                    res = run_selection(strategy, ctx, catalog, state, policy)
                    m = evaluate_selection(res, ctx, catalog, failed, detections, wall)
                    m.update(base, budget_fraction=b, repeat=rep)
                    rows.append(m)
        state.update(ctx, failed)
        if (k + 1) % 20 == 0:
            log.info(
                "[seed %d] replayed %d/%d held-out changes (%.1fs)",
                seed,
                k + 1,
                len(test_changes),
                time.perf_counter() - t0,
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / f"seed_{seed}" / "selection_metrics.csv", index=False)
    return df


def _jsonable(obj) -> dict:
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(obj)))


def _seed_worker(args: tuple[ExperimentConfig, int, str, bool]) -> pd.DataFrame:
    from cellselect.logging_utils import setup_logging

    setup_logging()
    cfg, seed, out, reuse = args
    return run_seed(cfg, seed, out, reuse)


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool per (seed, strategy, budget), then mean/std across seeds."""
    per_seed = []
    for (seed, strat, budget), g in df.groupby(["seed", "strategy", "budget_fraction"]):
        faulty = g[g["faults_detected_full"] > 0]
        ttfd = g["time_to_first_detection_s"].dropna()
        det_full = g["faults_detected_full"].sum()
        per_seed.append(
            {
                "seed": seed,
                "strategy": strat,
                "budget_fraction": budget,
                "scenarios": g["change_id"].nunique(),
                "faulty_scenarios": faulty["change_id"].nunique(),
                "tests_selected_mean": g["n_tests_selected"].mean(),
                "tests_full": g["n_tests_full"].mean(),
                "full_time_s_mean": g["full_time_s"].mean(),
                "selected_time_s_mean": g["selected_time_s"].mean(),
                "runtime_reduction_mean": g["runtime_reduction"].mean(),
                "selection_overhead_s_mean": g["selection_overhead_s"].mean(),
                "selection_overhead_s_p95": g["selection_overhead_s"].quantile(0.95),
                "e2e_runtime_reduction_mean": g["e2e_runtime_reduction"].mean(),
                "coverage_retained_mean": g["coverage_retained"].mean(),
                "changed_component_coverage_mean": g["changed_component_coverage"].mean(),
                "faults_detected_full_total": det_full / g["repeat"].nunique(),
                "faults_detected_selected_total": g["faults_detected_selected"].sum() / g["repeat"].nunique(),
                "faults_missed_total": g["faults_missed"].sum() / g["repeat"].nunique(),
                "fault_retention_pooled": (g["faults_detected_selected"].sum() / det_full)
                if det_full
                else np.nan,
                "fault_retention_scenario_mean": faulty["fault_retention"].mean(),
                "all_faults_caught_rate": (faulty["faults_missed"] == 0).mean() if len(faulty) else np.nan,
                "faulty_change_flagged_rate": faulty["change_flagged_selected"].mean()
                if len(faulty)
                else np.nan,
                "failing_test_recall": (g["failing_tests_selected"].sum() / g["failing_tests_full"].sum())
                if g["failing_tests_full"].sum()
                else np.nan,
                "time_to_first_detection_s_mean": ttfd.mean() if len(ttfd) else np.nan,
                "budget_overrun_rate": g["budget_overrun"].mean(),
                "fallback_rate": (g["n_fallbacks"] > 0).mean(),
                "selected_wall_s_mean": g["selected_wall_s"].mean() if "selected_wall_s" in g else np.nan,
                "full_wall_s_mean": g["full_wall_s"].mean() if "full_wall_s" in g else np.nan,
            }
        )
    ps = pd.DataFrame(per_seed)
    metrics = [c for c in ps.columns if c not in ("seed", "strategy", "budget_fraction")]
    agg = ps.groupby(["strategy", "budget_fraction"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = ps.groupby(["strategy", "budget_fraction"])["seed"].nunique().to_numpy()
    return ps, agg


def run_benchmark(cfg: ExperimentConfig, out_dir: str | Path, reuse: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    t0 = time.perf_counter()
    if cfg.jobs > 1 and len(cfg.seeds) > 1:
        with ProcessPoolExecutor(max_workers=min(cfg.jobs, len(cfg.seeds))) as ex:
            frames = list(ex.map(_seed_worker, [(cfg, s, str(out), reuse) for s in cfg.seeds]))
    else:
        frames = [run_seed(cfg, s, out, reuse) for s in cfg.seeds]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "selection_metrics_all.csv", index=False)
    per_seed, agg = aggregate(df)
    per_seed.to_csv(out / "summary_per_seed.csv", index=False)
    agg.to_csv(out / "summary.csv", index=False)
    cards = {s: json.loads((_seed_dirs(out, s)[1] / "model_card.json").read_text()) for s in cfg.seeds}
    ml = _ml_summary(cards)
    summary = {
        "synthetic_data": True,
        "seeds": cfg.seeds,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "held_out_scenarios_per_seed": {int(s): int(g["change_id"].nunique()) for s, g in df.groupby("seed")},
        "ml_prediction": ml,
        "strategies": json.loads(agg.to_json(orient="records")),
    }
    (out / "benchmark_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("benchmark finished in %.1fs -> %s", summary["elapsed_s"], out)
    return summary


def _ml_summary(cards: dict[int, dict]) -> dict:
    names = sorted({n for c in cards.values() for n in c["test"]})
    out: dict = {
        "best_model_per_seed": {str(s): c["best_model"] for s, c in cards.items()},
        "test_metrics": {},
    }
    for n in names:
        vals = {
            k: [c["test"][n][k] for c in cards.values() if c["test"].get(n) and c["test"][n][k] is not None]
            for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier")
        }
        out["test_metrics"][n] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in vals.items() if v
        }
        cms = [c["test"][n]["confusion_matrix"] for c in cards.values() if c["test"].get(n)]
        out["test_metrics"][n]["confusion_matrix_sum"] = {
            k: int(sum(cm[k] for cm in cms)) for k in ("tn", "fp", "fn", "tp")
        }
    return out
