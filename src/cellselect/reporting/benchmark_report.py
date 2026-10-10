"""Benchmark HTML report + plots from a benchmark output directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from cellselect.reporting import charts
from cellselect.reporting import html as H

log = logging.getLogger(__name__)


def build_benchmark_plots(bench_dir: str | Path) -> dict[str, Path]:
    d = Path(bench_dir)
    agg = pd.read_csv(d / "summary.csv")
    plots_dir = d / "plots"
    plots = {
        "retention_vs_budget": charts.plot_retention_vs_budget(agg, plots_dir / "retention_vs_budget.png"),
        "retention_vs_runtime": charts.plot_retention_vs_runtime(agg, plots_dir / "retention_vs_runtime.png"),
        "missed_faults": charts.plot_missed_faults(agg, plots_dir / "missed_faults.png"),
        "coverage": charts.plot_coverage(agg, plots_dir / "coverage_vs_budget.png"),
        "ablation": charts.plot_ablation(agg, plots_dir / "ablation.png"),
        "time_to_first_detection": charts.plot_time_to_first_detection(
            agg, plots_dir / "time_to_first_detection.png"
        ),
    }
    seed_dirs = sorted(d.glob("seed_*/models/test_predictions.csv"))
    if seed_dirs:
        first = seed_dirs[0].parent
        card = json.loads((first / "model_card.json").read_text())
        preds = pd.read_csv(first / "test_predictions.csv")
        plots["pr_curves"] = charts.plot_pr_curves(
            preds, plots_dir / "pr_curves_seed0.png", list(card["test"])
        )
        best = card["best_model"]
        plots["confusion"] = charts.plot_confusion(
            card["test"][best]["confusion_matrix"],
            f"{best.replace('_', ' ')} (seed {first.parent.name[5:]})",
            plots_dir / "confusion_best_seed0.png",
        )
    return plots


def _row(agg: pd.DataFrame, strategy: str, budget: float) -> pd.Series | None:
    g = agg[(agg.strategy == strategy) & (agg.budget_fraction.round(4) == round(budget, 4))]
    return None if g.empty else g.iloc[0]


def build_benchmark_report(bench_dir: str | Path) -> Path:
    d = Path(bench_dir)
    results = json.loads((d / "benchmark_results.json").read_text())
    agg = pd.read_csv(d / "summary.csv")
    plots = build_benchmark_plots(d)
    budgets = sorted(agg[agg.strategy != "full"].budget_fraction.unique())
    headline_b = 0.3 if 0.3 in [round(b, 4) for b in budgets] else budgets[len(budgets) // 2]
    ml, cov, full = _row(agg, "ml", headline_b), _row(agg, "coverage", headline_b), _row(agg, "full", 1.0)
    body = []
    body.append(
        '<div class="note"><b>Synthetic evaluation.</b> All numbers come from the CellSelect behavioural '
        "simulator with injected faults (not real RF/modem hardware). Runtime is nominal lab time from the "
        "test catalog; selection overhead is measured wall-clock on this machine.</div>"
    )
    if ml is not None:
        body.append(
            H.tiles(
                [
                    (
                        f"ML fault retention @ {headline_b:.0%} budget",
                        f"{ml['fault_retention_pooled__mean']:.1%}",
                    ),
                    ("ML runtime reduction (end-to-end)", f"{ml['e2e_runtime_reduction_mean__mean']:.1%}"),
                    (
                        "ML faults missed (per seed, mean)",
                        f"{ml['faults_missed_total__mean']:.1f} of {ml['faults_detected_full_total__mean']:.1f}",
                    ),
                    (
                        "ML selection overhead (mean)",
                        f"{ml['selection_overhead_s_mean__mean'] * 1000:.0f} ms",
                    ),
                    (
                        f"Coverage-greedy retention @ {headline_b:.0%}",
                        f"{cov['fault_retention_pooled__mean']:.1%}" if cov is not None else "n/a",
                    ),
                    (
                        "Full-suite lab time (mean)",
                        f"{full['full_time_s_mean__mean'] / 3600:.2f} h" if full is not None else "n/a",
                    ),
                ]
            )
        )
    body.append(
        f"<p class='muted'>Seeds: {results['seeds']}; held-out scenarios per seed: "
        f"{results['held_out_scenarios_per_seed']}; total benchmark time {results['elapsed_s']} s.</p>"
    )

    body.append("<h2>Strategy comparison</h2>")
    rows = []
    for _, r in agg.sort_values(["budget_fraction", "strategy"]).iterrows():
        rows.append(
            [
                r.strategy,
                f"{r.budget_fraction:.0%}",
                r["tests_selected_mean__mean"],
                r["runtime_reduction_mean__mean"],
                r["e2e_runtime_reduction_mean__mean"],
                r["selection_overhead_s_mean__mean"] * 1000,
                r["coverage_retained_mean__mean"],
                r["changed_component_coverage_mean__mean"],
                r["fault_retention_pooled__mean"],
                r["fault_retention_pooled__std"],
                r["faults_missed_total__mean"],
                r["faults_detected_full_total__mean"],
                r["faulty_change_flagged_rate__mean"],
                r["time_to_first_detection_s_mean__mean"],
            ]
        )
    body.append(
        H.table(
            [
                "strategy",
                "budget",
                "tests run",
                "runtime red.",
                "e2e red.",
                "overhead ms",
                "coverage",
                "changed-comp cov.",
                "fault retention",
                "± std (seeds)",
                "faults missed",
                "faults (full)",
                "faulty changes flagged",
                "time to 1st detection s",
            ],
            rows,
        )
    )
    body.append(
        "<p class='muted'>Runtime reduction = (full − selected)/full in nominal lab seconds; e2e adds the measured "
        "selection overhead to the selected time. Fault retention = faults detected by selection / faults detected "
        "by the full suite on the same scenarios (pooled; undefined scenarios with zero detectable faults are "
        "excluded from the denominator). Faults missed are absolute counts per seed (mean over seeds).</p>"
    )
    body.append(
        '<div class="grid2">'
        + "".join(
            H.img(plots[k], k)
            for k in (
                "retention_vs_budget",
                "retention_vs_runtime",
                "missed_faults",
                "coverage",
                "time_to_first_detection",
                "ablation",
            )
            if k in plots
        )
        + "</div>"
    )

    body.append("<h2>Failure-prediction models (held-out temporal test period)</h2>")
    mrows = []
    for name, m in results["ml_prediction"]["test_metrics"].items():
        cm = m["confusion_matrix_sum"]
        mrows.append(
            [
                name,
                m.get("pr_auc", {}).get("mean"),
                m.get("pr_auc", {}).get("std"),
                m.get("roc_auc", {}).get("mean"),
                m.get("precision", {}).get("mean"),
                m.get("recall", {}).get("mean"),
                m.get("f1", {}).get("mean"),
                cm["tp"],
                cm["fp"],
                cm["fn"],
                cm["tn"],
            ]
        )
    body.append(
        H.table(
            ["model", "PR-AUC", "± std", "ROC-AUC", "precision", "recall", "F1", "TP", "FP", "FN", "TN"],
            mrows,
        )
    )
    body.append(
        f"<p class='muted'>Best model per seed (by forward-chaining CV PR-AUC): "
        f"{results['ml_prediction']['best_model_per_seed']}. Thresholds tuned on CV folds only.</p>"
    )
    body.append(
        '<div class="grid2">'
        + "".join(H.img(plots[k], k) for k in ("pr_curves", "confusion") if k in plots)
        + "</div>"
    )
    imp = d / "evaluation" / "feature_importance.png"
    if imp.exists():
        body.append(H.img(imp, "feature importance"))
    out = H.write(
        d / "benchmark_report.html",
        H.page(
            "CellSelect benchmark report",
            "ML-driven cellular test selection vs. baselines (synthetic data)",
            "".join(body),
        ),
    )
    log.info("benchmark report: %s", out)
    return out
