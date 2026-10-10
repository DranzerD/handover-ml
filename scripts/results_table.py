"""Render README-ready markdown tables from a benchmark directory (numbers copied from measured output).

Usage: python scripts/results_table.py [results/benchmark]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

LABELS = {
    "full": "Full suite",
    "random": "Random",
    "history": "Historical failure rate",
    "coverage": "Coverage greedy (change-aware)",
    "ml": "**ML risk + coverage (CellSelect)**",
    "ml_nocov": "ablation: ML, no coverage term/guard",
    "ml_rank": "ablation: ML probability ranking only",
}
ORDER = list(LABELS)


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.1%}"


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark")
    agg = pd.read_csv(d / "summary.csv")
    res = json.loads((d / "benchmark_results.json").read_text())
    agg["order"] = agg.strategy.map(ORDER.index)
    print(
        "| Strategy | Budget | Tests run | Runtime reduction (e2e) | Selection overhead | Fault retention (± std over seeds) "
        "| Faults missed / detected by full (per seed) | Coverage retained | Time to first fault-revealing failure |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for _, r in agg.sort_values(["budget_fraction", "order"]).iterrows():
        if r.strategy == "full":
            continue
        print(
            f"| {LABELS[r.strategy]} | {r.budget_fraction:.0%} | {r['tests_selected_mean__mean']:.1f} "
            f"| {pct(r['e2e_runtime_reduction_mean__mean'])} | {r['selection_overhead_s_mean__mean'] * 1000:.0f} ms "
            f"| {pct(r['fault_retention_pooled__mean'])} ± {r['fault_retention_pooled__std']:.1%} "
            f"| {r['faults_missed_total__mean']:.1f} / {r['faults_detected_full_total__mean']:.1f} "
            f"| {pct(r['coverage_retained_mean__mean'])} | {r['time_to_first_detection_s_mean__mean']:.0f} s |"
        )
    full = agg[agg.strategy == "full"].iloc[0]
    print(
        f"\nFull suite: {full['tests_selected_mean__mean']:.0f} tests, {full['full_time_s_mean__mean']:.0f} s "
        f"({full['full_time_s_mean__mean'] / 3600:.2f} h) nominal lab time, time to first fault-revealing failure "
        f"{full['time_to_first_detection_s_mean__mean']:.0f} s (catalog order)."
    )
    print(
        f"Held-out scenarios per seed: {res['held_out_scenarios_per_seed']}; elapsed {res['elapsed_s']} s.\n"
    )
    print("| Model (held-out period) | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN | TN |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, m in res["ml_prediction"]["test_metrics"].items():
        cm = m["confusion_matrix_sum"]
        print(
            f"| {name} | {m['pr_auc']['mean']:.3f} ± {m['pr_auc']['std']:.3f} | {m['roc_auc']['mean']:.3f} "
            f"| {m['precision']['mean']:.3f} | {m['recall']['mean']:.3f} | {m['f1']['mean']:.3f} "
            f"| {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} |"
        )
    print(f"\nBest model per seed: {res['ml_prediction']['best_model_per_seed']}")


if __name__ == "__main__":
    main()
