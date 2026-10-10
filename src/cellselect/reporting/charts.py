"""Matplotlib charts for benchmark and selection reports (PNG, light surface)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"
CRITICAL = "#d03b3b"
# Categorical slots in fixed order (validated palette): ML gets slot 1.
STRATEGY_COLORS = {
    "ml": "#2a78d6",
    "coverage": "#eb6834",
    "history": "#1baf7a",
    "random": "#eda100",
    "full": NEUTRAL,
}
ABLATION_COLORS = {"ml": "#2a78d6", "ml_nocov": "#e87ba4", "ml_rank": "#008300"}
STRATEGY_LABELS = {
    "ml_nocov": "ML risk, no coverage term/guard",
    "ml_rank": "ML probability ranking only",
    "ml": "ML risk + coverage",
    "coverage": "Coverage greedy",
    "history": "Historical failure rate",
    "random": "Random",
    "full": "Full suite",
}
MODEL_COLORS = {
    "random_forest": "#2a78d6",
    "gradient_boosting": "#eb6834",
    "logistic_regression": "#1baf7a",
    "heuristic_last_failed": "#eda100",
    "heuristic_change_overlap": "#e87ba4",
    "heuristic_hist_fail_rate": "#008300",
}
ORDER = ["ml", "coverage", "history", "random"]


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def _fig(w: float = 7.5, h: float = 4.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _strategy_lines(
    ax, agg: pd.DataFrame, metric: str, full_value: float | None = None, pct: bool = True
) -> None:
    for s in ORDER:
        g = agg[agg.strategy == s].sort_values("budget_fraction")
        if g.empty:
            continue
        x = g.budget_fraction * 100
        y = g[f"{metric}__mean"]
        err = g[f"{metric}__std"].fillna(0)
        ax.errorbar(
            x,
            y * (100 if pct else 1),
            yerr=err * (100 if pct else 1),
            color=STRATEGY_COLORS[s],
            lw=2,
            marker="o",
            ms=6,
            capsize=3,
            label=STRATEGY_LABELS[s],
            mec=SURFACE,
            mew=1.5,
        )
    if full_value is not None:
        ax.axhline(full_value * (100 if pct else 1), color=NEUTRAL, ls="--", lw=1.5, label="Full suite")


def plot_retention_vs_budget(agg: pd.DataFrame, path: Path) -> Path:
    fig, ax = _fig()
    _strategy_lines(ax, agg, "fault_retention_pooled", full_value=1.0)
    _style(
        ax,
        "Fault-detection retention vs. budget",
        "Budget (% of full-suite lab time)",
        "Faults detected / full-suite detected (%)",
    )
    ax.set_xlim(left=0, right=agg[agg.strategy != "full"].budget_fraction.max() * 100 + 5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_retention_vs_runtime(agg: pd.DataFrame, path: Path) -> Path:
    fig, ax = _fig()
    for s in ORDER + ["full"]:
        g = agg[agg.strategy == s].sort_values("budget_fraction")
        if g.empty:
            continue
        x = (1 - g["e2e_runtime_reduction_mean__mean"]) * 100
        ax.plot(
            x,
            g["fault_retention_pooled__mean"] * 100,
            color=STRATEGY_COLORS[s],
            lw=2,
            marker="o",
            ms=7,
            label=STRATEGY_LABELS[s],
            mec=SURFACE,
            mew=1.5,
        )
    _style(
        ax,
        "Detection vs. end-to-end cost (incl. selection overhead)",
        "Executed lab time + selection overhead (% of full suite)",
        "Fault-detection retention (%)",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_missed_faults(agg: pd.DataFrame, path: Path) -> Path:
    fig, ax = _fig()
    budgets = sorted(agg[agg.strategy != "full"].budget_fraction.unique())
    width = 0.8 / len(ORDER)
    for i, s in enumerate(ORDER):
        g = agg[agg.strategy == s].set_index("budget_fraction")
        vals = [g.loc[b, "faults_missed_total__mean"] if b in g.index else np.nan for b in budgets]
        xs = np.arange(len(budgets)) + (i - (len(ORDER) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.92, color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
    ax.set_xticks(np.arange(len(budgets)), [f"{b:.0%}" for b in budgets])
    _style(
        ax,
        "Missed faults (absolute, mean per seed)",
        "Budget (% of full-suite lab time)",
        "Faults missed vs. full suite",
    )
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_coverage(agg: pd.DataFrame, path: Path) -> Path:
    fig, ax = _fig()
    _strategy_lines(ax, agg, "coverage_retained_mean", full_value=1.0)
    _style(
        ax, "Coverage retained vs. budget", "Budget (% of full-suite lab time)", "Coverage items retained (%)"
    )
    ax.set_xlim(left=0, right=agg[agg.strategy != "full"].budget_fraction.max() * 100 + 5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_time_to_first_detection(agg: pd.DataFrame, path: Path) -> Path:
    fig, ax = _fig()
    _strategy_lines(ax, agg, "time_to_first_detection_s_mean", pct=False)
    full = agg[agg.strategy == "full"]
    if not full.empty:
        ax.axhline(
            full["time_to_first_detection_s_mean__mean"].iloc[0],
            color=NEUTRAL,
            ls="--",
            lw=1.5,
            label="Full suite (catalog order)",
        )
    _style(
        ax,
        "Time to first fault-revealing failure (lower is better)",
        "Budget (% of full-suite lab time)",
        "Nominal lab seconds",
    )
    ax.set_xlim(left=0, right=agg[agg.strategy != "full"].budget_fraction.max() * 100 + 5)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def plot_pr_curves(preds: pd.DataFrame, path: Path, names: list[str]) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = _fig(7.5, 4.8)
    y = preds["failed"].to_numpy()
    for n in names:
        col = f"p_{n}"
        if col not in preds:
            continue
        prec, rec, _ = precision_recall_curve(y, preds[col])
        ap = average_precision_score(y, preds[col])
        dashed = n.startswith("heuristic")
        ax.plot(
            rec,
            prec,
            color=MODEL_COLORS.get(n, NEUTRAL),
            lw=1.6 if dashed else 2.2,
            ls="--" if dashed else "-",
            label=f"{n.replace('_', ' ')} (PR-AUC {ap:.3f})",
        )
    ax.axhline(y.mean(), color=NEUTRAL, lw=1, ls=":", label=f"base rate {y.mean():.3f}")
    _style(ax, "Precision-recall on held-out changes", "Recall", "Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    return _save(fig, path)


def plot_confusion(cm: dict, title: str, path: Path) -> Path:
    fig, ax = _fig(4.2, 3.8)
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1) * 1.15)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(
            j,
            i,
            f"{v:,}",
            ha="center",
            va="center",
            fontsize=12,
            color=INK if v < mat.max() * 0.6 else "white",
        )
    ax.set_xticks([0, 1], ["pred pass", "pred fail"])
    ax.set_yticks([0, 1], ["actual pass", "actual fail"])
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.tick_params(colors=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, path)


def plot_feature_importance(imp: pd.DataFrame, path: Path, top: int = 15) -> Path:
    imp = imp.sort_values("importance_mean", ascending=True).tail(top)
    fig, ax = _fig(7.5, 0.32 * len(imp) + 1.2)
    ax.barh(
        imp["feature"],
        imp["importance_mean"],
        xerr=imp["importance_std"],
        color=STRATEGY_COLORS["ml"],
        height=0.6,
        error_kw={"ecolor": INK_2, "lw": 1},
    )
    _style(ax, "Permutation importance (drop in PR-AUC)", "Mean decrease in average precision", "")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_selection_risk(decisions: list[dict], failed: set[str], path: Path, top: int = 30) -> Path:
    """Predicted risk (bars) vs. confirmed failures (marker + label) for the highest-risk tests."""
    rows = [d for d in decisions if d.get("predicted_p") is not None]
    rows.sort(key=lambda d: -d["predicted_p"])
    rows = rows[:top][::-1]
    fig, ax = _fig(7.8, 0.26 * len(rows) + 1.4)
    names = [d["test_id"] for d in rows]
    vals = [d["predicted_p"] for d in rows]
    colors = [STRATEGY_COLORS["ml"] if d["selected"] else "#b9c8dc" for d in rows]
    ax.barh(names, vals, color=colors, height=0.62)
    for i, d in enumerate(rows):
        if d["test_id"] in failed:
            ax.annotate(
                "✕ FAILED" if d["selected"] else "✕ FAILED (not run)",
                (vals[i], i),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
                color=CRITICAL,
                fontweight="bold",
            )
    _style(
        ax,
        "Predicted failure risk (bars) vs. confirmed failures (✕)",
        "Predicted probability of failure  (dark = selected, light = excluded)",
        "",
    )
    ax.set_xlim(0, max(max(vals, default=0.1) * 1.35, 0.1))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    return _save(fig, path)


def plot_strategy_comparison(rows: list[dict], path: Path) -> Path:
    """Single-scenario comparison: runtime fraction and faults detected per strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    names = [r["strategy"] for r in rows]
    colors = [STRATEGY_COLORS.get(n, NEUTRAL) for n in names]
    labels = [STRATEGY_LABELS.get(n, n) for n in names]
    axes[0].barh(labels, [r["selected_fraction"] * 100 for r in rows], color=colors, height=0.6)
    _style(axes[0], "Lab time executed", "% of full suite", "")
    axes[1].barh(labels, [r["faults_detected_selected"] for r in rows], color=colors, height=0.6)
    full_det = rows[0]["faults_detected_full"] if rows else 0
    axes[1].axvline(full_det, color=NEUTRAL, ls="--", lw=1.4)
    _style(axes[1], f"Faults detected (full suite: {full_det})", "injected faults", "")
    axes[1].xaxis.get_major_locator().set_params(integer=True)
    axes[1].set_yticklabels([])
    for ax in axes:
        ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_ablation(agg: pd.DataFrame, path: Path) -> Path:
    """Which ingredient of the ML strategy produces the gain? (history baseline for reference)."""
    fig, ax = _fig()
    for s, color in list(ABLATION_COLORS.items()) + [("history", STRATEGY_COLORS["history"])]:
        g = agg[agg.strategy == s].sort_values("budget_fraction")
        if g.empty:
            continue
        ax.errorbar(
            g.budget_fraction * 100,
            g["fault_retention_pooled__mean"] * 100,
            yerr=g["fault_retention_pooled__std"].fillna(0) * 100,
            color=color,
            lw=2,
            marker="o",
            ms=6,
            capsize=3,
            label=STRATEGY_LABELS[s],
            mec=SURFACE,
            mew=1.5,
            ls="--" if s == "history" else "-",
        )
    _style(
        ax,
        "Ablation: fault retention of ML-strategy variants",
        "Budget (% of full-suite lab time)",
        "Fault-detection retention (%)",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)
