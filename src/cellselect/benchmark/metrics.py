"""Per-scenario selection metrics (runtime, coverage, fault detection)."""

from __future__ import annotations

from collections.abc import Sequence

from cellselect.data.scenarios import ChangeContext
from cellselect.selection.engine import SelectionResult
from cellselect.suite.catalog import TestSpec


def safe_ratio(num: float, den: float) -> float | None:
    """num/den, or None when the denominator is zero (metric undefined, not 0 or 1)."""
    return None if den == 0 else num / den


def runtime_reduction(full_s: float, selected_s: float) -> float | None:
    """(full - selected) / full."""
    return safe_ratio(full_s - selected_s, full_s)


def fault_retention(detected_selected: int, detected_full: int) -> float | None:
    """Faults detected by the selected suite / faults detected by the full suite (same scenario)."""
    return safe_ratio(detected_selected, detected_full)


def time_to_first(order: Sequence[str], durations: dict[str, float], hits: set[str]) -> float | None:
    """Nominal lab seconds elapsed until the first test in ``hits`` completes (None if never)."""
    elapsed = 0.0
    for t in order:
        elapsed += durations[t]
        if t in hits:
            return elapsed
    return None


def evaluate_selection(
    result: SelectionResult,
    change: ChangeContext,
    catalog: Sequence[TestSpec],
    failed: dict[str, bool],
    detections: dict[str, set[str]],
    wall_s: dict[str, float] | None = None,
) -> dict:
    """Score one selection against the full-suite ground truth of the same scenario."""
    selected = set(result.selected)
    durations = {s.test_id: s.duration_s for s in catalog}
    full_s = float(sum(durations.values()))
    sel_s = float(sum(durations[t] for t in selected))
    detectable = {f: ts for f, ts in detections.items() if ts}
    caught = {f for f, ts in detectable.items() if ts & selected}
    detecting_tests = set().union(*detectable.values()) if detectable else set()
    failing_full = {t for t, f in failed.items() if f}
    row = {
        "change_id": change.change_id,
        "strategy": result.strategy,
        "effective_strategy": result.effective_strategy or result.strategy,
        "budget_s": result.budget_s,
        "n_tests_full": len(catalog),
        "n_tests_selected": len(selected),
        "full_time_s": full_s,
        "selected_time_s": sel_s,
        "selected_fraction": sel_s / full_s if full_s else 0.0,
        "runtime_reduction": runtime_reduction(full_s, sel_s),
        "selection_overhead_s": result.overhead_s,
        "e2e_runtime_reduction": runtime_reduction(full_s, sel_s + result.overhead_s),
        "budget_overrun": sel_s > result.budget_s + 1e-6,
        "coverage_retained": result.coverage_retained(catalog),
        "changed_component_coverage": result.changed_component_coverage(change),
        "faults_active": len(detections),
        "faults_detected_full": len(detectable),
        "faults_detected_selected": len(caught),
        "faults_missed": len(detectable) - len(caught),
        "fault_retention": fault_retention(len(caught), len(detectable)),
        "failing_tests_full": len(failing_full),
        "failing_tests_selected": len(failing_full & selected),
        "detecting_tests_full": len(detecting_tests),
        "detecting_tests_selected": len(detecting_tests & selected),
        "change_flagged_full": bool(failing_full),
        "change_flagged_selected": bool(failing_full & selected),
        "time_to_first_detection_s": time_to_first(result.selected, durations, detecting_tests),
        "n_warnings": len(result.warnings),
        "n_fallbacks": len(result.fallbacks),
    }
    if wall_s is not None:
        row["full_wall_s"] = float(sum(wall_s.values()))
        row["selected_wall_s"] = float(sum(wall_s[t] for t in selected))
    return row
