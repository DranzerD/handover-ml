"""Selection engine contracts: budgets, mandatory tests, fallbacks, explanations."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from cellselect.data.generate import load_changes, replay_history
from cellselect.data.scenarios import ChangeContext
from cellselect.ml.predictor import FailurePredictor
from cellselect.selection.engine import SelectionPolicy, make_strategy, run_selection
from cellselect.suite.catalog import build_catalog


@pytest.fixture(scope="module")
def setup(small_dataset, small_models):
    executions = pd.read_csv(small_dataset / "executions.csv")
    changes = load_changes(small_dataset / "changes.jsonl")
    state = replay_history(executions, changes, upto_change=40)
    ctx = ChangeContext.from_dict(changes[40]["visible"])
    return state, ctx, FailurePredictor.load(small_models)


@pytest.mark.parametrize("name", ["random", "history", "coverage", "ml", "ml_nocov", "ml_rank"])
def test_budget_respected_and_critical_always_selected(setup, name) -> None:
    state, ctx, pred = setup
    policy = SelectionPolicy(budget_fraction=0.2, max_overrun_fraction=0.0, min_history=0)
    res = run_selection(make_strategy(name, policy, pred), ctx, build_catalog(), state, policy)
    crit = {s.test_id for s in build_catalog() if s.critical}
    assert crit <= set(res.selected)
    assert res.selected_duration_s <= res.budget_s + 1e-6
    assert all(d.reasons for d in res.decisions.values())
    assert res.selected[: len(crit)] == [t for t in res.selected if t in crit]  # critical first
    assert res.overhead_s > 0


def test_tiny_budget_still_runs_critical_with_warning(setup) -> None:
    state, ctx, pred = setup
    policy = SelectionPolicy(budget_seconds=30)
    res = run_selection(make_strategy("ml", policy, pred), ctx, build_catalog(), state, policy)
    assert {s.test_id for s in build_catalog() if s.critical} <= set(res.selected)
    assert any("critical" in w for w in res.warnings)


def test_ml_without_model_falls_back_to_coverage(setup) -> None:
    state, ctx, _ = setup
    policy = SelectionPolicy(budget_fraction=0.2)
    res = run_selection(make_strategy("ml", policy, None), ctx, build_catalog(), state, policy)
    assert res.fallbacks and "no trained model" in res.fallbacks[0]
    assert res.effective_strategy == "ml->coverage_fallback"


def test_ml_out_of_distribution_change_falls_back(setup) -> None:
    state, ctx, pred = setup
    pred.card["seen_change_areas"] = ["phy"]
    try:
        odd = replace(ctx, area="mobility")
        policy = SelectionPolicy(budget_fraction=0.2)
        res = run_selection(make_strategy("ml", policy, pred), odd, build_catalog(), state, policy)
        assert any("out-of-distribution" in f for f in res.fallbacks)
    finally:
        pred.card["seen_change_areas"] = sorted(
            {"phy", "mac", "transport", "rrc", "mobility", "session", "power", "config"}
        )


def test_ml_forces_tests_with_insufficient_history(setup) -> None:
    state, ctx, pred = setup
    policy = SelectionPolicy(budget_fraction=0.3, min_history=10)
    res = run_selection(make_strategy("ml", policy, pred), ctx, build_catalog(), state, policy)
    new = [t for t in res.decisions if t.startswith("HO-PP")]
    assert new and all(res.decisions[t].selected for t in new)
    assert any("insufficient history" in r for t in new for r in res.decisions[t].reasons)


def test_ml_coverage_guard_covers_changed_components(setup) -> None:
    state, ctx, pred = setup
    policy = SelectionPolicy(budget_fraction=0.1)
    res = run_selection(make_strategy("ml", policy, pred), ctx, build_catalog(), state, policy)
    assert res.changed_component_coverage(ctx) == 1.0
    assert all(d.predicted_p is not None for d in res.decisions.values())


def test_random_strategy_is_reproducible(setup) -> None:
    state, ctx, pred = setup
    policy = SelectionPolicy(budget_fraction=0.2, random_seed=3)
    a = run_selection(make_strategy("random", policy, pred), ctx, build_catalog(), state, policy)
    b = run_selection(make_strategy("random", policy, pred), ctx, build_catalog(), state, policy)
    assert a.selected == b.selected


def test_full_suite_selects_everything(setup) -> None:
    state, ctx, pred = setup
    policy = SelectionPolicy(budget_fraction=0.1)
    res = run_selection(make_strategy("full", policy, pred), ctx, build_catalog(), state, policy)
    assert len(res.selected) == len(build_catalog())
