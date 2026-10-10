"""Selection engine: five competing strategies under a shared execution budget.

All strategies share the same contract:

* input: the *visible* change context, the active catalog, optional history features, a budget
  (nominal lab seconds),
* output: an ordered list of selected tests (execution priority) plus a decision record with
  human-readable reasons for every test, selected or not, and any warnings / fallbacks.

System-level rule enforced for every budgeted strategy: tests marked ``critical`` are always
selected first. If they alone exceed the budget they still run and a warning is raised.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from cellselect.data.features import HistoryState
from cellselect.data.scenarios import CHANGE_AREAS, CHANGE_TYPES, ChangeContext
from cellselect.ml.predictor import FailurePredictor, ModelUnavailableError
from cellselect.sim.faults import COMPONENT_NEIGHBOURS
from cellselect.suite.catalog import TestSpec

log = logging.getLogger(__name__)

STRATEGY_NAMES: tuple[str, ...] = ("full", "random", "history", "coverage", "ml")
# Ablations of the ML strategy (benchmark only): which ingredient produces the gain?
ABLATION_NAMES: tuple[str, ...] = ("ml_nocov", "ml_rank")


@dataclass
class SelectionPolicy:
    budget_fraction: float | None = 0.3
    budget_seconds: float | None = None
    include_critical: bool = True
    min_history: int = 10
    coverage_guard: bool = True
    max_overrun_fraction: float = 0.10
    uncertainty_std: float = 0.30
    ucb_k: float = 0.5
    coverage_weight: float = 0.4
    changed_item_weight: float = 5.0
    neighbour_item_weight: float = 2.0
    random_seed: int = 0

    def budget_for(self, catalog: Sequence[TestSpec]) -> float:
        full = float(sum(s.duration_s for s in catalog))
        if self.budget_seconds is not None:
            if self.budget_seconds <= 0:
                raise ValueError("budget_seconds must be positive")
            return float(self.budget_seconds)
        if self.budget_fraction is None or not 0 < self.budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1]")
        return full * self.budget_fraction


@dataclass
class TestDecision:
    __test__ = False

    test_id: str
    selected: bool
    duration_s: float
    category: str
    tags: list[str]
    critical: bool
    reasons: list[str] = field(default_factory=list)
    rank: int | None = None
    predicted_p: float | None = None
    uncertainty: float | None = None
    score: float | None = None


@dataclass
class SelectionResult:
    strategy: str
    change_id: int
    budget_s: float
    full_duration_s: float
    selected: list[str]
    decisions: dict[str, TestDecision]
    warnings: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    overhead_s: float = 0.0
    effective_strategy: str = ""
    model_name: str | None = None

    @property
    def selected_duration_s(self) -> float:
        return float(sum(self.decisions[t].duration_s for t in self.selected))

    def coverage_retained(self, catalog: Sequence[TestSpec]) -> float:
        all_items = set().union(*(s.coverage_items for s in catalog))
        by_id = {s.test_id: s for s in catalog}
        sel = set().union(*(by_id[t].coverage_items for t in self.selected)) if self.selected else set()
        return len(sel) / len(all_items) if all_items else 1.0

    def changed_component_coverage(self, change: ChangeContext) -> float:
        comps = set(change.changed_components)
        if not comps:
            return 1.0
        covered = {c for t in self.selected for c in self.decisions[t].tags if c in comps}
        return len(covered) / len(comps)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "effective_strategy": self.effective_strategy or self.strategy,
            "model_name": self.model_name,
            "change_id": self.change_id,
            "budget_s": self.budget_s,
            "full_duration_s": self.full_duration_s,
            "selected_duration_s": self.selected_duration_s,
            "selection_overhead_s": self.overhead_s,
            "n_selected": len(self.selected),
            "n_total": len(self.decisions),
            "selected": self.selected,
            "warnings": self.warnings,
            "fallbacks": self.fallbacks,
            "decisions": [
                asdict(d)
                for d in sorted(
                    self.decisions.values(), key=lambda d: (d.rank is None, d.rank or 0, d.test_id)
                )
            ],
        }


@dataclass
class SelectionRequest:
    change: ChangeContext
    catalog: list[TestSpec]
    budget_s: float
    features: pd.DataFrame | None = None  # indexed by test_id

    def feature(self, col: str) -> dict[str, float]:
        if self.features is None:
            raise ValueError(f"strategy needs history features ({col})")
        return self.features[col].astype(float).to_dict()


# ------------------------------------------------------------------------------------- base
class _Builder:
    """Accumulates the selection while tracking budget and reasons."""

    def __init__(self, req: SelectionRequest, strategy: str) -> None:
        self.req = req
        self.by_id = {s.test_id: s for s in req.catalog}
        self.order: list[str] = []
        self.used = 0.0
        self.decisions = {
            s.test_id: TestDecision(s.test_id, False, s.duration_s, s.category, list(s.tags), s.critical)
            for s in req.catalog
        }
        self.warnings: list[str] = []
        self.fallbacks: list[str] = []
        self.strategy = strategy

    @property
    def remaining(self) -> float:
        return self.req.budget_s - self.used

    def fits(self, tid: str, extra: float = 0.0) -> bool:
        return self.by_id[tid].duration_s <= self.remaining + extra + 1e-9

    def add(self, tid: str, reason: str) -> None:
        if tid in self.order:
            self.decisions[tid].reasons.append(reason)
            return
        self.order.append(tid)
        self.used += self.by_id[tid].duration_s
        d = self.decisions[tid]
        d.selected = True
        d.reasons.append(reason)

    def unselected(self) -> list[str]:
        chosen = set(self.order)
        return [s.test_id for s in self.req.catalog if s.test_id not in chosen]

    def add_critical(self, policy: SelectionPolicy) -> None:
        if not policy.include_critical:
            return
        crit = [s for s in self.req.catalog if s.critical]
        total = sum(s.duration_s for s in crit)
        for s in crit:
            self.add(s.test_id, "mandatory: critical test, never skipped")
        if total > self.req.budget_s:
            self.warnings.append(
                f"budget {self.req.budget_s:.0f}s is smaller than the mandatory critical set ({total:.0f}s); "
                "critical tests run anyway (budget overrun)"
            )

    def result(
        self, overhead: float, priority: dict[str, float] | None = None, model_name: str | None = None
    ) -> SelectionResult:
        crit = [t for t in self.order if self.by_id[t].critical]
        rest = [t for t in self.order if not self.by_id[t].critical]
        if priority is not None:
            rest.sort(key=lambda t: -priority.get(t, 0.0))
        ordered = crit + rest
        for i, t in enumerate(ordered, 1):
            self.decisions[t].rank = i
        for t in self.unselected():
            if not self.decisions[t].reasons:
                self.decisions[t].reasons.append("not selected: budget exhausted")
        effective = self.strategy if not self.fallbacks else f"{self.strategy}->fallback"
        return SelectionResult(
            self.strategy,
            self.req.change.change_id,
            self.req.budget_s,
            float(sum(s.duration_s for s in self.req.catalog)),
            ordered,
            self.decisions,
            self.warnings,
            self.fallbacks,
            overhead,
            effective,
            model_name,
        )


def _coverage_weights(req: SelectionRequest, policy: SelectionPolicy) -> dict[str, float]:
    changed = set(req.change.changed_components)
    neigh = {n for c in changed for n in COMPONENT_NEIGHBOURS.get(c, ())} - changed
    weights: dict[str, float] = {}
    for s in req.catalog:
        for item in s.coverage_items:
            if item in changed:
                w = policy.changed_item_weight
            elif item in neigh:
                w = policy.neighbour_item_weight
            else:
                w = 1.0
            weights[item] = w
    return weights


# ------------------------------------------------------------------------------- strategies
class Strategy:
    name = "base"
    needs_features = False

    def __init__(self, policy: SelectionPolicy) -> None:
        self.policy = policy

    def select(self, req: SelectionRequest) -> SelectionResult:  # pragma: no cover - interface
        raise NotImplementedError


class FullSuite(Strategy):
    name = "full"

    def select(self, req: SelectionRequest) -> SelectionResult:
        b = _Builder(req, self.name)
        req.budget_s = float(sum(s.duration_s for s in req.catalog))
        b.add_critical(self.policy)
        for s in req.catalog:
            b.add(s.test_id, "full-suite baseline: every test runs")
        return b.result(0.0)


class RandomSelection(Strategy):
    name = "random"

    def select(self, req: SelectionRequest) -> SelectionResult:
        b = _Builder(req, self.name)
        b.add_critical(self.policy)
        rng = np.random.default_rng([self.policy.random_seed, req.change.change_id, req.change.seed])
        pool = b.unselected()
        rng.shuffle(pool)
        for tid in pool:
            if b.fits(tid):
                b.add(tid, "random draw within budget")
            else:
                b.decisions[tid].reasons.append("not selected: random draw did not fit remaining budget")
        return b.result(0.0)


class HistoricalFailureRate(Strategy):
    name = "history"
    needs_features = True

    def select(self, req: SelectionRequest) -> SelectionResult:
        b = _Builder(req, self.name)
        b.add_critical(self.policy)
        rate = req.feature("hist_fail_rate")
        pool = sorted(b.unselected(), key=lambda t: (-rate[t], b.by_id[t].duration_s, t))
        for i, tid in enumerate(pool, 1):
            b.decisions[tid].score = rate[tid]
            if b.fits(tid):
                b.add(tid, f"historical failure rate {rate[tid]:.3f} (rank {i})")
            else:
                b.decisions[tid].reasons.append(
                    f"not selected: failure rate {rate[tid]:.3f} (rank {i}), budget exhausted"
                )
        return b.result(0.0, priority=rate)


class CoverageGreedy(Strategy):
    """Change-aware 'additional greedy' coverage: maximise newly covered weighted items per second."""

    name = "coverage"

    def select(self, req: SelectionRequest) -> SelectionResult:
        b = _Builder(req, self.name)
        b.add_critical(self.policy)
        weights = _coverage_weights(req, self.policy)
        covered: set[str] = set().union(*(b.by_id[t].coverage_items for t in b.order)) if b.order else set()
        gain_at_pick: dict[str, float] = {}
        rounds = 1
        while True:
            pool = [t for t in b.unselected() if b.fits(t)]
            if not pool:
                break
            gains = {t: sum(weights[i] for i in b.by_id[t].coverage_items - covered) for t in pool}
            best = max(pool, key=lambda t: (gains[t] / b.by_id[t].duration_s, -b.by_id[t].duration_s, t))
            if gains[best] <= 0:
                covered = set()  # all reachable items covered: reset and continue (additional greedy)
                rounds += 1
                continue
            new = sorted(b.by_id[best].coverage_items - covered)
            changed_new = [i for i in new if i in req.change.changed_components]
            reason = f"coverage round {rounds}: +{len(new)} uncovered items" + (
                f" incl. changed component(s) {', '.join(changed_new)}" if changed_new else ""
            )
            b.add(best, reason)
            gain_at_pick[best] = gains[best] / b.by_id[best].duration_s
            b.decisions[best].score = gain_at_pick[best]
            covered |= b.by_id[best].coverage_items
        for t in b.unselected():
            b.decisions[t].reasons.append(
                "not selected: lower coverage gain per second than selected tests / budget exhausted"
            )
        order_priority = {t: -i for i, t in enumerate(b.order)}
        return b.result(0.0, priority=order_priority)


class MLRiskCoverage(Strategy):
    """ML-predicted failure risk + coverage, cost-aware greedy with safety fallbacks.

    Utility of adding test i to the current selection S:
        value_i = p_i * mean_{c in tags_i} prod_{j in S, c in tags_j} (1 - p_j)   (expected *new* detection)
        gain_i  = coverage_weight * (weighted uncovered coverage items of i) / (total item weight)
        score_i = (value_i + gain_i) / duration_i
    """

    name = "ml"
    needs_features = True

    def __init__(self, policy: SelectionPolicy, predictor: FailurePredictor | None) -> None:
        super().__init__(policy)
        self.predictor = predictor

    def _fallback(self, req: SelectionRequest, why: str) -> SelectionResult:
        log.warning("ML strategy fallback: %s", why)
        res = CoverageGreedy(self.policy).select(req)
        res.strategy = self.name
        res.fallbacks.append(f"{why}; used change-aware coverage-greedy selection instead")
        res.effective_strategy = "ml->coverage_fallback"
        for d in res.decisions.values():
            d.reasons.insert(0, "[safe fallback]")
        return res

    def _utility(
        self,
        spec: TestSpec,
        p: float,
        residual: dict[str, float],
        covered: set[str],
        weights: dict[str, float],
        total_w: float,
    ) -> tuple[float, float, float]:
        """Return (utility per second, expected new detection, coverage gain) for adding ``spec``."""
        value = p * sum(residual.get(c, 1.0) for c in spec.tags) / len(spec.tags)
        gain = self.policy.coverage_weight * sum(weights[i] for i in spec.coverage_items - covered) / total_w
        return (value + gain) / spec.duration_s, value, gain

    def select(self, req: SelectionRequest) -> SelectionResult:
        if self.predictor is None:
            return self._fallback(req, "no trained model available")
        unseen = []
        if (
            req.change.change_type not in self.predictor.seen_change_types
            and self.predictor.seen_change_types
        ):
            unseen.append(f"change type '{req.change.change_type}'")
        if req.change.area not in self.predictor.seen_change_areas and self.predictor.seen_change_areas:
            unseen.append(f"change area '{req.change.area}'")
        if unseen:
            return self._fallback(
                req, f"out-of-distribution input ({', '.join(unseen)} never seen in training)"
            )
        try:
            pred = self.predictor.predict(req.features.loc[[s.test_id for s in req.catalog]].reset_index())
        except Exception as exc:  # noqa: BLE001 - any inference failure must degrade safely
            return self._fallback(req, f"model inference failed ({exc})")

        b = _Builder(req, self.name)
        ids = [s.test_id for s in req.catalog]
        p_raw = dict(zip(ids, pred.probability.astype(float)))
        std = dict(zip(ids, pred.uncertainty.astype(float))) if pred.uncertainty is not None else {}
        p: dict[str, float] = {}
        for t in ids:
            d = b.decisions[t]
            d.predicted_p = p_raw[t]
            d.uncertainty = std.get(t)
            p[t] = p_raw[t]
            if std and std[t] >= self.policy.uncertainty_std:
                p[t] = min(1.0, p_raw[t] + self.policy.ucb_k * std[t])
                d.reasons.append(
                    f"model uncertain (tree std {std[t]:.2f}): risk raised to upper bound {p[t]:.3f}"
                )

        b.add_critical(self.policy)

        # Safe fallback 1: tests with insufficient history always run (model cannot judge them).
        runs = req.feature("hist_n_runs")
        new_tests = sorted(
            (t for t in b.unselected() if runs[t] < self.policy.min_history),
            key=lambda t: b.by_id[t].duration_s,
        )
        if new_tests:
            b.fallbacks.append(
                f"{len(new_tests)} test(s) with < {self.policy.min_history} prior runs forced into selection"
            )
        for t in new_tests:
            if b.fits(t, extra=self.policy.max_overrun_fraction * req.budget_s):
                b.add(
                    t,
                    f"safe fallback: only {int(runs[t])} prior executions (< {self.policy.min_history}); "
                    "insufficient history, run unconditionally",
                )
            else:
                b.warnings.append(f"{t} has insufficient history but does not fit budget + allowed overrun")
                b.decisions[t].reasons.append(
                    "insufficient history but excluded: exceeds budget + allowed overrun (WARNING)"
                )

        # Safe fallback 2: coverage guard - every changed component gets at least one test.
        if self.policy.coverage_guard:
            for comp in req.change.changed_components:
                if any(comp in b.by_id[t].tags for t in b.order):
                    continue
                cands = [t for t in b.unselected() if comp in b.by_id[t].tags]
                if not cands:
                    b.warnings.append(f"no test in the catalog covers changed component {comp}")
                    continue
                cands.sort(key=lambda t: (-(p[t] + 1e-4) / b.by_id[t].duration_s, t))
                fitting = [
                    t for t in cands if b.fits(t, extra=self.policy.max_overrun_fraction * req.budget_s)
                ]
                if fitting:
                    b.add(
                        fitting[0],
                        f"coverage guard: covers changed component {comp} "
                        f"(best risk/cost of {len(cands)} candidates, p={p_raw[fitting[0]]:.3f})",
                    )
                else:
                    b.warnings.append(
                        f"changed component {comp} cannot be covered within budget + "
                        f"{self.policy.max_overrun_fraction:.0%} overrun"
                    )

        # Cost-aware greedy on risk + coverage.
        weights = _coverage_weights(req, self.policy)
        total_w = sum(weights.values()) or 1.0
        covered: set[str] = set().union(*(b.by_id[t].coverage_items for t in b.order)) if b.order else set()
        residual: dict[str, float] = {}
        for t in b.order:
            for c in b.by_id[t].tags:
                residual[c] = residual.get(c, 1.0) * (1.0 - p[t])
        threshold = self.predictor.threshold
        pick_no = 0
        while True:
            pool = [t for t in b.unselected() if b.fits(t)]
            if not pool:
                break

            scored = {t: self._utility(b.by_id[t], p[t], residual, covered, weights, total_w) for t in pool}
            best = max(pool, key=lambda t: (scored[t][0], -b.by_id[t].duration_s, t))
            s_val, value, gain = scored[best]
            pick_no += 1
            level = "high" if p_raw[best] >= threshold else ("moderate" if p_raw[best] >= 0.05 else "low")
            risk = f"p={p_raw[best]:.3f}" + (
                f" (uncertainty-adjusted {p[best]:.3f})" if p[best] != p_raw[best] else ""
            )
            b.add(
                best,
                f"{level} predicted failure risk {risk}; expected new detection {value:.3f}; "
                f"coverage gain {gain:.3f}; utility {s_val * 3600:.3f}/h (greedy pick {pick_no})",
            )
            b.decisions[best].score = s_val
            covered |= b.by_id[best].coverage_items
            for c in b.by_id[best].tags:
                residual[c] = residual.get(c, 1.0) * (1.0 - p[best])
        for t in b.unselected():
            level = "high" if p_raw[t] >= threshold else ("moderate" if p_raw[t] >= 0.05 else "low")
            b.decisions[t].reasons.append(
                f"not selected: {level} predicted risk p={p_raw[t]:.3f}, utility below cut-off / budget exhausted "
                f"(duration {b.by_id[t].duration_s:.0f}s)"
            )
            if p_raw[t] >= threshold:
                b.warnings.append(f"{t} predicted high-risk (p={p_raw[t]:.2f}) but excluded by budget")
        # Execution priority: risk density (probability of failure per second) -> early fault detection.
        priority = {t: p[t] / b.by_id[t].duration_s for t in b.order}
        return b.result(0.0, priority=priority, model_name=self.predictor.name)


class MLRiskRank(Strategy):
    """Ablation: rank purely by predicted probability (no cost, coverage or safety logic)."""

    name = "ml_rank"
    needs_features = True

    def __init__(self, policy: SelectionPolicy, predictor: FailurePredictor | None) -> None:
        super().__init__(policy)
        if predictor is None:
            raise ValueError("ml_rank ablation requires a trained model")
        self.predictor = predictor

    def select(self, req: SelectionRequest) -> SelectionResult:
        b = _Builder(req, self.name)
        ids = [s.test_id for s in req.catalog]
        p = dict(
            zip(ids, self.predictor.predict(req.features.loc[ids].reset_index()).probability.astype(float))
        )
        b.add_critical(self.policy)
        for i, t in enumerate(sorted(b.unselected(), key=lambda t: (-p[t], t)), 1):
            b.decisions[t].predicted_p = p[t]
            if b.fits(t):
                b.add(t, f"predicted failure probability {p[t]:.3f} (rank {i})")
        return b.result(0.0, priority=p, model_name=self.predictor.name)


def make_strategy(name: str, policy: SelectionPolicy, predictor: FailurePredictor | None = None) -> Strategy:
    if name == "full":
        return FullSuite(policy)
    if name == "random":
        return RandomSelection(policy)
    if name == "history":
        return HistoricalFailureRate(policy)
    if name == "coverage":
        return CoverageGreedy(policy)
    if name == "ml":
        return MLRiskCoverage(policy, predictor)
    if name == "ml_nocov":
        strat = MLRiskCoverage(replace(policy, coverage_weight=0.0, coverage_guard=False), predictor)
        strat.name = "ml_nocov"
        return strat
    if name == "ml_rank":
        return MLRiskRank(policy, predictor)
    raise ValueError(f"unknown strategy {name!r}; choose from {STRATEGY_NAMES + ABLATION_NAMES}")


def load_predictor_safely(
    models_dir: str | None, name: str | None = None
) -> tuple[FailurePredictor | None, str | None]:
    if not models_dir:
        return None, "no models directory configured"
    try:
        return FailurePredictor.load(models_dir, name), None
    except ModelUnavailableError as exc:
        return None, str(exc)


def run_selection(
    strategy: Strategy,
    change: ChangeContext,
    catalog: Sequence[TestSpec],
    history: HistoryState | None,
    policy: SelectionPolicy,
) -> SelectionResult:
    """Compute features (if needed), run the strategy and measure end-to-end selection overhead."""
    if change.change_type not in CHANGE_TYPES or change.area not in CHANGE_AREAS:
        log.warning(
            "change %s has unknown type/area (%s/%s)", change.change_id, change.change_type, change.area
        )
    t0 = time.perf_counter()
    features = None
    if strategy.needs_features:
        if history is None:
            raise ValueError(f"strategy {strategy.name} requires execution history")
        features = history.features_for_change(change).set_index("test_id")
    req = SelectionRequest(change, list(catalog), policy.budget_for(catalog), features)
    result = strategy.select(req)
    result.overhead_s = time.perf_counter() - t0
    return result
