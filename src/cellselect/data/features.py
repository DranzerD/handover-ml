"""Leakage-safe feature engineering.

``HistoryState`` is an append-only record of past executions. ``features_for_change`` computes one
feature row per test for the *current* change using only (a) static test metadata, (b) the
visible change context and (c) outcomes of strictly earlier changes. Outcomes of the current
change are added via ``update`` only *after* features have been extracted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cellselect.data.scenarios import ChangeContext
from cellselect.sim.faults import COMPONENT_NEIGHBOURS, COMPONENTS
from cellselect.suite.catalog import TestSpec

CATEGORICAL_FEATURES: list[str] = [
    "test_category",
    "template",
    "band",
    "change_type",
    "change_area",
    "device",
    "firmware_branch",
]
NUMERIC_FEATURES: list[str] = [
    "duration_log",
    "n_tags",
    "is_critical",
    "overlap_count",
    "overlap_frac",
    "primary_overlap",
    "neighbor_overlap",
    "lines_changed_log",
    "files_changed",
    "n_changed_components",
    "lab_interference_db",
    "temperature_c",
    "hist_n_runs",
    "hist_fail_rate",
    "hist_fail_rate_last10",
    "hist_fail_rate_last25",
    "last_failed",
    "runs_since_fail",
    "flip_rate_last20",
    "hist_overlap_runs",
    "hist_fail_rate_when_overlap",
    "comp_recent_fail_rate",
    "category_prev_fail_frac",
]
FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "failed"
PRIOR_A, PRIOR_B = 0.5, 9.5  # Beta prior for smoothed failure rates (mean 0.05)


class HistoryState:
    """Per-test execution history accumulated change by change (no look-ahead possible)."""

    def __init__(self, catalog: Sequence[TestSpec]) -> None:
        self.catalog = list(catalog)
        self.test_ids = [s.test_id for s in self.catalog]
        self.index = {t: i for i, t in enumerate(self.test_ids)}
        n = len(self.catalog)
        self.comp_index = {c: j for j, c in enumerate(COMPONENTS)}
        self.tag_matrix = np.zeros((n, len(COMPONENTS)), dtype=bool)
        for i, s in enumerate(self.catalog):
            for t in s.tags:
                self.tag_matrix[i, self.comp_index[t]] = True
        self.primary = np.array([self.comp_index[s.tags[0]] for s in self.catalog])
        self.categories = np.array([s.category for s in self.catalog])
        self._fails: list[np.ndarray] = []  # float arrays, nan = not executed
        self._overlap: list[np.ndarray] = []
        self.change_ids: list[int] = []

    @property
    def n_changes(self) -> int:
        return len(self._fails)

    def overlap_vector(self, change: ChangeContext) -> np.ndarray:
        cmask = np.zeros(len(COMPONENTS), dtype=bool)
        for c in change.changed_components:
            cmask[self.comp_index[c]] = True
        return (self.tag_matrix & cmask).sum(axis=1)

    def update(self, change: ChangeContext, failed: dict[str, bool]) -> None:
        """Record outcomes of ``change`` (tests absent from ``failed`` were not executed)."""
        row = np.full(len(self.test_ids), np.nan)
        for tid, f in failed.items():
            row[self.index[tid]] = float(f)
        self._fails.append(row)
        self._overlap.append(self.overlap_vector(change) > 0)
        self.change_ids.append(change.change_id)

    def runs_per_test(self) -> np.ndarray:
        if not self._fails:
            return np.zeros(len(self.test_ids), dtype=int)
        return (~np.isnan(np.vstack(self._fails))).sum(axis=0)

    def features_for_change(self, change: ChangeContext) -> pd.DataFrame:
        n = len(self.test_ids)
        overlap = self.overlap_vector(change)
        n_tags = self.tag_matrix.sum(axis=1)
        changed = np.zeros(len(COMPONENTS), dtype=bool)
        neigh = np.zeros(len(COMPONENTS), dtype=bool)
        for c in change.changed_components:
            changed[self.comp_index[c]] = True
            for nb in COMPONENT_NEIGHBOURS[c]:
                neigh[self.comp_index[nb]] = True
        neigh &= ~changed
        feats: dict[str, np.ndarray | list] = {
            "test_id": self.test_ids,
            "test_category": [s.category for s in self.catalog],
            "template": [s.template for s in self.catalog],
            "band": [s.band for s in self.catalog],
            "change_type": [change.change_type] * n,
            "change_area": [change.area] * n,
            "device": [change.env.device] * n,
            "firmware_branch": [change.env.firmware_branch] * n,
            "duration_log": np.log10([s.duration_s for s in self.catalog]),
            "n_tags": n_tags,
            "is_critical": np.array([s.critical for s in self.catalog], dtype=float),
            "overlap_count": overlap.astype(float),
            "overlap_frac": overlap / n_tags,
            "primary_overlap": changed[self.primary].astype(float),
            "neighbor_overlap": (self.tag_matrix & neigh).sum(axis=1).astype(float),
            "lines_changed_log": np.full(n, np.log10(max(change.lines_changed, 1))),
            "files_changed": np.full(n, float(change.files_changed)),
            "n_changed_components": np.full(n, float(len(change.changed_components))),
            "lab_interference_db": np.full(n, change.env.lab_interference_db),
            "temperature_c": np.full(n, change.env.temperature_c),
        }
        feats.update(self._history_features())
        return pd.DataFrame(feats)

    def _history_features(self) -> dict[str, np.ndarray]:
        n = len(self.test_ids)
        if not self._fails:
            z = np.zeros(n)
            prior = np.full(n, PRIOR_A / (PRIOR_A + PRIOR_B))
            return {
                "hist_n_runs": z,
                "hist_fail_rate": prior,
                "hist_fail_rate_last10": z,
                "hist_fail_rate_last25": z,
                "last_failed": z,
                "runs_since_fail": np.full(n, 100.0),
                "flip_rate_last20": z,
                "hist_overlap_runs": z,
                "hist_fail_rate_when_overlap": prior,
                "comp_recent_fail_rate": z,
                "category_prev_fail_frac": z,
            }
        F = np.vstack(self._fails)  # (t, n) with nan for not executed
        overlap_hist = np.vstack(self._overlap)
        ran = ~np.isnan(F)
        Fz = np.nan_to_num(F, nan=0.0)
        n_runs = ran.sum(axis=0)
        fails = Fz.sum(axis=0)
        out: dict[str, np.ndarray] = {
            "hist_n_runs": n_runs.astype(float),
            "hist_fail_rate": (fails + PRIOR_A) / (n_runs + PRIOR_A + PRIOR_B),
        }
        # Per-test "last k executions" statistics (tests may have skipped changes).
        last10 = np.zeros(n)
        last25 = np.zeros(n)
        last_failed = np.zeros(n)
        since = np.full(n, 100.0)
        flips = np.zeros(n)
        for i in range(n):
            seq = F[ran[:, i], i]
            if seq.size == 0:
                continue
            last10[i] = seq[-10:].mean()
            last25[i] = seq[-25:].mean()
            last_failed[i] = seq[-1]
            idx = np.flatnonzero(seq)
            since[i] = min(100.0, float(seq.size - 1 - idx[-1])) if idx.size else 100.0
            tail = seq[-20:]
            flips[i] = np.abs(np.diff(tail)).sum() / max(tail.size - 1, 1)
        out.update(
            hist_fail_rate_last10=last10,
            hist_fail_rate_last25=last25,
            last_failed=last_failed,
            runs_since_fail=since,
            flip_rate_last20=flips,
        )
        ov_runs = (overlap_hist & ran).sum(axis=0)
        ov_fails = (Fz * (overlap_hist & ran)).sum(axis=0)
        out["hist_overlap_runs"] = ov_runs.astype(float)
        out["hist_fail_rate_when_overlap"] = (ov_fails + PRIOR_A) / (ov_runs + PRIOR_A + PRIOR_B)
        # Component activity: fraction of the last 10 changes where any test tagged with c failed.
        recent = Fz[-10:]
        comp_fail = (recent @ self.tag_matrix.astype(float)) > 0  # (k, n_comp)
        comp_rate = comp_fail.mean(axis=0)
        out["comp_recent_fail_rate"] = (self.tag_matrix * comp_rate).sum(axis=1) / self.tag_matrix.sum(axis=1)
        prev = Fz[-1]
        cat_frac = {c: prev[self.categories == c].mean() for c in set(self.categories)}
        out["category_prev_fail_frac"] = np.array([cat_frac[c] for c in self.categories])
        return out
