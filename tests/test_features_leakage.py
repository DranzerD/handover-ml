"""Leakage controls: features may only depend on strictly earlier changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cellselect.data.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    HistoryState,
)
from cellselect.data.generate import load_changes, replay_history
from cellselect.data.scenarios import ChangeContext, generate_change
from cellselect.ml.train import forward_chaining_folds, temporal_split
from cellselect.suite.catalog import build_catalog


def _changes(n: int) -> list[ChangeContext]:
    rng = np.random.default_rng(0)
    return [generate_change(rng, i).visible for i in range(n)]


def test_target_and_outcome_columns_are_not_features() -> None:
    forbidden = {TARGET, "failure_kind", "message", "wall_s", "test_id", "change_id"}
    assert not forbidden & set(FEATURE_COLUMNS)


def test_features_ignore_current_and_future_outcomes() -> None:
    catalog = build_catalog()
    changes = _changes(12)
    rng = np.random.default_rng(1)
    outcomes = [{s.test_id: bool(rng.random() < 0.1) for s in catalog} for _ in changes]
    state_a, state_b = HistoryState(catalog), HistoryState(catalog)
    for ch, out in zip(changes[:8], outcomes[:8]):
        state_a.update(ch, out)
        state_b.update(ch, out)
    feats_a = state_a.features_for_change(changes[8])
    # state_b additionally learns the "future" (changes 8..11) only after features are taken
    feats_b = state_b.features_for_change(changes[8])
    for ch, out in zip(changes[8:], outcomes[8:]):
        state_b.update(ch, {t: not v for t, v in out.items()})
    pd.testing.assert_frame_equal(feats_a, feats_b)
    assert state_b.n_changes == 12


def test_history_features_reflect_past_failures() -> None:
    catalog = build_catalog()
    state = HistoryState(catalog)
    ch = _changes(3)
    target = catalog[10].test_id
    state.update(ch[0], {s.test_id: s.test_id == target for s in catalog})
    state.update(ch[1], {s.test_id: False for s in catalog})
    f = state.features_for_change(ch[2]).set_index("test_id")
    assert f.loc[target, "hist_n_runs"] == 2
    assert f.loc[target, "last_failed"] == 0
    assert f.loc[target, "runs_since_fail"] == 1
    assert f.loc[target, "hist_fail_rate"] > f.loc[catalog[0].test_id, "hist_fail_rate"]


def test_stored_features_match_replayed_history(small_dataset) -> None:
    executions = pd.read_csv(small_dataset / "executions.csv")
    changes = load_changes(small_dataset / "changes.jsonl")
    cid = 30
    state = replay_history(executions, changes, upto_change=cid)
    ctx = ChangeContext.from_dict(changes[cid]["visible"])
    recomputed = state.features_for_change(ctx).set_index("test_id")
    stored = executions[executions.change_id == cid].set_index("test_id")
    num = NUMERIC_FEATURES
    np.testing.assert_allclose(
        recomputed.loc[stored.index, num].to_numpy(float), stored[num].to_numpy(float), atol=1e-9
    )
    for c in CATEGORICAL_FEATURES:
        assert (recomputed.loc[stored.index, c].astype(str) == stored[c].astype(str)).all(), c


def test_temporal_split_and_folds_never_mix_changes(small_dataset) -> None:
    executions = pd.read_csv(small_dataset / "executions.csv")
    train, test, cutoff = temporal_split(executions, 0.7)
    assert train.change_id.max() < cutoff <= test.change_id.min()
    for tr, va in forward_chaining_folds(train.change_id.to_numpy(), 3):
        assert tr.max() < va.min()


def test_new_tests_have_no_history_before_introduction(small_dataset) -> None:
    executions = pd.read_csv(small_dataset / "executions.csv")
    pp = executions[executions.test_id.str.startswith("HO-PP")]
    assert pp.change_id.min() == 36  # introduced at 80 % of 45 changes
    assert pp.groupby("test_id").hist_n_runs.min().eq(0).all()
