"""Test catalog integrity and fault detectability of the validation suite."""

from __future__ import annotations

from collections import Counter

from cellselect.sim.context import Environment
from cellselect.sim.faults import COMPONENTS, FAULTS
from cellselect.suite.catalog import CATEGORIES, build_catalog
from cellselect.suite.checks import CHECKS
from cellselect.suite.runner import attribute_faults, run_suite


def test_catalog_ids_unique_and_well_formed() -> None:
    specs = build_catalog()
    ids = [s.test_id for s in specs]
    assert len(ids) == len(set(ids)) == 160
    for s in specs:
        assert s.template in CHECKS
        assert s.category in CATEGORIES
        assert s.duration_s > 0
        assert s.tags and all(t in COMPONENTS for t in s.tags)
    assert sum(s.critical for s in specs) >= 5
    assert set(Counter(s.category for s in specs)) == set(CATEGORIES)


def test_full_suite_passes_without_faults_in_default_environment() -> None:
    res = run_suite(build_catalog(), 0, (), Environment())
    failing = {t: o.message for t, o in res.items() if not o.passed}
    assert not failing, failing


def test_every_fault_is_detectable_by_some_test() -> None:
    specs = build_catalog()
    envs = [(11, Environment()), (12, Environment("devB", 1.5, 30.0)), (13, Environment("devC", 0.5, 22.0))]
    undetected = []
    for fid in FAULTS:
        if not any(not o.passed for seed, env in envs for o in run_suite(specs, seed, (fid,), env).values()):
            undetected.append(fid)
    assert not undetected


def test_attribution_separates_faults() -> None:
    specs = build_catalog()
    env = Environment()
    faults = ("F_MTU", "F_SCHED_MULTIUSER")
    combined = run_suite(specs, 5, faults, env)
    det, noise = attribute_faults(specs, 5, faults, env, combined)
    assert det["F_MTU"] == {"CFG-APPLY-006"}
    assert det["F_SCHED_MULTIUSER"] and all(t.startswith("TPUT-MU") for t in det["F_SCHED_MULTIUSER"])
    assert not noise
