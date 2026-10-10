"""Benchmark metric definitions, including zero-denominator handling."""

from __future__ import annotations

from cellselect.benchmark.metrics import evaluate_selection, fault_retention, runtime_reduction, time_to_first
from cellselect.data.scenarios import ChangeContext
from cellselect.selection.engine import SelectionResult, TestDecision
from cellselect.suite.catalog import build_catalog


def test_runtime_reduction_formula() -> None:
    assert runtime_reduction(100.0, 30.0) == 0.7
    assert runtime_reduction(0.0, 0.0) is None


def test_fault_retention_zero_denominator_is_undefined() -> None:
    assert fault_retention(2, 4) == 0.5
    assert fault_retention(0, 0) is None


def test_time_to_first() -> None:
    d = {"a": 10.0, "b": 5.0, "c": 1.0}
    assert time_to_first(["a", "b", "c"], d, {"b"}) == 15.0
    assert time_to_first(["a"], d, {"c"}) is None


def test_evaluate_selection_counts_missed_faults() -> None:
    catalog = list(build_catalog())[:6]
    ids = [s.test_id for s in catalog]
    decisions = {
        s.test_id: TestDecision(
            s.test_id, s.test_id in ids[:2], s.duration_s, s.category, list(s.tags), s.critical
        )
        for s in catalog
    }
    res = SelectionResult("x", 1, 1e9, sum(s.duration_s for s in catalog), ids[:2], decisions)
    change = ChangeContext(1, "feature", "rrc", ("rrc.connection",), 10, 1)
    failed = {t: t in (ids[1], ids[4]) for t in ids}
    detections = {"F1": {ids[1]}, "F2": {ids[4]}, "F3": set()}  # F3 undetectable by full suite
    m = evaluate_selection(res, change, catalog, failed, detections)
    assert m["faults_detected_full"] == 2
    assert m["faults_detected_selected"] == 1
    assert m["faults_missed"] == 1
    assert m["fault_retention"] == 0.5
    assert m["failing_tests_selected"] == 1
