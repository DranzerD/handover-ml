"""Reproducible generation of the synthetic historical execution dataset.

For each simulated change (in chronological order):
  1. draw the change (visible context + hidden faults, unfixed faults persist),
  2. compute features for every test from history of *earlier* changes only,
  3. execute every active test in-process against the faulty build (genuine KPI checks),
  4. attribute failures to individual faults (single-fault re-runs) or to noise,
  5. append outcomes to history, then decide which faults get fixed.

The dataset is **synthetic**: outcomes come from the behavioural simulator, not real devices.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cellselect import __version__
from cellselect.data.features import HistoryState
from cellselect.data.scenarios import ChangeScenario, generate_change
from cellselect.suite.catalog import TestSpec, build_catalog
from cellselect.suite.runner import attribute_faults, run_suite

log = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    seed: int = 0
    n_changes: int = 400
    # Test-ID prefix -> fraction of the history after which the family is introduced (new tests).
    new_tests: dict[str, float] = field(default_factory=lambda: {"HO-PP": 0.80})
    fix_prob_detected: float = 0.70
    fix_prob_undetected: float = 0.25


def introduced_at(spec: TestSpec, cfg: GenerationConfig) -> int:
    for prefix, frac in cfg.new_tests.items():
        if spec.test_id.startswith(prefix):
            return int(round(frac * cfg.n_changes))
    return 0


@dataclass
class ChangeRecord:
    scenario: ChangeScenario
    executions: pd.DataFrame
    detections: dict[str, set[str]]
    noise: set[str]


def execute_change(scenario: ChangeScenario, specs: list[TestSpec], features: pd.DataFrame) -> ChangeRecord:
    """Run all ``specs`` for ``scenario`` and assemble feature rows + labels."""
    vis = scenario.visible
    outcomes = run_suite(specs, vis.seed, scenario.active_faults, vis.env)
    detections, noise = attribute_faults(specs, vis.seed, scenario.active_faults, vis.env, outcomes)
    detecting = {t for ts in detections.values() for t in ts}
    ids = [s.test_id for s in specs]
    df = features.set_index("test_id").loc[ids].reset_index()
    df.insert(0, "change_id", vis.change_id)
    df["duration_s"] = [s.duration_s for s in specs]
    df["failed"] = [int(not outcomes[t].passed) for t in ids]
    df["failure_kind"] = [
        ""
        if outcomes[t].passed
        else ("fault" if t in detecting else ("noise" if t in noise else "interaction"))
        for t in ids
    ]
    df["message"] = [outcomes[t].message for t in ids]
    df["wall_s"] = [round(outcomes[t].wall_s, 5) for t in ids]
    return ChangeRecord(scenario, df, detections, noise)


def generate_history(cfg: GenerationConfig, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = list(build_catalog())
    rng = np.random.default_rng(cfg.seed)
    state = HistoryState(catalog)
    open_faults: set[str] = set()
    exec_frames: list[pd.DataFrame] = []
    change_rows: list[dict] = []
    detection_rows: list[dict] = []
    t0 = time.perf_counter()
    for cid in range(cfg.n_changes):
        scenario = generate_change(rng, cid, carried_faults=tuple(sorted(open_faults)))
        specs = [s for s in catalog if introduced_at(s, cfg) <= cid]
        feats = state.features_for_change(scenario.visible)
        rec = execute_change(scenario, specs, feats)
        state.update(
            scenario.visible, dict(zip(rec.executions["test_id"], rec.executions["failed"].astype(bool)))
        )
        exec_frames.append(rec.executions)
        detected = {f for f, ts in rec.detections.items() if ts}
        for f, ts in rec.detections.items():
            for t in sorted(ts):
                detection_rows.append({"change_id": cid, "fault_id": f, "test_id": t})
        row = scenario.to_dict()
        row["ground_truth"]["detected_by_full_suite"] = sorted(detected)
        row["ground_truth"]["noise_failures"] = sorted(rec.noise)
        change_rows.append(row)
        open_faults = {
            f
            for f in scenario.active_faults
            if rng.random() >= (cfg.fix_prob_detected if f in detected else cfg.fix_prob_undetected)
        }
        if (cid + 1) % 50 == 0 or cid + 1 == cfg.n_changes:
            log.info("generated %d/%d changes (%.1fs)", cid + 1, cfg.n_changes, time.perf_counter() - t0)

    executions = pd.concat(exec_frames, ignore_index=True)
    paths = {
        "executions": out / "executions.csv",
        "changes": out / "changes.jsonl",
        "fault_detection": out / "fault_detection.csv",
        "catalog": out / "catalog.json",
        "metadata": out / "metadata.json",
    }
    executions.to_csv(paths["executions"], index=False)
    with paths["changes"].open("w", encoding="utf-8") as fh:
        for row in change_rows:
            fh.write(json.dumps(row) + "\n")
    pd.DataFrame(detection_rows, columns=["change_id", "fault_id", "test_id"]).to_csv(
        paths["fault_detection"], index=False
    )
    paths["catalog"].write_text(json.dumps([s.to_dict() for s in catalog], indent=1), encoding="utf-8")
    n_faulty = sum(1 for r in change_rows if r["ground_truth"]["active_faults"])
    meta = {
        "synthetic": True,
        "note": "Synthetic dataset produced by the CellSelect behavioural simulator; not real device data.",
        "config": asdict(cfg),
        "cellselect_version": __version__,
        "python": platform.python_version(),
        "n_changes": cfg.n_changes,
        "n_executions": int(len(executions)),
        "n_tests": len(catalog),
        "failure_rate": float(executions["failed"].mean()),
        "changes_with_active_faults": n_faulty,
        "failure_kinds": executions.loc[executions["failed"] == 1, "failure_kind"].value_counts().to_dict(),
        "generation_seconds": round(time.perf_counter() - t0, 1),
    }
    paths["metadata"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "dataset written to %s: %d executions, failure rate %.3f", out, len(executions), meta["failure_rate"]
    )
    return paths


def load_changes(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def replay_history(
    executions: pd.DataFrame, changes: list[dict], upto_change: int | None = None
) -> HistoryState:
    """Rebuild a HistoryState from stored executions for changes with id < ``upto_change``."""
    from cellselect.data.scenarios import ChangeContext

    state = HistoryState(build_catalog())
    by_change = {cid: g for cid, g in executions.groupby("change_id", sort=True)}
    for row in changes:
        ctx = ChangeContext.from_dict(row["visible"])
        if upto_change is not None and ctx.change_id >= upto_change:
            break
        g = by_change.get(ctx.change_id)
        if g is None:
            continue
        state.update(ctx, dict(zip(g["test_id"], g["failed"].astype(bool))))
    return state
