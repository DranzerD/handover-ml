"""Calibration utility: baseline (fault-free) failure rates and the fault -> detecting-tests matrix.

Usage: python scripts/fault_matrix.py [--envs 20] [--out artifacts/fault_matrix.csv]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from cellselect.sim.context import Environment
from cellselect.sim.faults import FAULTS
from cellselect.suite.catalog import build_catalog
from cellselect.suite.runner import run_suite


def random_env(rng: np.random.Generator) -> Environment:
    return Environment(
        device=str(rng.choice(["devA", "devB", "devC"])),
        lab_interference_db=float(rng.uniform(0, 3)),
        temperature_c=float(rng.uniform(20, 45)),
        firmware_branch=str(rng.choice(["main", "release"])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--fault-envs", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    specs = build_catalog()
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    base_fail: Counter[str] = Counter()
    wall: defaultdict[str, float] = defaultdict(float)
    envs = [(int(rng.integers(1 << 30)), random_env(rng)) for _ in range(args.envs)]
    for seed, env in envs:
        res = run_suite(specs, seed, (), env)
        for tid, o in res.items():
            wall[tid] += o.wall_s
            if not o.passed:
                base_fail[tid] += 1
                if base_fail[tid] == 1 and args.verbose:
                    print(f"  baseline fail {tid}: {o.message}")
    print(f"baseline: {args.envs} suite runs in {time.perf_counter() - t0:.1f}s")
    print("baseline failing tests:", {k: v / args.envs for k, v in base_fail.most_common()})
    slow = sorted(wall.items(), key=lambda kv: -kv[1])[:8]
    print("slowest tests (total wall s):", [(k, round(v, 2)) for k, v in slow])
    rows = []
    for fid in FAULTS:
        detect: Counter[str] = Counter()
        for seed, env in envs[: args.fault_envs]:
            res = run_suite(specs, seed, (fid,), env)
            for tid, o in res.items():
                if not o.passed:
                    detect[tid] += 1
        dets = {
            t: round(c / args.fault_envs, 2)
            for t, c in sorted(detect.items())
            if base_fail[t] / args.envs <= 0.2 and c / args.fault_envs > base_fail[t] / args.envs + 0.2
        }
        print(f"{fid:18s} {FAULTS[fid].component:22s} detects {len(dets):3d}: {dets}")
        for t, c in detect.items():
            rows.append({"fault_id": fid, "test_id": t, "rate": c / args.fault_envs})
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
