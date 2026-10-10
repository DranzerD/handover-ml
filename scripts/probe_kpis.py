"""Print KPI distributions for tests matching a prefix, with and without a given fault.

Usage: python scripts/probe_kpis.py LAT-LOSS [F_HARQ_MAXRETX] [--envs 10]
"""

from __future__ import annotations

import argparse

import numpy as np
from fault_matrix import random_env  # type: ignore[import-not-found]

from cellselect.suite.catalog import build_catalog
from cellselect.suite.runner import run_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("fault", nargs="?", default=None)
    ap.add_argument("--envs", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    envs = [(int(rng.integers(1 << 30)), random_env(rng)) for _ in range(args.envs)]
    for spec in build_catalog():
        if not spec.test_id.startswith(args.prefix):
            continue
        for label, faults in (("clean", ()), (args.fault, (args.fault,))):
            if label is None:
                continue
            outs = [run_test(spec, s, faults, e) for s, e in envs]
            fails = sum(not o.passed for o in outs)
            msgs = sorted({o.message for o in outs if not o.passed})[:2]
            kp = {
                k: round(float(np.mean([o.kpis.get(k, np.nan) for o in outs if o.kpis])), 4)
                for k in (outs[0].kpis or {})
            }
            print(f"{spec.test_id:14s} {str(label):16s} fail {fails}/{len(outs)} {spec.params} {kp} {msgs}")


if __name__ == "__main__":
    main()
