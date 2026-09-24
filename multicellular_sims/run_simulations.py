#!/usr/bin/env python
"""
Driver for the mechano-chemical gland model.

Everything that defines a run lives in the CONFIG block below. A run is fully
specified by a dict of keyword arguments to run_sim, so adding a new experiment 
means appending to EXPERIMENTS, not editing
any of the machinery.

    python driver.py                       # everything enabled
    python driver.py --only baseline mut   # selected experiments
    python driver.py --list                # show what would run
    python driver.py --dry-run             # build tasks, run nothing
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys

import yaml
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from run_model import (
    run_sim, S_step
)

# ==========================================================================
# CONFIG -- edit here, not below
# ==========================================================================
# OUT_ROOT = Path("./outputs_params_refined")
OUT_ROOT = Path("./test_outputs")

PHI_D_HEALTHY = None

MECH = dict(kappa=1350.0)

SOURCE = dict(a_step=0.0, b_step=0.7)

perturb_factors = {
    "alpha_M": 0.2,
}

MUTATIONS = {
    "alphaM_high":  {'alpha_M': (1.0 + perturb_factors["alpha_M"])},
    "alphaM_low":   {'alpha_M': (1.0 - perturb_factors["alpha_M"])},
}

N_REPS = dict(baseline=100, mutation=25, slough=100)
MAX_WORKERS = 1
CHUNK_SIZE = 30

# ==========================================================================
# task construction
# ==========================================================================

def _common(outdir, rep, **extra):
    """Every task shares this skeleton; extra overrides any of it."""
    d = dict(
        outdir=str(outdir), rep=rep,
        phi_D=PHI_D_HEALTHY,
        a_step=SOURCE["a_step"], b_step=SOURCE["b_step"],
        source_fn=S_step, source_kwargs=dict(a=SOURCE["a_step"], b=SOURCE["b_step"]),
        amp_key="S0", rapid_slough=False,
        nd_overrides=None, post_burnin_scale=None, baseline_dir=None, tag=None,
        minimal_save=False, maximal_save=True,
        D_W=None, D_D=None, init_length=1,
        **MECH,
    )
    d.update(extra)
    Path(d["outdir"]).mkdir(parents=True, exist_ok=True)
    return d


def build_baseline(n):
    out = OUT_ROOT / "baseline"
    tasks = [_common(out / "healthy", r) for r in range(n)]
    tasks += [_common(out / "noECM", r, no_ecm=True, tag="noECM") for r in range(n)]
    return tasks


def build_mutations(n):
    """Mutations initialise from the healthy rep0 run, then perturb at t=0."""
    base = OUT_ROOT / "baseline" / "healthy"
    tasks = []
    for tag, scale in MUTATIONS.items():
        out = OUT_ROOT / "mutations" / tag
        tasks += [_common(out, r, post_burnin_scale=scale, tag=tag,
                          baseline_dir=str(base))
                  for r in range(n)]
    return tasks

def build_rapid_slough(n):
    out = OUT_ROOT / "rapid_slough"
    base = OUT_ROOT / "baseline" / "healthy"
    return [_common(out, r, rapid_slough=True, tag="rapid_slough",
                    baseline_dir=str(base)) for r in range(n)]

EXPERIMENTS = {
    "baseline": lambda: build_baseline(N_REPS["baseline"]),
    "mut":      lambda: build_mutations(N_REPS["mutation"]),
    "slough":   lambda: build_rapid_slough(N_REPS["slough"]),
}
DEFAULT_ORDER = ["baseline", "mut", "slough"]


# ==========================================================================
# runner
# ==========================================================================

def run_tasks(tasks, max_workers, desc):
    if not tasks:
        print(f"no tasks for {desc}"); return []
    max_workers = max_workers or os.cpu_count() or 1
    t0 = datetime.now()
    print(f"[{t0:%H:%M:%S}] {desc}: {len(tasks)} runs on {max_workers} workers")
    results = []
    for i in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[i:i + CHUNK_SIZE]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_sim, **kw): kw for kw in chunk}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{desc} {i//CHUNK_SIZE+1}"):
                try:
                    results.append(fut.result())
                except Exception as e:
                    kw = futs[fut]
                    print(f"  FAILED rep={kw['rep']} {kw['outdir']}: {type(e).__name__}: {e}")
    print(f"[{datetime.now():%H:%M:%S}] {desc} done in "
          f"{(datetime.now()-t0).total_seconds():.0f}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=None)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    order = a.only or DEFAULT_ORDER
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # baseline must complete first: mutations and rapid_slough initialise from it
    if "mut" in order or "slough" in order:
        if "baseline" in order and order.index("baseline") > min(
                [order.index(x) for x in ("mut", "slough") if x in order]):
            raise SystemExit("baseline must be listed before mut/slough")

    for name in order:
        tasks = EXPERIMENTS[name]()
        if a.list or a.dry_run:
            print(f"{name}: {len(tasks)} tasks -> "
                  f"{sorted({t['outdir'] for t in tasks})[:3]} ...")
            continue
        run_tasks(tasks, a.workers, name)
    if not (a.list or a.dry_run):
        print("all enabled experiments complete.")


if __name__ == "__main__":
    main()