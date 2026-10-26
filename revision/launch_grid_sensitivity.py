"""KAN hyperparameter sensitivity study.

Sweeps the two hyperparameters that have no counterpart in the other DL
architectures, the spline grid size and the spline order, on one
representative dataset, for the five model configurations that contain KAN
layers. Every point is run with three seeds (42, 123 and 2024 by default).

The grid size means the number of B-spline grid intervals in the ConKAN
variants and in the KAN layer of Conv & KAN, and the number of radial basis
functions in the KAN model. The spline order does not apply to the KAN model,
whose FastKAN layers use radial basis functions instead of B-splines.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repository root
sys.path.insert(0, HERE)
from run_seed_experiment import DATASETS, slug  # noqa: E402

# Values used in the main experiment (the library defaults), marked in the tables.
DEFAULT_GRID = {
    "ConvKAN (Small)": 5,
    "ConvKAN (Medium)": 5,
    "ConvKAN (Big)": 5,
    "Convs_and_KAN": 10,
    "KAN": 8,
}
DEFAULT_ORDER = 3

GRID_SIZES = [3, 5, 10, 20]          # B-spline models
GRID_SIZES_KAN = [4, 8, 16, 32]      # radial basis functions (FastKAN)
SPLINE_ORDERS = [1, 2, 3, 5]         # secondary axis, B-spline models only


def jobs_for(models, seeds, sweep):
    out = []
    for m in models:
        grids = GRID_SIZES_KAN if m == "KAN" else GRID_SIZES
        if sweep in ("grid", "both"):
            for g in grids:
                for s in seeds:
                    out.append((m, g, None, s))
        if sweep in ("order", "both") and m != "KAN":
            for k in SPLINE_ORDERS:
                if k == DEFAULT_ORDER:
                    continue  # already covered by the grid sweep at the default order
                for s in seeds:
                    out.append((m, DEFAULT_GRID[m], k, s))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cicmaldroid", choices=sorted(DATASETS))
    p.add_argument("--models", nargs="+", default=list(DEFAULT_GRID))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--sweep", choices=["grid", "order", "both"], default="both")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results_sensitivity"))
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs_sensitivity"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    pending = []
    for m, g, k, s in jobs_for(args.models, args.seeds, args.sweep):
        tag = f"__g{g}" + (f"__k{k}" if k is not None else "")
        out = os.path.join(args.out_dir, args.dataset, f"{slug(m)}{tag}__seed{s}.csv")
        if not os.path.exists(out):
            pending.append((m, g, k, s))
    print(f"{len(pending)} pending jobs on {args.dataset}, {args.workers} workers", flush=True)
    if args.dry_run:
        for j in pending:
            print(j)
        return

    python = sys.executable

    def run(job):
        m, g, k, s = job
        tag = f"g{g}" + (f"_k{k}" if k is not None else "")
        log_path = os.path.join(args.log_dir, f"{args.dataset}__{slug(m)}__{tag}__seed{s}.log")
        cmd = [python, os.path.join(HERE, "run_seed_experiment.py"), "--dataset", args.dataset,
               "--model", m, "--seed", str(s), "--grid-size", str(g),
               "--data-dir", args.data_dir, "--out-dir", args.out_dir, "--threads", str(args.threads)]
        if k is not None:
            cmd += ["--spline-order", str(k)]
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        t0 = time.time()
        with open(log_path, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        status = "OK " if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{time.strftime('%H:%M:%S')}] {status} {m:17s} grid={g:<3d} order={str(k):<4s} "
              f"seed={s:<5d} {time.time() - t0:8.1f}s", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rcs = list(ex.map(run, pending))
    failed = sum(1 for rc in rcs if rc != 0)
    print(f"finished: {len(rcs) - failed} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
