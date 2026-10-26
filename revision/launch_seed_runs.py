"""Job-queue launcher for ``run_seed_experiment.py``.

Runs every (dataset, model, seed) combination as an independent subprocess,
``--workers`` at a time, longest jobs first. Each job writes its own CSV, so
the launcher can be re-run at any time and only missing results are computed.
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
from run_seed_experiment import DATASETS, MODELS, slug  # noqa: E402

# Rough single-run cost (s) from the original paper tables, used only to order the queue.
COST = {
    "edge_iiotset": 5, "wustl_iiot_2021": 4, "x_iiotid": 3,
    "cicmaldroid": 1, "drebin": 1, "tuandromd": 1, "malgenome": 1,
}
MODEL_COST = {
    "ConvKAN (Big)": 6, "ConvKAN (Medium)": 3.5, "ConvKAN (Small)": 3.3, "Convs_and_KAN": 2,
    "CNN (Big)": 1.7, "CNN (Medium)": 1.3, "CNN (Small)": 1.2, "KAN": 1.2,
    "RNN": 0.1, "LSTM": 0.1, "GRU": 0.1,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    p.add_argument("--datasets", nargs="+", default=sorted(DATASETS))
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results_seeds"))
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs_seeds"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    jobs = []
    for d in args.datasets:
        for m in args.models:
            for s in args.seeds:
                out = os.path.join(args.out_dir, d, f"{slug(m)}__seed{s}.csv")
                if os.path.exists(out):
                    continue
                jobs.append((COST[d] * MODEL_COST[m], d, m, s))
    jobs.sort(reverse=True)
    print(f"{len(jobs)} pending jobs, {args.workers} workers", flush=True)
    if args.dry_run:
        for j in jobs:
            print(j)
        return

    python = sys.executable

    def run(indexed_job):
        idx, (_, d, m, s) = indexed_job
        time.sleep(min(idx, args.workers) * 10)  # stagger the first wave of CSV loads
        log_path = os.path.join(args.log_dir, f"{d}__{slug(m)}__seed{s}.log")
        cmd = [python, os.path.join(HERE, "run_seed_experiment.py"), "--dataset", d, "--model", m,
               "--seed", str(s), "--data-dir", args.data_dir, "--out-dir", args.out_dir,
               "--threads", str(args.threads)]
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        t0 = time.time()
        with open(log_path, "w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        status = "OK " if rc == 0 else f"FAIL(rc={rc})"
        print(f"[{time.strftime('%H:%M:%S')}] {status} {d:16s} {m:17s} seed={s:<5d} {time.time() - t0:8.1f}s", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rcs = list(ex.map(run, enumerate(jobs)))
    failed = sum(1 for rc in rcs if rc != 0)
    print(f"finished: {len(rcs) - failed} ok, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
