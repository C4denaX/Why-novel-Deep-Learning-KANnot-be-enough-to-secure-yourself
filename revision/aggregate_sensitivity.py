"""Aggregate the KAN grid-size / spline-order sweep and emit the LaTeX table
for the sensitivity analysis.

Reads the per-run CSVs written by ``run_seed_experiment.py`` under
``results_sensitivity/<dataset>/`` and prints, for each KAN configuration,
the mean and standard deviation of the F1-Score over the seeds, together
with the parameter count and the mean runtime.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

# Paper row name -> (run model name, default grid, default spline order)
ROWS = [
    ("ConKAN (Small)", "ConvKAN (Small)", 5, 3),
    ("ConKAN (Medium)", "ConvKAN (Medium)", 5, 3),
    ("ConKAN (Big)", "ConvKAN (Big)", 5, 3),
    ("Conv \\& KAN", "Convs_and_KAN", 10, 3),
    ("KAN", "KAN", 8, None),
]


def load(results_dir, dataset):
    files = glob.glob(os.path.join(results_dir, dataset, "*.csv"))
    if not files:
        raise SystemExit(f"no results under {os.path.join(results_dir, dataset)}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["F1 (%)"] = df["Test F1 Score"] * 100
    df["Spline Order"] = df["Spline Order"].fillna(-1)  # -1 = library default (3)
    return df


def summarise(df):
    return df.groupby(["Model", "Grid Size", "Spline Order"]).agg(
        n=("Seed", "count"),
        f1_mean=("F1 (%)", "mean"),
        f1_std=("F1 (%)", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
        params=("Number of Parameters", "first"),
        runtime=("Runtime (s)", "mean"),
    ).reset_index()


def cell(r):
    if np.isnan(r.f1_std):
        return f"{r.f1_mean:.2f}"
    return f"{r.f1_mean:.2f} $\\pm$ {r.f1_std:.2f}"


def emit(agg, sweep, caption, label):
    """sweep='grid' varies the grid at the default order; 'order' varies the order."""
    lines = ["\\begin{table}[h!]", "\\centering", "\\small",
             "\\begin{tabular}{|l|c|c|c|}", "\\hline",
             ("\\textbf{Model} & \\textbf{Grid size} & \\textbf{F1-Score (\\%)} & "
              "\\textbf{Parameters} \\\\ \\hline")
             if sweep == "grid" else
             ("\\textbf{Model} & \\textbf{Spline order} & \\textbf{F1-Score (\\%)} & "
              "\\textbf{Parameters} \\\\ \\hline")]
    for paper_name, run_name, dgrid, dorder in ROWS:
        sub = agg[agg.Model == run_name]
        if sweep == "grid":
            sub = sub[sub["Spline Order"] == -1].sort_values("Grid Size")
            key = "Grid Size"
        else:
            if dorder is None:
                continue  # the KAN model has no spline order
            sub = sub[sub["Grid Size"] == dgrid].sort_values("Spline Order")
            sub = sub.replace({"Spline Order": {-1: dorder}}).sort_values("Spline Order")
            key = "Spline Order"
        if sub.empty:
            continue
        for i, r in enumerate(sub.itertuples()):
            val = int(getattr(r, "_2" if key == "Grid Size" else "_3"))
            default = (val == dgrid) if sweep == "grid" else (val == dorder)
            mark = "*" if default else ""
            name = paper_name if i == 0 else ""
            lines.append(f"{name} & {val}{mark} & {cell(r)} & {r.params:,} \\\\"
                         + (" \\hline" if i == len(sub) - 1 else ""))
    lines += ["\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(ROOT, "results_sensitivity"))
    p.add_argument("--dataset", default="cicmaldroid")
    p.add_argument("--out-tex", default=os.path.join(ROOT, "results_sensitivity", "tables_sensitivity.tex"))
    args = p.parse_args()

    df = load(args.results_dir, args.dataset)
    agg = summarise(df)

    print(f"{'model':17s} {'grid':>5s} {'order':>6s} {'n':>2s} {'F1 mean':>8s} {'std':>6s} {'params':>9s} {'runtime':>8s}")
    for r in agg.itertuples():
        order = "def" if r._3 == -1 else int(r._3)
        print(f"{r.Model:17s} {int(r._2):5d} {str(order):>6s} {r.n:2d} {r.f1_mean:8.2f} "
              f"{0.0 if np.isnan(r.f1_std) else r.f1_std:6.2f} {r.params:9d} {r.runtime:8.1f}")

    cap_g = ("Sensitivity of the KAN configurations to the grid size on CiCMaldroid2020. Each entry is the mean "
             "and standard deviation of three independent training runs with seeds 42, 123 and 2024. An asterisk "
             "marks the default value used in Tables \\ref{tab:results_malware} and \\ref{tab:results_traffic}.")
    cap_o = ("Sensitivity of the KAN configurations to the spline order on CiCMaldroid2020, at the default grid "
             "size of each configuration. The KAN model is omitted because its FastKAN layers use radial basis "
             "functions rather than B-splines. An asterisk marks the default value.")
    tex = (emit(agg, "grid", cap_g, "tab:kan_sensitivity") + "\n\n"
           + emit(agg, "order", cap_o, "tab:kan_sensitivity_order"))
    with open(args.out_tex, "w") as fh:
        fh.write(tex + "\n")
    print(f"\nwrote {args.out_tex}")


if __name__ == "__main__":
    main()
