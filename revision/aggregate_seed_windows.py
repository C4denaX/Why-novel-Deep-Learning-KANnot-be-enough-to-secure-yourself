"""F1 variance across seeds 40-44: every consecutive trio around 42 and all five seeds.

Reads ``results_seeds/<dataset>/<model>__seed<N>.csv`` and writes, next to them:

* ``seed_windows_per_cell.csv``: mean, std and variance of the F1-Score (in points)
  for every (dataset, model) cell, for each trio and for the five seeds.
* ``seed_windows_summary.csv``: one row per seed set and group of datasets.
* ``seed_windows_tables.md``: the same numbers as Markdown tables.

The best trio is the one with the lowest mean variance over the 77 cells, the same
trio for every cell.
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

DATASETS = ["drebin", "malgenome", "tuandromd", "cicmaldroid",
            "wustl_iiot_2021", "x_iiotid", "edge_iiotset"]
DATASET_NAMES = {"drebin": "Drebin-215", "malgenome": "Malgenome-215", "tuandromd": "Tuandromd",
                 "cicmaldroid": "CiCMaldroid2020", "wustl_iiot_2021": "WUSTL-IIOT-2021",
                 "x_iiotid": "X-IIoTID", "edge_iiotset": "Edge-IIoTset"}
MODELS = ["CNN (Small)", "CNN (Medium)", "CNN (Big)", "RNN", "LSTM", "GRU", "KAN",
          "ConvKAN (Small)", "ConvKAN (Medium)", "ConvKAN (Big)", "Convs_and_KAN"]
MALWARE = DATASETS[:4]
SEED_SETS = {"40-41-42": (40, 41, 42), "41-42-43": (41, 42, 43),
             "42-43-44": (42, 43, 44), "40-44": (40, 41, 42, 43, 44)}
TRIOS = ["40-41-42", "41-42-43", "42-43-44"]


def load(results_dir):
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, "*", "*__seed*.csv")):
        row = next(csv.DictReader(open(path)))
        runs[(row["Dataset"], row["Model"])][int(row["Seed"])] = {
            "f1": float(row["Test F1 Score"]) * 100,
            "collapsed": float(row["Test Recall"]) > 0.9999,
        }
    return runs


def cell_stats(runs, cell, seeds):
    x = [runs[cell][s]["f1"] for s in seeds]
    return {"mean": st.mean(x), "std": st.stdev(x), "var": st.variance(x),
            "collapsed": any(runs[cell][s]["collapsed"] for s in seeds)}


def summarise(stats, cells):
    var = [stats[c]["var"] for c in cells]
    std = [stats[c]["std"] for c in cells]
    return {"mean_var": st.mean(var), "median_var": st.median(var), "max_var": max(var),
            "median_std": st.median(std), "cells_std_gt_5": sum(s > 5 for s in std)}


def fmt_cell(s):
    text = f"{s['var']:.2f} ({s['std']:.2f}){' †' if s['collapsed'] else ''}"
    return f"**{text}**" if s["var"] >= 25 else text


def md_matrix(stats):
    lines = ["| Modelo | " + " | ".join(DATASET_NAMES[d] for d in DATASETS) + " |",
             "|---" * (len(DATASETS) + 1) + "|"]
    for m in MODELS:
        cells = [fmt_cell(stats[(d, m)]) for d in DATASETS]
        lines.append(f"| {m.replace('Convs_and_KAN', 'Conv & KAN')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "results_seeds"))
    args = p.parse_args()

    runs = load(args.results_dir)
    cells = [(d, m) for d in DATASETS for m in MODELS]
    needed = sorted({s for seeds in SEED_SETS.values() for s in seeds})
    missing = [(c, s) for c in cells for s in needed if s not in runs.get(c, {})]
    if missing:
        raise SystemExit(f"{len(missing)} runs missing, e.g. {missing[:5]}")

    stats = {name: {c: cell_stats(runs, c, seeds) for c in cells} for name, seeds in SEED_SETS.items()}
    groups = {"todas": cells,
              "malware": [c for c in cells if c[0] in MALWARE],
              "intrusiones": [c for c in cells if c[0] not in MALWARE]}
    summary = {name: {g: summarise(stats[name], cs) for g, cs in groups.items()} for name in SEED_SETS}
    best = min(TRIOS, key=lambda t: summary[t]["todas"]["mean_var"])

    with open(os.path.join(args.results_dir, "seed_windows_per_cell.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Dataset", "Model", "Seeds", "f1_mean", "f1_std", "f1_var", "any_collapsed"])
        for name in SEED_SETS:
            for c in cells:
                s = stats[name][c]
                w.writerow([c[0], c[1], name, f"{s['mean']:.4f}", f"{s['std']:.4f}",
                            f"{s['var']:.4f}", int(s["collapsed"])])

    with open(os.path.join(args.results_dir, "seed_windows_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Seeds", "Group", "mean_var", "median_var", "max_var", "median_std", "cells_std_gt_5"])
        for name in SEED_SETS:
            for g in groups:
                s = summary[name][g]
                w.writerow([name, g, f"{s['mean_var']:.4f}", f"{s['median_var']:.4f}",
                            f"{s['max_var']:.4f}", f"{s['median_std']:.4f}", s["cells_std_gt_5"]])

    out = ["# Varianza del F1 entre semillas", "",
           f"Mejor trío (menor varianza media en las 77 celdas): **{best}**.", "",
           "## Resumen por conjunto de semillas", "",
           "| Semillas | Grupo | σ² media | σ² mediana | σ² máxima | σ mediana | Celdas con σ > 5 |",
           "|---|---|---|---|---|---|---|"]
    for name in SEED_SETS:
        for g in groups:
            s = summary[name][g]
            out.append(f"| {name} | {g} | {s['mean_var']:.2f} | {s['median_var']:.2f} | "
                       f"{s['max_var']:.2f} | {s['median_std']:.2f} | {s['cells_std_gt_5']}/{len(groups[g])} |")
    for title, name in [(f"Mejor trío: semillas {best}", best), ("Cinco semillas: 40-44", "40-44")]:
        out += ["", f"## {title}", "",
                "Cada celda: varianza σ² (desviación σ), F1 en puntos. Negrita: σ² ≥ 25. "
                "†: alguna semilla colapsa a una sola clase.", "", md_matrix(stats[name])]
    path = os.path.join(args.results_dir, "seed_windows_tables.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
