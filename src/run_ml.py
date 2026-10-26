"""ML models: grid search with stratified k-fold cross-validation (scikit-learn
``GridSearchCV``) on the training split, retraining with the best values and
evaluation on the held-out test split.

The search spaces are those of the configuration (values inside the ranges of
Table 3 of the paper). The cross-validation score is the macro F1-Score.

Example:
    python -m src.run_ml --dataset cicmaldroid
"""
import argparse
import importlib
import os
import sys
import time
import warnings

import numpy as np
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_split  # noqa: E402
from src.metrics import ml_metrics  # noqa: E402
from src.preprocessing import prepare_data  # noqa: E402
from src.utils import ROOT, load_config, log, resolve_device, update_json, write_results  # noqa: E402

COLUMNS = ["Model", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Runtime (s)"]

# LightGBM names the columns of NumPy input, which makes scikit-learn warn at prediction time.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def make_estimator(spec, seed, device):
    module, cls = spec["class"].rsplit(".", 1)
    params = dict(spec.get("fixed", {}))
    if spec.get("seed_param") and spec["seed_param"] not in spec["grid"]:
        params[spec["seed_param"]] = seed
    if spec.get("device_param"):
        params[spec["device_param"]] = device
    return getattr(importlib.import_module(module), cls)(**params)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset key from the configuration, e.g. cicmaldroid")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of models (default: the six models of the result tables)")
    p.add_argument("--config", default=None, help="configuration file (default: src/config.json)")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--seed", type=int, default=None, help="random seed (default: value in the configuration)")
    p.add_argument("--folds", type=int, default=None, help="number of CV folds (default: value in the configuration)")
    p.add_argument("--n-jobs", type=int, default=1, help="parallel jobs of GridSearchCV")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="device for XGBoost (the other ML models run on the CPU)")
    p.add_argument("--subsample", type=float, default=None,
                   help="use a stratified fraction of the rows (only for quick checks)")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    n_folds = cfg["split"]["cv_folds"] if args.folds is None else args.folds
    specs = cfg["ml"]["models"]
    models = args.models or cfg["ml"]["default_models"]
    unknown = [m for m in models if m not in specs]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choose from {list(specs)}")
    if args.dataset not in cfg["datasets"]:
        raise SystemExit(f"unknown dataset: {args.dataset}; choose from {list(cfg['datasets'])}")
    device = resolve_device(args.device)
    log(f"device={device} dataset={args.dataset} seed={seed} folds={n_folds}")

    path = os.path.join(args.data_dir, cfg["datasets"][args.dataset]["file"])
    X_train, X_test, y_train, y_test, info = prepare_data(cfg, path, seed, args.subsample, log)

    out_csv = os.path.join(args.out_dir, "ml", f"{args.dataset}.csv")
    out_json = os.path.join(args.out_dir, "ml", f"{args.dataset}_search.json")
    order = list(specs)
    for name in models:
        spec = specs[name]
        search = GridSearchCV(make_estimator(spec, seed, device), spec["grid"], scoring=cfg["ml"]["scoring"],
                              cv=data_split.cv_folds(n_folds, seed), n_jobs=args.n_jobs, refit=False,
                              error_score="raise")
        t0 = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - t0
        log(f"{name}: best {search.best_params_} (CV {cfg['ml']['scoring']}={search.best_score_:.4f}); "
            f"grid search took {search_time:.1f}s")

        model = make_estimator(spec, seed, device).set_params(**search.best_params_)
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test)).ravel()
        accuracy, precision, recall, f1 = ml_metrics(y_test, y_pred)
        runtime = time.time() - start

        write_results([[name, accuracy, precision, recall, f1, runtime]], COLUMNS, out_csv, order)
        grid = [dict(params, cv_score_mean=float(m), cv_score_std=float(s))
                for params, m, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                        search.cv_results_["std_test_score"])]
        update_json(out_json, name, {"best_params": search.best_params_, "best_cv_score": search.best_score_,
                                     "scoring": cfg["ml"]["scoring"], "grid_search_time_s": search_time,
                                     "cv_folds": n_folds, "grid": grid}, info)
        log(f"{name}: test accuracy={accuracy:.4f} F1={f1:.4f} runtime={runtime:.1f}s -> {out_csv}")


if __name__ == "__main__":
    main()
