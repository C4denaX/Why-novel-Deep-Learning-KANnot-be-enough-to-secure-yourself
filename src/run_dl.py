"""DL and KAN models: grid search with stratified k-fold cross-validation on the
training split, retraining with the best values and evaluation on the held-out
test split.

For every model and every combination of learning_rate, weight_decay and gamma
in the configuration, the model is trained on k-1 folds with AdamW, an
ExponentialLR scheduler and CrossEntropyLoss, and evaluated on the remaining fold
after each number of epochs listed in the configuration. Since the learning rate
schedule does not depend on the total number of epochs, a run of E epochs is the
first E epochs of a run of max(epochs) epochs with the same seed, so one run per
fold covers every value of ``epochs``. The combination with the highest mean
F1-Score over the folds is used to retrain the model on the whole training split,
which is then evaluated on the test split.

Example:
    python -m src.run_dl --dataset cicmaldroid
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_split  # noqa: E402
from src.dl_models import LAYOUT, build_model, count_parameters  # noqa: E402
from src.metrics import dl_metrics  # noqa: E402
from src.preprocessing import prepare_data  # noqa: E402
from src.utils import ROOT, load_config, log, resolve_device, seed_everything, update_json, write_results  # noqa: E402

COLUMNS = ["Model", "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Test Loss",
           "Number of Parameters", "Runtime (s)"]


def make_tensors(X, y, layout):
    x = torch.tensor(X, dtype=torch.float32)
    if layout in ("CNN", "RNN"):
        x = x.unsqueeze(1)  # one input channel / one time step
    return x, torch.tensor(y, dtype=torch.long)


def train(model, device, train_loader, optimizer, criterion):
    """Train for one epoch and return the average training loss."""
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / (batch_idx + 1)


def test(model, device, test_loader, criterion):
    """Evaluate the model: loss, accuracy, precision, recall and F1-Score."""
    model.eval()
    test_loss = 0
    correct = 0
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            correct += (target == predicted).sum().item()
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    precision, recall, f1 = dl_metrics(all_targets, all_predictions)
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy, precision, recall, f1


def fit_and_evaluate(name, X_tr, y_tr, X_ev, y_ev, num_classes, params, eval_epochs, cfg, device, seed):
    """Train one model and evaluate it on (X_ev, y_ev) after each epoch in ``eval_epochs``.

    Returns the model, {epoch: (loss, accuracy, precision, recall, f1)} and the
    time spent training and evaluating, in seconds.
    """
    seed_everything(seed)
    layout = LAYOUT[name]
    batch_size = cfg["dl"]["batch_size"][layout]
    train_loader = DataLoader(TensorDataset(*make_tensors(X_tr, y_tr, layout)), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(*make_tensors(X_ev, y_ev, layout)), batch_size=batch_size, shuffle=False)
    model = build_model(name, X_tr.shape[1], num_classes, cfg["dl"]["kan"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["gamma"])
    criterion = nn.CrossEntropyLoss()

    scores = {}
    start = time.time()
    for epoch in range(1, max(eval_epochs) + 1):
        train(model, device, train_loader, optimizer, criterion)
        if epoch in eval_epochs:
            scores[epoch] = test(model, device, eval_loader, criterion)
        scheduler.step()
    return model, scores, time.time() - start


def grid_search(name, X_train, y_train, num_classes, cfg, device, seed, n_folds):
    space = cfg["dl"]["search_space"]
    epochs_grid = sorted(int(e) for e in space["epochs"])
    folds = list(data_split.cv_folds(n_folds, seed).split(X_train, y_train))
    results = []
    for lr, wd, gamma in itertools.product(space["learning_rate"], space["weight_decay"], space["gamma"]):
        params = {"learning_rate": lr, "weight_decay": wd, "gamma": gamma}
        per_epoch = {e: [] for e in epochs_grid}
        for k, (tr, va) in enumerate(folds):
            _, scores, _ = fit_and_evaluate(name, X_train[tr], y_train[tr], X_train[va], y_train[va],
                                            num_classes, params, epochs_grid, cfg, device, seed)
            for e in epochs_grid:
                per_epoch[e].append(float(scores[e][4]))
        for e in epochs_grid:
            row = dict(params, epochs=e, cv_f1_mean=float(np.mean(per_epoch[e])),
                       cv_f1_std=float(np.std(per_epoch[e])), cv_f1_folds=per_epoch[e])
            results.append(row)
            log(f"{name}: epochs={e} lr={lr} weight_decay={wd} gamma={gamma} "
                f"CV F1={row['cv_f1_mean']:.4f} (+- {row['cv_f1_std']:.4f})")
    best = max(results, key=lambda r: r["cv_f1_mean"])  # first one wins ties
    best_params = {k: best[k] for k in ("epochs", "learning_rate", "weight_decay", "gamma")}
    return best_params, best["cv_f1_mean"], results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset key from the configuration, e.g. cicmaldroid")
    p.add_argument("--models", nargs="+", default=None, help="subset of models (default: all eleven)")
    p.add_argument("--config", default=None, help="configuration file (default: src/config.json)")
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    p.add_argument("--seed", type=int, default=None, help="random seed (default: value in the configuration)")
    p.add_argument("--folds", type=int, default=None, help="number of CV folds (default: value in the configuration)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--subsample", type=float, default=None,
                   help="use a stratified fraction of the rows (only for quick checks)")
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    n_folds = cfg["split"]["cv_folds"] if args.folds is None else args.folds
    models = args.models or cfg["dl"]["default_models"]
    unknown = [m for m in models if m not in LAYOUT]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choose from {list(LAYOUT)}")
    if args.dataset not in cfg["datasets"]:
        raise SystemExit(f"unknown dataset: {args.dataset}; choose from {list(cfg['datasets'])}")
    device = torch.device(resolve_device(args.device))
    log(f"device={device} dataset={args.dataset} seed={seed} folds={n_folds}")

    path = os.path.join(args.data_dir, cfg["datasets"][args.dataset]["file"])
    X_train, X_test, y_train, y_test, info = prepare_data(cfg, path, seed, args.subsample, log)
    num_classes = len(np.unique(y_train))

    out_csv = os.path.join(args.out_dir, "dl_kan", f"{args.dataset}.csv")
    out_json = os.path.join(args.out_dir, "dl_kan", f"{args.dataset}_search.json")
    for name in models:
        t0 = time.time()
        best_params, best_cv, results = grid_search(name, X_train, y_train, num_classes, cfg, device, seed, n_folds)
        search_time = time.time() - t0
        log(f"{name}: best {best_params} (CV F1={best_cv:.4f}); grid search took {search_time:.1f}s")

        model, scores, runtime = fit_and_evaluate(name, X_train, y_train, X_test, y_test, num_classes,
                                                  best_params, [best_params["epochs"]], cfg, device, seed)
        test_loss, accuracy, precision, recall, f1 = scores[best_params["epochs"]]
        row = [name, accuracy, precision, recall, f1, test_loss, count_parameters(model), runtime]
        write_results([row], COLUMNS, out_csv, cfg["dl"]["default_models"])
        update_json(out_json, name, {"best_params": best_params, "best_cv_f1": best_cv,
                                     "grid_search_time_s": search_time, "cv_folds": n_folds,
                                     "grid": results}, info)
        log(f"{name}: test accuracy={accuracy:.4f} F1={f1:.4f} runtime={runtime:.1f}s -> {out_csv}")


if __name__ == "__main__":
    main()
