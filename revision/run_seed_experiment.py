"""Headless, seed-controlled replica of the original DL/KAN experiment notebook.

The training loop, data split, architectures, batch sizes, optimiser,
scheduler, number of epochs and metric selection are exactly those used in
the notebook. The main additions are:

* an explicit random seed (Python, NumPy, PyTorch, CUDA) fixed *before* the
  DataLoaders and the model are built, so weight initialisation and batch
  shuffling are reproducible;
* a CLI so that a single (dataset, seed, model) job can be launched
  non-interactively and its result written to disk.

The 70/30 stratified split keeps ``random_state=42`` in every run, so the
run-to-run variation reported reflects training stochasticity only.
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
sys.path.insert(0, ROOT)

from architectures.SimpleModels import SimpleCNN, SimpleCNN_2  # noqa: E402
from architectures.ConvNet import ConvNet1D  # noqa: E402
from architectures.conv_and_kan import NormalConvsKAN  # noqa: E402
from architectures.kan1d import ConvKAN, Simple_ConvKAN, Simple_ConvKAN_2  # noqa: E402
from architectures.RnnModels import (  # noqa: E402
    GRUMalware_Model1,
    LSTMMalware_Model1,
    RNNMalware_Model1,
)
from fastkan import FastKAN as KAN  # noqa: E402

warnings.filterwarnings("ignore")

# Dataset key -> CSV file (last column is the integer-encoded label).
DATASETS = {
    "cicmaldroid": "cicmaldroid.csv",
    "drebin": "debrin.csv",
    "malgenome": "malgenome.csv",
    "tuandromd": "TUANDROMD.csv",
    "wustl_iiot_2021": "wustl_iiot_2021.csv",
    "x_iiotid": "X-IIoTID dataset.csv",
    "edge_iiotset": "DNN-EdgeIIoT-dataset.csv",
}

# Model name (as in the notebook / paper CSVs) -> (input layout, batch size).
MODELS = {
    "CNN (Small)": ("CNN", 4),
    "CNN (Medium)": ("CNN", 4),
    "CNN (Big)": ("CNN", 4),
    "ConvKAN (Small)": ("CNN", 4),
    "ConvKAN (Medium)": ("CNN", 4),
    "ConvKAN (Big)": ("CNN", 4),
    "Convs_and_KAN": ("CNN", 4),
    "RNN": ("RNN", 64),
    "LSTM": ("RNN", 64),
    "GRU": ("RNN", 64),
    "KAN": ("None", 64),
}

EPOCHS = 11
LR = 1e-3
WEIGHT_DECAY = 1e-4
GAMMA = 0.8
SPLIT_SEED = 42  # unchanged from the notebook


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MyDataset(Dataset):
    """Verbatim port of the notebook dataset class (np.long -> np.int64)."""

    def __init__(self, root, test_size=0.3, random_state=SPLIT_SEED, model_type="CNN"):
        df = pd.read_csv(root, low_memory=True)
        data = df.to_numpy()
        x = data[:, :-1].astype(np.float32)
        y = data[:, -1].astype(np.int64)
        y = self._convert_labels_to_indices(y)
        del df, data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
        )
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)
        if model_type in ("CNN", "RNN"):
            self.x_train = self.x_train.unsqueeze(1)
            self.x_test = self.x_test.unsqueeze(1)

    @staticmethod
    def _convert_labels_to_indices(y):
        unique_labels = np.unique(y)
        label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
        return np.array([label_to_index[label] for label in y], dtype=np.int64)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx, :], self.y_train[idx]

    def get_train_set(self):
        return self.x_train, self.y_train

    def get_test_set(self):
        return self.x_test, self.y_test


def build_model(name, num_features, num_classes, grid_size=None, spline_order=None):
    """Build one model. ``grid_size``/``spline_order`` default to the values used
    in the main experiment: 5 and 3 for the ConKAN layers, 10 and 3 for the KAN
    layer of Conv & KAN, and 8 radial basis functions for the KAN model."""
    kan_kw = {}
    if grid_size is not None:
        kan_kw["grid_size"] = grid_size
    if spline_order is not None:
        kan_kw["spline_order"] = spline_order
    if name == "CNN (Small)":
        return SimpleCNN(num_features)
    if name == "CNN (Medium)":
        return SimpleCNN_2(num_features)
    if name == "CNN (Big)":
        return ConvNet1D(num_features)
    if name == "ConvKAN (Small)":
        return Simple_ConvKAN(input_size=num_features, num_classes=num_classes, **kan_kw)
    if name == "ConvKAN (Medium)":
        return Simple_ConvKAN_2(input_size=num_features, num_classes=num_classes, **kan_kw)
    if name == "ConvKAN (Big)":
        return ConvKAN(input_size=num_features, num_classes=num_classes, **kan_kw)
    if name == "Convs_and_KAN":
        return NormalConvsKAN(num_features, **kan_kw)
    if name == "RNN":
        return RNNMalware_Model1(input_dim=num_features, output_dim=num_classes)
    if name == "LSTM":
        return LSTMMalware_Model1(input_dim=num_features, output_dim=num_classes)
    if name == "GRU":
        return GRUMalware_Model1(input_dim=num_features, output_dim=num_classes)
    if name == "KAN":
        kw = {"num_grids": grid_size} if grid_size is not None else {}
        return KAN(layers_hidden=[num_features, 128, num_classes], **kw)
    raise ValueError(name)


def train(model, device, train_loader, optimizer, criterion):
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
    # Metric of the original notebook: macro average over the classes that the
    # model actually predicted. A model that collapses onto a single class is
    # scored on that class alone, which inflates its F1-Score.
    labels = np.unique(all_predictions)
    precision = precision_score(all_targets, all_predictions, average="macro", labels=labels)
    recall = recall_score(all_targets, all_predictions, average="macro", labels=labels)
    f1 = f1_score(all_targets, all_predictions, average="macro", labels=labels)
    # Same metrics averaged over every class present in the test set, which is
    # the standard macro average and does not reward collapsed models.
    all_labels = np.unique(all_targets)
    precision_all = precision_score(all_targets, all_predictions, average="macro",
                                    labels=all_labels, zero_division=0)
    recall_all = recall_score(all_targets, all_predictions, average="macro",
                              labels=all_labels, zero_division=0)
    f1_all = f1_score(all_targets, all_predictions, average="macro",
                      labels=all_labels, zero_division=0)
    n_predicted = int(len(labels))
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return (test_loss, accuracy, precision, recall, f1,
            precision_all, recall_all, f1_all, n_predicted)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_job(dataset_key, model_name, seed, data_dir, out_dir, log, grid_size=None, spline_order=None):
    layout, batch_size = MODELS[model_name]
    tag = ""
    if grid_size is not None:
        tag += f"__g{grid_size}"
    if spline_order is not None:
        tag += f"__k{spline_order}"
    out_path = os.path.join(out_dir, dataset_key, f"{slug(model_name)}{tag}__seed{seed}.csv")
    if os.path.exists(out_path):
        log(f"skip (exists): {out_path}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} dataset={dataset_key} model={model_name} seed={seed} grid_size={grid_size} spline_order={spline_order}")
    if device.type != "cuda":
        raise SystemExit("CUDA is not available; refusing to run on CPU.")

    seed_everything(seed)
    ds = MyDataset(os.path.join(data_dir, DATASETS[dataset_key]), model_type=layout)
    x_train, y_train = ds.get_train_set()
    x_test, y_test = ds.get_test_set()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    num_features = x_train.size(-1)
    num_classes = len(torch.unique(y_train))
    log(f"num_features={num_features} num_classes={num_classes} train={len(x_train)} test={len(x_test)}")

    seed_everything(seed)
    model = build_model(model_name, num_features, num_classes, grid_size, spline_order).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()

    hist = {k: [] for k in ("train_loss", "test_loss", "accuracy", "precision", "recall", "f1",
                            "precision_all", "recall_all", "f1_all", "n_predicted_classes")}
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train(model, device, train_loader, optimizer, criterion)
        (te_loss, acc, prec, rec, f1,
         prec_all, rec_all, f1_all, n_pred) = test(model, device, test_loader, criterion)
        for k, v in zip(hist, (tr_loss, te_loss, acc, prec, rec, f1,
                               prec_all, rec_all, f1_all, n_pred)):
            hist[k].append(float(v))
        log(f"epoch {epoch:2d}/{EPOCHS} train_loss={tr_loss:.6f} test_loss={te_loss:.4f} "
            f"acc={acc:.2%} f1={f1:.4f} f1_all={f1_all:.4f} classes_predicted={n_pred}")
        scheduler.step()
    runtime = time.time() - start

    best = int(np.argmax(hist["accuracy"]))  # same selection rule as the notebook
    row = {
        "Dataset": dataset_key,
        "Seed": seed,
        "Model": model_name,
        "Grid Size": grid_size,
        "Spline Order": spline_order,
        "Test Accuracy": hist["accuracy"][best],
        "Test Precision": hist["precision"][best],
        "Test Recall": hist["recall"][best],
        "Test F1 Score": hist["f1"][best],
        "Test Precision (all classes)": hist["precision_all"][best],
        "Test Recall (all classes)": hist["recall_all"][best],
        "Test F1 Score (all classes)": hist["f1_all"][best],
        "Predicted Classes": int(hist["n_predicted_classes"][best]),
        "Test Loss": hist["test_loss"][best],
        "Number of Parameters": count_parameters(model),
        "Best Epoch": best + 1,
        "Runtime (s)": runtime,
        "Device": torch.cuda.get_device_name(0),
    }
    pd.DataFrame([row]).to_csv(out_path, index=False)
    with open(out_path.replace(".csv", "_history.json"), "w") as fh:
        json.dump(hist, fh)
    log(f"done in {runtime:.1f}s -> F1={row['Test F1 Score']:.4f} "
        f"F1_all={row['Test F1 Score (all classes)']:.4f} (epoch {best + 1}); saved {out_path}")


def slug(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "and")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results_seeds"))
    parser.add_argument("--threads", type=int, default=2, help="torch intra-op CPU threads")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="override the KAN grid size (num_grids for the FastKAN model)")
    parser.add_argument("--spline-order", type=int, default=None,
                        help="override the spline order of the KAN layers (no effect on the KAN model)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    run_job(args.dataset, args.model, args.seed, args.data_dir, args.out_dir, log,
            grid_size=args.grid_size, spline_order=args.spline_order)


if __name__ == "__main__":
    main()
