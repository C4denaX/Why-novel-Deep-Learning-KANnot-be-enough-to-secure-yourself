"""Shared helpers: configuration, seeds, devices, logging and output files."""
import json
import os
import random
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DEFAULT_CONFIG = os.path.join(ROOT, "src", "config.json")


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return json.load(fh)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed):
    """Seed Python, NumPy and, when installed, PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def resolve_device(choice):
    """'auto' selects CUDA when it is available and the CPU otherwise."""
    if choice == "auto":
        return "cuda" if cuda_available() else "cpu"
    if choice == "cuda" and not cuda_available():
        raise SystemExit("--device cuda was requested but CUDA is not available")
    return choice


def write_results(rows, columns, path, model_order):
    """Write one CSV per dataset. Rows of models that are already in the file are
    replaced, so the models of a dataset can be run in separate invocations."""
    new = pd.DataFrame(rows, columns=columns)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old["Model"].isin(new["Model"])]
        new = pd.concat([old, new], ignore_index=True)
    rank = {m: i for i, m in enumerate(model_order)}
    new = new.sort_values("Model", key=lambda s: s.map(lambda m: rank.get(m, len(rank))), kind="stable")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new.to_csv(path, index=False)


def update_json(path, key, value, data_info):
    """Store the search details of one model under ``key`` in a per-dataset JSON file."""
    content = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            content = json.load(fh)
    content["data"] = data_info
    content.setdefault("models", {})[key] = value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, default=_to_builtin)


def _to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not serialisable: {type(obj)}")
