"""Data preprocessing (Methodology, "Data Preprocessing").

* Data transformation: non-numeric feature columns are binarised with one-hot
  encoding, boolean columns become 0/1, and the class label is mapped to the
  integers 0..C-1 (one value per category, in sorted order).
* Data optimisation: integer columns are downcast to the smallest integer type
  that holds their values and floating-point columns are stored as float32.
* Feature selection: Extremely Randomized Trees (scikit-learn
  ``ExtraTreesClassifier``) rank the features by impurity-based importance and
  ``SelectFromModel`` keeps those at or above the configured threshold. The selector is
  fitted on the training split only (see ``prepare_data``).
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from src import data_split


def load_dataset(path, label_column=None, drop_columns=()):
    """Read a CSV file. The label is ``label_column`` or, if None, the last column."""
    df = pd.read_csv(path, low_memory=False)
    if drop_columns:
        df = df.drop(columns=list(drop_columns))
    label = label_column if label_column is not None else df.columns[-1]
    y_raw = df.pop(label)
    return df, y_raw


def encode_labels(y_raw):
    """Map each class to an integer 0..C-1 (sorted order, as LabelEncoder does)."""
    classes, y = np.unique(np.asarray(y_raw), return_inverse=True)
    return y.astype(np.int64), classes


def transform_features(X):
    """One-hot encode non-numeric columns and convert boolean columns to 0/1."""
    categorical = [c for c in X.columns
                   if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
    if categorical:
        X = pd.get_dummies(X, columns=categorical, dtype=np.uint8)
    bool_cols = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(np.uint8)
    return X


def optimise_dtypes(X):
    """Downcast integer columns and store floating-point columns as float32."""
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_integer_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], downcast="integer")
        elif pd.api.types.is_float_dtype(X[c]):
            X[c] = X[c].astype(np.float32)
    return X


class ERTFeatureSelector:
    """Feature selection with Extremely Randomized Trees."""

    def __init__(self, threshold="mean", max_features=None, random_state=None, estimator_params=None):
        self.threshold = threshold
        self.max_features = max_features
        self.random_state = random_state
        self.estimator_params = dict(estimator_params or {})

    def fit(self, X, y):
        estimator = ExtraTreesClassifier(random_state=self.random_state, **self.estimator_params)
        self.selector_ = SelectFromModel(estimator, threshold=self.threshold,
                                         max_features=self.max_features).fit(X, y)
        self.importances_ = self.selector_.estimator_.feature_importances_
        self.support_ = self.selector_.get_support()
        self.selected_ = list(X.columns[self.support_])
        return self

    def transform(self, X):
        return X.loc[:, self.support_]


def prepare_data(cfg, dataset_path, seed, subsample=None, log=print):
    """Full preprocessing and split used by both runners.

    Returns ``X_train, X_test, y_train, y_test`` (float32 / int64 arrays) and a
    dictionary that describes the data actually used.
    """
    pre = cfg["preprocessing"]
    X, y_raw = load_dataset(dataset_path, pre.get("label_column"), pre.get("drop_columns", []))
    y, classes = encode_labels(y_raw)
    X = optimise_dtypes(transform_features(X))
    n_rows, n_features_in = X.shape
    log(f"loaded {dataset_path}: {n_rows} rows, {n_features_in} features after transformation, "
        f"{len(classes)} classes")

    if subsample:
        X, y = data_split.stratified_subsample(X, y, subsample, seed)
        log(f"stratified subsample ({subsample}): {len(y)} rows")

    split = cfg["split"]
    X_train, X_test, y_train, y_test = data_split.train_test(X, y, split["test_size"], split["random_state"])
    log(f"split: {len(y_train)} training rows, {len(y_test)} test rows")

    fs = pre["feature_selection"]
    selected = list(X.columns)
    if fs.get("enabled", True):
        selector = ERTFeatureSelector(fs.get("threshold", "mean"), fs.get("max_features"), seed,
                                      fs.get("estimator_params")).fit(X_train, y_train)
        X_train, X_test = selector.transform(X_train), selector.transform(X_test)
        selected = selector.selected_
        log(f"ERT feature selection: {len(selected)} of {n_features_in} features kept")

    info = {
        "dataset_file": os.path.basename(dataset_path),
        "rows": int(n_rows),
        "rows_used": int(len(y_train) + len(y_test)),
        "features_after_transformation": int(n_features_in),
        "selected_features": selected,
        "classes": [c.item() if hasattr(c, "item") else c for c in classes],
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "subsample": subsample,
        "seed": seed,
    }
    return (X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32),
            np.asarray(y_train), np.asarray(y_test), info)
