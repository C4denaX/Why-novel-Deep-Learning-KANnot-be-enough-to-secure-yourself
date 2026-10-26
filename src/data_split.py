"""Data split (Methodology, "Model Training"): a stratified 70/30 split into a
training set and a held-out test set, and stratified k-fold cross-validation
(five folds by default) on the training set for the grid search."""
from sklearn.model_selection import StratifiedKFold, train_test_split


def train_test(X, y, test_size=0.3, random_state=42):
    """Stratified, shuffled split; the same call as in the experiment code."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)


def cv_folds(n_splits, seed):
    """Stratified k-fold splitter used by the grid searches of every model."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def stratified_subsample(X, y, fraction, seed):
    """Keep a stratified fraction of the rows (only for quick checks of the pipeline)."""
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=fraction, random_state=seed, shuffle=True, stratify=y)
    return X_sub, y_sub
