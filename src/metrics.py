"""Evaluation metrics (Methodology, "Model Validation").

Both functions keep the computation of the experiment code unchanged:

* ``dl_metrics``: precision, recall and F1-Score as computed by the ``test``
  function used for the DL and KAN models, i.e. scikit-learn macro averages with
  ``labels`` set to the classes that appear in the predictions.
* ``ml_metrics``: accuracy and scikit-learn macro averages (``average="macro"``)
  as computed for the ML models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def dl_metrics(targets, predictions):
    labels = np.unique(predictions)
    precision = precision_score(targets, predictions, average="macro", labels=labels)
    recall = recall_score(targets, predictions, average="macro", labels=labels)
    f1 = f1_score(targets, predictions, average="macro", labels=labels)
    return precision, recall, f1


def ml_metrics(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro"),
            recall_score(y_true, y_pred, average="macro"),
            f1_score(y_true, y_pred, average="macro"))
