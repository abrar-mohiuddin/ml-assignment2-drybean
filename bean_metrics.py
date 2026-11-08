"""
Shared definitions for the Dry Bean classification project.

Both the training script (train_models.py) and the Streamlit app (app.py)
import from this file, so the feature list, the model file names and the
way every metric is computed are guaranteed to be identical in both places.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The 16 morphological features extracted from bean images (12 dimensions + 4 shape factors).
FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
    "ShapeFactor4",
]
TARGET = "Class"

# Display name -> file name inside model/. The order here is the order used in tables.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def macro_ovr_auc(y_true, proba, classes):
    """
    Multi-class AUC computed one-vs-rest and macro-averaged.

    Each class c is scored as a binary problem (c vs. everything else) using the
    predicted probability of c, and the per-class AUCs are averaged.
    A class that is absent from y_true (possible when a user uploads a small CSV)
    has no positives, so its AUC is undefined and it is skipped instead of crashing.
    """
    y_true = np.asarray(y_true)
    aucs = []
    for i, c in enumerate(classes):
        is_c = (y_true == c)
        if is_c.any() and (~is_c).any():
            aucs.append(roc_auc_score(is_c, proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_model(model, X, y):
    """
    Score an already-trained model on (X, y) and return the six required metrics.

    Precision, Recall and F1 are macro-averaged: every bean variety counts equally,
    which matters here because the classes are imbalanced (BOMBAY ~4%, DERMASON ~26%).
    MCC uses the multi-class generalisation built into scikit-learn.
    """
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": macro_ovr_auc(y, proba, model.classes_),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


def confusion_and_report(model, y, y_pred):
    """Confusion matrix (rows = actual, columns = predicted) and the per-class report as a dict."""
    labels = list(model.classes_)
    cm = confusion_matrix(y, y_pred, labels=labels)
    report = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
    return labels, cm, report
