"""
ML Assignment 2 - Dry Bean variety classification
Md Abrar Mohiuddin | 2026ak05323 | M.Tech AIML

Trains the five required classifiers on the UCI Dry Bean dataset, reports the six
evaluation metrics for each on a held-out test set, and saves:
    model/<name>.joblib        - one fitted pipeline per classifier (preprocessing included)
    model/metrics_summary.csv  - the comparison table used in the README
    model/metadata.json        - feature list, classes, chosen hyper-parameters, library versions
    test_data.csv              - the held-out 20% (features + Class) to upload into the Streamlit app

Run:  python train_models.py
"""

import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bean_metrics import FEATURES, MODEL_FILES, TARGET, evaluate_model

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "dry_bean.csv"
MODEL_DIR = ROOT / "model"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
SEED = 2026  # fixed seed so the split, the CV folds and every model are reproducible


def load_dataset():
    """Read data/dry_bean.csv; if it is missing, download the official UCI zip and convert the ARFF file once."""
    if not DATA_CSV.exists():
        print("data/dry_bean.csv not found - downloading from UCI ...")
        raw = urllib.request.urlopen(UCI_ZIP, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            arff = zf.read("DryBeanDataset/Dry_Bean_Dataset.arff").decode("utf-8")
        rows = arff.split("@DATA", 1)[1].strip().splitlines()
        df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=FEATURES + [TARGET])
        DATA_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
    return pd.read_csv(DATA_CSV)


def build_candidates():
    """
    Each classifier is wrapped in a Pipeline so the scaler fitted on the TRAINING data
    is saved inside the model file and re-applied automatically at prediction time.

    StandardScaler is essential for the distance/margin based models (KNN, SVM) and for
    Logistic Regression (features range from ~0.001 to ~250,000). The Decision Tree splits
    on one feature at a time and is scale-invariant, so it gets no scaler.

    The grids are kept small on purpose: they tune the one or two knobs that matter most
    for each model, using 5-fold stratified CV on the training split only.
    """
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "Logistic Regression": (
            scaled(LogisticRegression(max_iter=5000, random_state=SEED)),
            {"clf__C": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [6, 8, 10, 12, None], "clf__min_samples_leaf": [1, 5, 10]},
        ),
        "KNN": (
            scaled(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 9, 15, 21], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            scaled(GaussianNB()),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        # probability=True is needed for AUC; it is switched on only for the final refit below
        # because Platt scaling inside every CV fit would multiply the search time by ~5.
        "Support Vector Machine": (
            scaled(SVC(kernel="rbf", random_state=SEED)),
            {"clf__C": [1.0, 10.0, 100.0], "clf__gamma": ["scale", 0.1]},
        ),
    }


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns")

    # The raw file contains exact duplicate rows; dropping them before splitting stops the
    # same bean appearing in both train and test, which would inflate the test scores.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {n_before - len(df)} duplicate rows -> {len(df)} unique rows")
    print("Class distribution:\n" + df[TARGET].value_counts().to_string())

    X, y = df[FEATURES], df[TARGET]
    # Stratified 80/20 split keeps each variety's proportion identical in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    # The held-out test split is what gets uploaded into the Streamlit app.
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, chosen = [], {}

    for name, (pipe, grid) in build_candidates().items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        if name == "Support Vector Machine":
            best.set_params(clf__probability=True)
            best.fit(X_train, y_train)

        metrics, _ = evaluate_model(best, X_test, y_test)
        chosen[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rows.append({"ML Model Name": name, **metrics})
        joblib.dump(best, MODEL_DIR / MODEL_FILES[name])
        print(f"{name:<24} best={chosen[name]}  CV-F1={search.best_score_:.4f}  "
              f"test-acc={metrics['Accuracy']:.4f}  ({time.time() - t0:.1f}s)")

    table = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    table.to_csv(MODEL_DIR / "metrics_summary.csv")

    # Saved so the app can show the training-time numbers and warn on a scikit-learn version mismatch.
    meta = {
        "dataset": "UCI Dry Bean Dataset (id 602)",
        "features": FEATURES,
        "target": TARGET,
        "classes": sorted(y.unique().tolist()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": SEED,
        "best_params": chosen,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n=== Test-set comparison (macro-averaged Precision / Recall / F1, one-vs-rest AUC) ===")
    print(table.to_string())
    print(f"\nBest model by F1: {table['F1'].idxmax()}  |  Best by MCC: {table['MCC'].idxmax()}")


if __name__ == "__main__":
    main()
