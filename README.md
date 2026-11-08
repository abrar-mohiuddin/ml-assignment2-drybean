# Dry Bean Variety Classification — ML Assignment 2

**Md Abrar Mohiuddin · BITS ID 2026ak05323 · M.Tech (AIML), WILP**

- **GitHub repository:** <GITHUB_REPO_URL>
- **Live Streamlit app:** <STREAMLIT_APP_URL>

---

## a. Problem statement

Dry beans are the most widely produced pulse crop, and seed quality and market price both depend on
selling a **pure, correctly labelled variety**. Sorting beans by eye is slow and inconsistent.
The goal of this project is to **predict which of seven registered dry-bean varieties a single grain
belongs to**, using only 16 shape and size measurements taken from a camera image of that grain.

This is a **multi-class classification** problem (7 classes). Five classical classifiers are trained
on the same data, compared on six evaluation metrics, and served through an interactive Streamlit
app that scores the saved models on an uploaded test file **without retraining them**.

## b. Dataset description

| Property | Value |
|---|---|
| Name / source | **Dry Bean Dataset**, UCI Machine Learning Repository (id 602) — Koklu, M. & Ozkan, I. A. (2020), *Computers and Electronics in Agriculture* 174, 105507 |
| Instances | 13,611 grains (13,543 after removing 68 exact duplicate rows) |
| Features | **16**, all numeric (12 dimensional + 4 shape factors) |
| Target | `Class` — 7 varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Missing values | None |

**Features.** `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation` (major/minor
axis), `Eccentricity`, `ConvexArea`, `EquivDiameter` (diameter of a circle with the same area),
`Extent` (area / bounding-box area), `Solidity` (area / convex area), `roundness`, `Compactness`,
and `ShapeFactor1`–`ShapeFactor4`.

**Class distribution (after de-duplication)** — the classes are imbalanced, which is why
Precision, Recall and F1 are reported as **macro** averages and MCC is included:

| DERMASON | SIRA | SEKER | HOROZ | CALI | BARBUNYA | BOMBAY |
|---:|---:|---:|---:|---:|---:|---:|
| 3,546 | 2,636 | 2,027 | 1,860 | 1,630 | 1,322 | 522 |

**Pre-processing and split.**
1. Exact duplicate rows were dropped **before** splitting, so the same grain cannot appear in both
   train and test.
2. Stratified 80 / 20 split (seed 2026): **10,834 training rows, 2,709 test rows**. The test rows are
   saved as `test_data.csv` and are the file uploaded into the app.
3. A `StandardScaler` is fitted on the training rows only and stored **inside each model pipeline**
   (not used for the Decision Tree, which is scale-invariant). Feature magnitudes range from ~0.0006
   (`ShapeFactor2`) to ~250,000 (`Area`), so scaling is essential for LR, KNN and SVM.
4. Hyper-parameters were chosen by **5-fold stratified cross-validation on the training split only**,
   optimising macro-F1. The test split was used once, for the final numbers below.

## c. GitHub repository link

**<GITHUB_REPO_URL>**

```
ml-assignment2-drybean/
├── app.py                  # Streamlit app (loads saved models, never retrains)
├── train_models.py         # trains + tunes the 5 models, writes model/ and test_data.csv
├── bean_metrics.py         # feature list + metric functions shared by training and the app
├── requirements.txt
├── README.md
├── test_data.csv           # 2,709-row held-out test split (16 features + Class)
├── data/dry_bean.csv       # full dataset converted from the UCI ARFF file
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metrics_summary.csv # the comparison table below
│   └── metadata.json       # classes, tuned hyper-parameters, library versions
└── .streamlit/config.toml  # app theme
```

Run locally: `pip install -r requirements.txt` → `python train_models.py` → `streamlit run app.py`.

## d. Models used

All five models are trained on the same 10,834 training rows and evaluated on the same 2,709 test rows.

| Model | Tuned hyper-parameters (5-fold CV) |
|---|---|
| Logistic Regression | multinomial, `C = 1.0`, scaled features |
| Decision Tree | `max_depth = 10`, `min_samples_leaf = 10` (195 leaves) |
| KNN | `k = 9`, `weights = distance`, scaled features |
| Naive Bayes | Gaussian, `var_smoothing = 1e-9`, scaled features |
| Support Vector Machine | RBF kernel, `C = 10`, `gamma = 0.1`, scaled features, Platt-scaled probabilities |

### Comparison table — test set (2,709 rows)

Precision, Recall and F1 are macro-averaged over the 7 classes; AUC is one-vs-rest, macro-averaged;
MCC is the multi-class Matthews correlation coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9251 | **0.9947** | 0.9373 | 0.9375 | 0.9374 | 0.9094 |
| Decision Tree | 0.9110 | 0.9821 | 0.9252 | 0.9232 | 0.9241 | 0.8924 |
| KNN | 0.9228 | 0.9882 | 0.9374 | 0.9364 | 0.9369 | 0.9067 |
| Naive Bayes | 0.8989 | 0.9914 | 0.9065 | 0.9120 | 0.9089 | 0.8781 |
| Support Vector Machine | **0.9280** | 0.9939 | **0.9407** | **0.9397** | **0.9402** | **0.9129** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A very strong linear baseline: accuracy 0.9251 and the **highest AUC (0.9947)**, meaning its probability ranking separates the varieties best of all five. Training and test accuracy are identical (0.9251 vs 0.9251), so it does not overfit. After scaling, the varieties are almost linearly separable in the 16 shape features. Nearly all of its errors are SIRA ↔ DERMASON (46 each way), the two most similar-looking varieties. |
| Decision Tree | The weakest model apart from Naive Bayes (accuracy 0.9110, MCC 0.8924), and the **lowest AUC (0.9821)**: each leaf outputs one fixed class proportion, so its probabilities are coarse and rank poorly. There is mild overfitting (training 0.9421 vs test 0.9110) even after CV capped the depth at 10. It is the most interpretable model: `ShapeFactor1` (importance 0.33), `Compactness` (0.26) and `MajorAxisLength` (0.19) drive most splits. |
| KNN | Close to Logistic Regression (F1 0.9369 vs 0.9374), which shows that the scaled feature space clusters well by variety. With distance weighting it reproduces its training set perfectly (training accuracy 1.000 vs test 0.9228), which is expected for KNN rather than a defect. It needs feature scaling, keeps the whole training set inside the model (1.5 MB, the largest file), and is the slowest to predict. |
| Naive Bayes | The **lowest on every threshold metric** (accuracy 0.8989, F1 0.9089, MCC 0.8781). Its independence assumption is badly violated here: `Area` and `ConvexArea` correlate at r = 1.000, and `Perimeter` and `EquivDiameter` at r = 0.991, so the same evidence is counted several times. Even so, its **AUC is high (0.9914)**: it ranks the classes well but places the decision boundaries poorly. It is the only model with a notable BARBUNYA → CALI error (29 grains). Training is the fastest (0.3 s). |
| Support Vector Machine | **Best on Accuracy (0.9280), Precision (0.9407), Recall (0.9397), F1 (0.9402) and MCC (0.9129)**, and second on AUC (0.9939, only 0.0008 behind LR). The RBF kernel bends the SIRA/DERMASON boundary slightly better than a linear model can. Its small train–test gap (0.9417 vs 0.9280) shows good generalisation. It is the most expensive to train (≈11 s including probability calibration). |
| **Overall winner for this dataset** | **Support Vector Machine (RBF kernel)**: it wins 5 of the 6 metrics, including MCC, the most informative single score for imbalanced multi-class data. Its lead over Logistic Regression is small (+0.3 points accuracy, +0.0035 MCC), so LR is a sensible lighter-weight alternative. Across every model, **BOMBAY is easy** (per-class F1 0.99–1.00: its mean area is 173,485 px against at most 75,538 for any other variety) and **SIRA is the hardest** (F1 0.845–0.873, confused mostly with DERMASON). |

## Streamlit app features

- **CSV upload** of test data (the bundled `test_data.csv` is used until a file is uploaded, and can be
  downloaded from the sidebar). Columns and labels are validated, with a clear message if anything is missing.
- **Model selection dropdown** covering all five classifiers.
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) computed live on the uploaded data.
- **Confusion matrix** (counts or row %) and the **classification report** per class.
- **Compare all models** tab: all five side by side on the uploaded data, with a configurable bar chart and CSV export.
- **Predictions** tab: row-level actual vs predicted, a filter for misclassified rows, and CSV download.
