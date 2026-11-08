"""
Dry Bean Variety Classifier - Streamlit front end for ML Assignment 2
Md Abrar Mohiuddin | 2026ak05323

The app never retrains anything: it loads the five pipelines saved by train_models.py,
scores them on the CSV the user uploads, and shows the metrics, the confusion matrix,
the classification report and a side-by-side comparison of all models.
"""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st

from bean_metrics import FEATURES, METRIC_NAMES, MODEL_FILES, TARGET, confusion_and_report, evaluate_model

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier | 2026ak05323", page_icon="🫘", layout="wide")

# Small amount of custom CSS: a coloured header band and tighter metric cards.
st.markdown(
    """
    <style>
      .bean-hero {background: linear-gradient(90deg,#5b3a1e 0%,#a0522d 55%,#d2a679 100%);
                  padding: 1.1rem 1.4rem; border-radius: 14px; color: #fff8ee; margin-bottom: 1rem;}
      .bean-hero h1 {color:#fff8ee; margin:0; font-size:1.9rem;}
      .bean-hero p  {margin:.3rem 0 0 0; opacity:.92;}
      div[data-testid="stMetric"] {background:#fbf4ea; border:1px solid #ead7bf; border-radius:12px; padding:.6rem .8rem;}
      div[data-testid="stMetric"] * {color:#3b2412 !important;}
    </style>
    <div class="bean-hero">
      <h1>🫘 Dry Bean Variety Classifier</h1>
      <p>Seven Turkish dry-bean varieties identified from 16 shape measurements of a single grain image -
         five classical ML models, evaluated live on your test data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load every saved pipeline once per server process (cached across reruns and users)."""
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text())


@st.cache_data
def read_csv(raw: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(raw))


@st.cache_data(show_spinner="Scoring all five models on the uploaded data ...")
def score_all(data_hash: str, raw: bytes):
    """
    Evaluate every model on the uploaded CSV. Cached by the file's hash, so switching the
    dropdown between models is instant and the work is done only once per upload.
    """
    df = read_csv(raw)
    X, y = df[FEATURES], df[TARGET].astype(str).str.strip().str.upper()
    results = {}
    for name, model in load_models().items():
        metrics, y_pred = evaluate_model(model, X, y)
        labels, cm, report = confusion_and_report(model, y, y_pred)
        results[name] = {"metrics": metrics, "y_pred": y_pred, "labels": labels, "cm": cm, "report": report}
    return results, y


meta = load_metadata()
models = load_models()

# ---------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload a test CSV (16 features + Class)", type=["csv"])
    if uploaded is None:
        st.caption("No file uploaded yet - the bundled test_data.csv is used until you upload one.")
    st.download_button(
        "⬇️ Download sample test_data.csv", SAMPLE_CSV.read_bytes(), "test_data.csv", "text/csv",
        help="The 20% hold-out split created by train_models.py - never seen during training.",
    )

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    normalise = st.checkbox("Show confusion matrix as row %", value=False)

    st.divider()
    st.caption("Md Abrar Mohiuddin · 2026ak05323\n\nM.Tech AIML · ML Assignment 2")

if meta.get("sklearn_version") != sklearn.__version__:
    st.warning(
        f"Models were saved with scikit-learn {meta.get('sklearn_version')} but this server runs "
        f"{sklearn.__version__}. Pin the training version in requirements.txt to avoid subtle differences."
    )

# ---------------------------------------------------------------- choose the data source
if uploaded is not None:
    raw, source = uploaded.getvalue(), f"uploaded file **{uploaded.name}**"
else:
    raw, source = SAMPLE_CSV.read_bytes(), "bundled **test_data.csv**"

try:
    df = read_csv(raw)
except Exception as exc:  # malformed CSV - tell the user instead of showing a stack trace
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error("The CSV is missing required column(s): " + ", ".join(missing)
             + ". Download the sample file from the sidebar to see the expected layout.")
    st.stop()
if df[FEATURES].isna().any().any():
    st.error("Some feature values are empty. Please remove or fill the incomplete rows and upload again.")
    st.stop()

unknown = sorted(set(df[TARGET].astype(str).str.strip().str.upper()) - set(meta["classes"]))
if unknown:
    st.error(f"Unknown class label(s) in '{TARGET}': {unknown}. Expected one of {meta['classes']}.")
    st.stop()

results, y_true = score_all(hashlib.sha256(raw).hexdigest(), raw)
st.info(f"Evaluating on the {source}: **{len(df):,} rows**, {df[TARGET].nunique()} of 7 varieties present.")

tab_eval, tab_compare, tab_pred, tab_about = st.tabs(
    ["📊 Model evaluation", "🏆 Compare all models", "🔎 Predictions", "ℹ️ About the data"]
)

# ---------------------------------------------------------------- tab 1: one model in depth
with tab_eval:
    res = results[model_name]
    st.subheader(f"{model_name}")
    params = meta["best_params"].get(model_name, {})
    st.caption("Tuned hyper-parameters (5-fold CV on training data): "
               + ", ".join(f"`{k}={v}`" for k, v in params.items()))

    cols = st.columns(6)
    for col, metric in zip(cols, METRIC_NAMES):
        col.metric(metric, f"{res['metrics'][metric]:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged over the 7 classes; AUC is one-vs-rest, macro-averaged.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
        cm = res["cm"].astype(float)
        if normalise:
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sns.heatmap(cm, annot=True, fmt=".1f" if normalise else ".0f", cmap="YlOrBr", cbar=False,
                    xticklabels=res["labels"], yticklabels=res["labels"], linewidths=.5, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        rep = pd.DataFrame(res["report"]).T
        rep = rep.drop(index="accuracy", errors="ignore")
        rep["support"] = rep["support"].astype(int)
        st.dataframe(rep.round(4), height=390)

        # Point out the single most frequent mistake - usually SIRA <-> DERMASON for this dataset.
        off = res["cm"].copy(); np.fill_diagonal(off, 0)
        if off.max() > 0:
            i, j = np.unravel_index(off.argmax(), off.shape)
            st.markdown(f"Most frequent error: **{off[i, j]}** `{res['labels'][i]}` beans predicted as "
                        f"`{res['labels'][j]}`.")

# ---------------------------------------------------------------- tab 2: all five side by side
with tab_compare:
    st.subheader("All five models on the same test data")
    table = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T[METRIC_NAMES]
    table.index.name = "ML Model Name"
    st.dataframe(
        table,
        column_config={m: st.column_config.NumberColumn(m, format="%.4f") for m in METRIC_NAMES},
    )
    best_f1, best_mcc = table["F1"].idxmax(), table["MCC"].idxmax()
    st.success(f"Best macro-F1: **{best_f1}** ({table.loc[best_f1, 'F1']:.4f}) · "
               f"Best MCC: **{best_mcc}** ({table.loc[best_mcc, 'MCC']:.4f})")

    pick = st.multiselect("Metrics to plot", METRIC_NAMES, default=["Accuracy", "F1", "MCC"])
    if pick:
        long = table[pick].reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=long, x="ML Model Name", y="Score", hue="Metric", palette="copper", ax=ax)
        ax.set_ylim(max(0.0, long["Score"].min() - 0.05), 1.0)
        ax.set_xlabel(""); ax.legend(loc="lower right", ncol=len(pick))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.download_button("⬇️ Download comparison table (CSV)", table.round(4).to_csv().encode(),
                       "model_comparison.csv", "text/csv")

# ---------------------------------------------------------------- tab 3: row-level predictions
with tab_pred:
    st.subheader(f"Row-level predictions - {model_name}")
    out = df[FEATURES].copy()
    out.insert(0, "Actual", y_true.values)
    out.insert(1, "Predicted", results[model_name]["y_pred"])
    out.insert(2, "Correct", out["Actual"] == out["Predicted"])
    only_wrong = st.checkbox("Show misclassified rows only")
    view = out[~out["Correct"]] if only_wrong else out
    st.caption(f"{(~out['Correct']).sum()} of {len(out)} rows misclassified.")
    st.dataframe(view, height=420)
    st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                       f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv", "text/csv")

# ---------------------------------------------------------------- tab 4: context
with tab_about:
    st.subheader("Dataset")
    st.markdown(
        f"""
- **Source:** UCI Machine Learning Repository - *Dry Bean Dataset* (Koklu & Ozkan, 2020).
- **Task:** multi-class classification into 7 varieties: {", ".join(meta["classes"])}.
- **Features (16):** 12 size/dimension measures (Area, Perimeter, axis lengths, ConvexArea,
  EquivDiameter, ...) and 4 shape factors, all computed from a high-resolution image of one grain.
- **Training split:** {meta['n_train']:,} rows · **held-out test split:** {meta['n_test']:,} rows
  (stratified 80/20, duplicates removed first, seed {meta['random_state']}).
"""
    )
    st.markdown("**Class balance in the data being evaluated**")
    st.bar_chart(y_true.value_counts().sort_values(ascending=False))
    st.markdown("**Reference: metrics recorded at training time on the hold-out split**")
    st.dataframe(pd.read_csv(MODEL_DIR / "metrics_summary.csv", index_col=0))
