"""Logistic Regression — linear decision boundary in 2D, with held-out metrics."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from utils.data import get_dataset
from utils.plot import plot_decision, new_fig

st.set_page_config(page_title="Logistic Regression", page_icon="🎯", layout="wide")

st.title("🎯 Logistic Regression")
st.caption("A linear classifier: sigmoid of (w·x + b) gives class probability.")


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("Controls")
    anim_mode = st.selectbox(
        "Animation mode",
        ["Static", "Iterations (SGD)", "Regularization sweep"],
    )
    play = st.toggle("▶ Auto-play", value=False, disabled=anim_mode == "Static")
    speed = st.slider("Frame delay (s)", 0.05, 1.0, 0.25, step=0.05,
                      disabled=anim_mode == "Static" or not play)
    n_frames = st.slider("Frames (animation length)", 5, 60, 25, step=1,
                         disabled=anim_mode == "Static")
    st.markdown("---")
    st.subheader("Train / Val / Test split")
    train_frac = st.slider("Train", 0.3, 0.9, 0.6, step=0.05)
    val_frac = st.slider("Validation", 0.05, 0.5, 0.2, step=0.05)
    test_frac = max(1.0 - train_frac - val_frac, 0.05)
    st.caption(f"Test fraction (auto): **{test_frac:.2f}**")
    seed = st.number_input("Random seed", value=7, step=1)


# ───────────────────────── Hyperparameters ─────────────────────────
st.subheader("Hyperparameters")
st.caption("Values you set **before** training. They control model capacity and how it's fit.")

h1, h2, h3, h4 = st.columns(4)
dataset = h1.selectbox("Dataset (2 features)", ["moons", "circles", "blobs"], index=2)
PENALTY_LABELS = {"l2 (Ridge)": "l2", "l1 (Lasso)": "l1", "none": "none"}
penalty_label = h2.selectbox("penalty", list(PENALTY_LABELS.keys()))
penalty = PENALTY_LABELS[penalty_label]
C = h3.slider("C (inverse regularization)", 0.01, 10.0, 1.0, step=0.01,
              disabled=penalty == "none")
fit_intercept = h4.toggle("fit_intercept", value=True)

h5, h6, h7, h8 = st.columns(4)
class_weight = h5.selectbox("class_weight", ["None", "balanced"])
n_samples = h6.slider("Samples", 80, 600, 240, step=20)
noise = h7.slider("Noise", 0.0, 0.6, 0.25, step=0.05)
max_iter = h8.slider("max_iter", 50, 2000, 200, step=50)

# Hidden internal defaults for the animation modes (kept off the UI for clarity).
SGD_ETA0 = 0.02
C_SWEEP_MIN = 0.01
C_SWEEP_MAX = 20.0


# ───────────────────────── Data + splits ─────────────────────────
X, y = get_dataset(dataset, n_samples=n_samples, noise=noise, seed=int(seed))
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=1.0 - train_frac, random_state=int(seed), stratify=y
)
val_relative = val_frac / (val_frac + test_frac)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1.0 - val_relative,
    random_state=int(seed), stratify=y_temp,
)


# ───────────────────────── Model factory ─────────────────────────
SOLVER_FOR = {"l2": "lbfgs", "l1": "liblinear", "none": "lbfgs"}
SGD_PENALTY = {"l2": "l2", "l1": "l1", "none": None}
CW = None if class_weight == "None" else "balanced"


def build_closed_form(C_value: float):
    pen_arg = penalty if penalty != "none" else None
    return LogisticRegression(
        penalty=pen_arg,
        C=C_value,
        solver=SOLVER_FOR[penalty],
        fit_intercept=fit_intercept,
        class_weight=CW,
        max_iter=int(max_iter),
        random_state=int(seed),
    )


def build_sgd():
    return SGDClassifier(
        loss="log_loss",
        penalty=SGD_PENALTY[penalty],
        alpha=1.0 / max(C, 1e-4) if penalty != "none" else 1e-4,
        fit_intercept=fit_intercept,
        learning_rate="constant",
        eta0=SGD_ETA0,
        class_weight=CW,
        random_state=int(seed),
        warm_start=True,
        max_iter=1,
        tol=None,
        shuffle=True,
    )


def metrics_row(model, X_part, y_part):
    pred = model.predict(X_part)
    row = {
        "Accuracy":  float(accuracy_score(y_part, pred)),
        "Precision": float(precision_score(y_part, pred, zero_division=0)),
        "Recall":    float(recall_score(y_part, pred, zero_division=0)),
        "F1":        float(f1_score(y_part, pred, zero_division=0)),
    }
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_part)[:, 1]
        else:
            scores = model.decision_function(X_part)
            proba = 1.0 / (1.0 + np.exp(-scores))
        row["ROC-AUC"] = float(roc_auc_score(y_part, proba))
    except Exception:
        row["ROC-AUC"] = float("nan")
    return row


# ───────────────────────── Render ─────────────────────────
placeholder = st.empty()


def render(model, *, frame_caption: str = ""):
    w = np.ravel(model.coef_)
    b = float(np.ravel(model.intercept_)[0]) if fit_intercept else 0.0

    col1, col2 = placeholder.container().columns([1.1, 1], gap="large")

    with col1:
        fig, ax = new_fig(6.5, 5.5)
        title = f"P(y=1) = σ({w[0]:+.2f}·x₁ {w[1]:+.2f}·x₂ {b:+.2f})"
        if frame_caption:
            title = f"{frame_caption}\n{title}"
        plot_decision(ax, model, X, y, title=title, show_misclassified=False)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**Parameters** (learned during training)")
        params_df = pd.DataFrame(
            {"value": [w[0], w[1], b]},
            index=["w₁ (coef for x₁)", "w₂ (coef for x₂)", "b (intercept)"],
        )
        st.dataframe(params_df.style.format("{:+.4f}"),
                     use_container_width=True)

        st.markdown("**Metrics on each split**")
        rows = {
            "Train": metrics_row(model, X_train, y_train),
            "Val":   metrics_row(model, X_val,   y_val),
            "Test":  metrics_row(model, X_test,  y_test),
        }
        metrics_df = pd.DataFrame(rows).T[
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        ]
        st.dataframe(metrics_df.style.format("{:.4f}"),
                     use_container_width=True)
        st.caption(
            f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
        )

        st.markdown("**Confusion matrix (test set)**")
        cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["actual 0", "actual 1"],
            columns=["pred 0", "pred 1"],
        )
        st.dataframe(cm_df, use_container_width=True)


# ───────────────────────── Drive the animation ─────────────────────────
if anim_mode == "Static":
    model = build_closed_form(C).fit(X_train, y_train)
    render(model)

elif anim_mode == "Iterations (SGD)":
    sgd = build_sgd()
    # SGDClassifier.partial_fit needs the classes on first call
    classes_arr = np.array([0, 1])
    if play:
        for ep in range(1, n_frames + 1):
            sgd.partial_fit(X_train, y_train, classes=classes_arr)
            render(sgd, frame_caption=f"SGD epoch {ep}/{n_frames}")
            time.sleep(speed)
    else:
        chosen_ep = st.slider("Epoch", 1, n_frames, n_frames, key="logr_iter_step")
        for _ in range(chosen_ep):
            sgd.partial_fit(X_train, y_train, classes=classes_arr)
        render(sgd, frame_caption=f"SGD epoch {chosen_ep}/{n_frames}")

else:  # Regularization sweep
    Cs = np.logspace(np.log10(C_SWEEP_MIN), np.log10(C_SWEEP_MAX), n_frames)
    if play:
        for c_val in Cs:
            m = build_closed_form(float(c_val)).fit(X_train, y_train)
            render(m, frame_caption=f"C = {c_val:.4g}")
            time.sleep(speed)
    else:
        idx = st.slider("Sweep step", 1, n_frames, n_frames, key="logr_sweep_step")
        c_val = float(Cs[idx - 1])
        m = build_closed_form(c_val).fit(X_train, y_train)
        render(m, frame_caption=f"C = {c_val:.4g}")


# ───────────────────────── Explanation ─────────────────────────
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
**Logistic regression** models the log-odds of class 1 as a *linear*
function of the features:

```
log( p / (1 − p) ) = w₁·x₁ + w₂·x₂ + b
```

Passing that through the **sigmoid** gives a probability in (0, 1). The
**decision boundary** — the locus where `p = 0.5` — is the straight line
`w₁·x₁ + w₂·x₂ + b = 0`. That's why the shaded region in the plot is split
by a single line: logistic regression is fundamentally linear.

- **Hyperparameters** are set before fit: the `penalty` (l2 / l1 / none),
  the inverse-regularization strength `C` (smaller `C` = stronger
  regularization), the solver, class weighting, max iterations, and for
  SGD, the learning rate.
- **Parameters** that training learns: just `w₁`, `w₂`, and `b` — three
  numbers, exactly the line you see.
- **moons / circles**: the true class structure isn't linearly separable,
  so a straight line can only approximate it — useful for seeing the
  *limits* of a linear model.
- **blobs**: linearly separable, so accuracy can reach ~1.0.
- **Train / Val / Test**: only train is used for fitting. Validation lets
  you compare `C`, `penalty`, or epoch counts. Test is the final estimate.
- **Metrics**: Accuracy can mislead under class imbalance; Precision /
  Recall / F1 / ROC-AUC give a more complete picture. The confusion
  matrix breaks down the actual vs. predicted counts on the test set.
"""
    )
