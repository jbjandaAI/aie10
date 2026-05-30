"""k-Nearest Neighbors — lazy, instance-based classification by majority vote."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from utils.data import get_dataset
from utils.plot import plot_decision, new_fig, grid_fig

st.set_page_config(page_title="k-Nearest Neighbors", page_icon="📍", layout="wide")

st.title("📍 k-Nearest Neighbors")
st.caption("No training, no parameters — just ask the k closest points to vote on every new location.")


# ───────────────────────── Sidebar (shared) ─────────────────────────
with st.sidebar:
    st.header("Controls")
    st.subheader("Train / Val / Test split")
    train_frac = st.slider("Train", 0.3, 0.9, 0.6, step=0.05)
    val_frac = st.slider("Validation", 0.05, 0.5, 0.2, step=0.05)
    test_frac = max(1.0 - train_frac - val_frac, 0.05)
    st.caption(f"Test fraction (auto): **{test_frac:.2f}**")
    seed = st.number_input("Random seed", value=7, step=1)


# ───────────────────────── Hyperparameters ─────────────────────────
st.subheader("Hyperparameters")
st.caption("k-NN does no real *training* — these choose how neighbors are found and how their votes are counted.")

c1, c2, c3, c4 = st.columns(4)
dataset = c1.selectbox("Dataset (2 features)",
                       ["moons", "circles", "blobs", "classification"], index=0)
k = c2.slider("k (n_neighbors)", 1, 51, 15, step=2)
weights = c3.selectbox("weights", ["uniform", "distance"])
metric = c4.selectbox("metric", ["euclidean", "manhattan", "minkowski"])

c5, c6, c7 = st.columns(3)
p = c5.slider("p (minkowski power)", 1, 5, 2, step=1, disabled=metric != "minkowski")
n_samples = c6.slider("Samples", 80, 600, 240, step=20)
noise = c7.slider("Noise", 0.0, 0.6, 0.25, step=0.05)


# ───────────────────────── Data + splits ─────────────────────────
X_cls, y_cls = get_dataset(dataset, n_samples=n_samples, noise=noise, seed=int(seed))
X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
    X_cls, y_cls, test_size=1.0 - train_frac,
    random_state=int(seed), stratify=y_cls,
)
val_relative_c = val_frac / (val_frac + test_frac)
X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
    X_temp_c, y_temp_c, test_size=1.0 - val_relative_c,
    random_state=int(seed), stratify=y_temp_c,
)

# k can't exceed the number of training points we fit on.
k_eff = int(min(k, len(X_train_c)))


def build_knn(n_neighbors: int, w: str = weights, m: str = metric) -> KNeighborsClassifier:
    return KNeighborsClassifier(
        n_neighbors=int(min(n_neighbors, len(X_train_c))),
        weights=w,
        metric=m,
        p=int(p),
    )


def metrics_row_cls(model, X_part, y_part):
    pred = model.predict(X_part)
    row = {
        "Accuracy":  float(accuracy_score(y_part, pred)),
        "Precision": float(precision_score(y_part, pred, zero_division=0)),
        "Recall":    float(recall_score(y_part, pred, zero_division=0)),
        "F1":        float(f1_score(y_part, pred, zero_division=0)),
    }
    try:
        proba = model.predict_proba(X_part)
        row["ROC-AUC"] = float(roc_auc_score(y_part, proba[:, 1]))
        row["LogLoss"] = float(log_loss(y_part, proba, labels=[0, 1]))
    except Exception:
        row["ROC-AUC"] = float("nan")
        row["LogLoss"] = float("nan")
    return row


# ───────────────────────── Fit + render ─────────────────────────
model_cls = build_knn(k_eff).fit(X_train_c, y_train_c)

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    fig, ax = new_fig(6.5, 5.5)
    plot_decision(
        ax, model_cls, X_cls, y_cls,
        title=f"k-NN — k={k_eff}, weights={weights}, metric={metric}",
        show_misclassified=True,
    )
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Shading = share of the k nearest neighbors voting for each class "
        "(the predicted probability). Red outlines = misclassified points. "
        "There is no smooth equation — the boundary follows the data."
    )

with col2:
    st.markdown("**Metrics on each split**")
    rows = {
        "Train": metrics_row_cls(model_cls, X_train_c, y_train_c),
        "Val":   metrics_row_cls(model_cls, X_val_c,   y_val_c),
        "Test":  metrics_row_cls(model_cls, X_test_c,  y_test_c),
    }
    metrics_df = pd.DataFrame(rows).T[
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "LogLoss"]
    ]
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
    st.caption(
        f"Split sizes — train: {len(X_train_c)}, val: {len(X_val_c)}, test: {len(X_test_c)}"
    )

    st.markdown("**k-NN diagnostics**")
    if k_eff == 1 and weights == "uniform":
        k1_note = "k=1: each train point is its own nearest neighbor → train accuracy ≈ 1.0 (memorizes)"
    else:
        k1_note = "larger k averages more neighbors → smoother boundary, train accuracy drops"
    metric_str = f"{metric} (p={p})" if metric == "minkowski" else metric
    diag_df = pd.DataFrame(
        {
            "value": [
                f"{k_eff}" + ("  (clamped to train size)" if k_eff != k else ""),
                weights,
                metric_str,
                k1_note,
            ]
        },
        index=["effective k", "vote weighting", "distance metric", "what k controls"],
    )
    st.dataframe(diag_df, use_container_width=True)

    st.markdown("**Confusion matrix (test set)**")
    cm = confusion_matrix(y_test_c, model_cls.predict(X_test_c), labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["actual 0", "actual 1"],
        columns=["pred 0", "pred 1"],
    )
    st.dataframe(cm_df, use_container_width=True)

with st.expander("Compare values of k on this dataset", expanded=False):
    k_grid = [1, 5, 15, 31]
    fig_k, axes_k = grid_fig(len(k_grid), ncols=4, cell=3.4)
    for ax_k, kk in zip(axes_k, k_grid):
        m_k = build_knn(kk).fit(X_train_c, y_train_c)
        plot_decision(ax_k, m_k, X_cls, y_cls,
                      title=f"k={min(kk, len(X_train_c))}", show_misclassified=False)
    st.pyplot(fig_k, use_container_width=True)
    st.caption(
        "Same dataset, weighting, and metric — only k changes. Small k → jagged, "
        "high-variance boundary that chases noise. Large k → smooth, higher-bias "
        "boundary that can wash out real structure."
    )

with st.expander("Accuracy vs k (how to choose k)", expanded=False):
    n_train = len(X_train_c)
    k_max = int(min(51, n_train))
    ks = list(range(1, k_max + 1, 2))
    train_acc, val_acc = [], []
    for kk in ks:
        m_kk = build_knn(kk).fit(X_train_c, y_train_c)
        train_acc.append(accuracy_score(y_train_c, m_kk.predict(X_train_c)))
        val_acc.append(accuracy_score(y_val_c, m_kk.predict(X_val_c)))

    best_i = int(np.argmax(val_acc))
    best_k = ks[best_i]

    fig_v, ax_v = new_fig(7, 4)
    ax_v.plot(ks, train_acc, color="#4c78a8", marker="o", markersize=3,
              linewidth=1.8, label="train")
    ax_v.plot(ks, val_acc, color="#e45756", marker="o", markersize=3,
              linewidth=1.8, label="validation")
    ax_v.axvline(best_k, color="#888", linestyle="--", linewidth=1.0)
    ax_v.scatter([best_k], [val_acc[best_i]], s=90, facecolors="none",
                 edgecolors="#111", linewidths=1.6, zorder=5)
    ax_v.set_xlabel("k (n_neighbors)", fontsize=10, color="#555")
    ax_v.set_ylabel("accuracy", fontsize=10, color="#555")
    for s in ("top", "right"):
        ax_v.spines[s].set_visible(False)
    ax_v.spines["left"].set_color("#bbb")
    ax_v.spines["bottom"].set_color("#bbb")
    ax_v.tick_params(colors="#888", labelsize=8)
    ax_v.legend(loc="lower left", frameon=False, fontsize=9)
    ax_v.set_title(f"Best validation accuracy at k={best_k}", fontsize=11)
    st.pyplot(fig_v, use_container_width=True)
    st.caption(
        "Train accuracy starts near 1.0 at k=1 (every point predicts itself) and "
        "falls as k grows. Validation accuracy peaks at an intermediate k — the "
        "sweet spot between memorizing noise (small k) and over-smoothing (large k)."
    )


# ───────────────────────── Explanation ─────────────────────────
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
**k-Nearest Neighbors** is the laziest classifier there is: it doesn't *learn* a
model at all. "Training" just stores the data. To classify a new point, it finds
the **k closest training points** and lets them vote — the majority class wins.
All the work happens at *prediction* time.

Because the prediction at any location depends only on whichever training points
are nearest, the decision boundary is an implicit **Voronoi-like partition** of
the plane — a patchwork of regions stitched directly to the data, with no smooth
equation behind it.

- **`k`** is the entire bias/variance dial. **`k=1`** memorizes the training set
  (each point is its own nearest neighbor → ~100% train accuracy, jagged
  boundary, high variance). Large **`k`** averages over a big neighborhood →
  smooth boundary, more bias, and at the extreme it just predicts the majority
  class everywhere. The *Accuracy vs k* panel shows this trade-off directly.
- **`weights`** decides how votes are counted. `uniform` gives every one of the k
  neighbors an equal vote; `distance` weights each vote by `1 / distance`, so
  nearer neighbors count more — useful when the right answer is dominated by the
  very closest points.
- **`metric` / `p`** defines "closest." `euclidean` is straight-line distance;
  `manhattan` sums absolute coordinate differences (grid-like); `minkowski` is
  the general form where `p=1` is Manhattan and `p=2` is Euclidean.
- **Feature scaling matters.** Distances are dominated by whichever feature has
  the largest range, so a feature measured in thousands would drown out one
  measured in tenths. These synthetic datasets are already standardized in
  `get_dataset`, so both axes contribute fairly.

**Synthetic data.** `make_moons` (interleaved crescents), `make_circles`
(concentric rings), `make_blobs` (well-separated Gaussians), and
`make_classification` (an informative two-feature problem). k-NN handles the
curved, non-linear shapes effortlessly — bending around moons and circles is
exactly what a data-hugging boundary is good at.
"""
    )
