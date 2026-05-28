"""Support Vector Machines — max-margin classification and ε-insensitive regression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, hinge_loss,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.model_selection import train_test_split

from utils.data import get_dataset, get_svr_dataset, mesh_grid
from utils.plot import plot_decision, plot_regression, new_fig, grid_fig

st.set_page_config(page_title="Support Vector Machines", page_icon="🧭", layout="wide")

st.title("🧭 Support Vector Machines")
st.caption("Fit the widest possible margin between classes — or an ε-insensitive tube around a regression.")


# ───────────────────────── Sidebar (shared) ─────────────────────────
with st.sidebar:
    st.header("Controls")
    st.subheader("Train / Val / Test split")
    train_frac = st.slider("Train", 0.3, 0.9, 0.6, step=0.05)
    val_frac = st.slider("Validation", 0.05, 0.5, 0.2, step=0.05)
    test_frac = max(1.0 - train_frac - val_frac, 0.05)
    st.caption(f"Test fraction (auto): **{test_frac:.2f}**")
    seed = st.number_input("Random seed", value=7, step=1)


# ───────────────────────── Overlay helpers ─────────────────────────
def _resolve_gamma(gamma_mode: str, gamma_manual: float):
    return float(gamma_manual) if gamma_mode == "manual" else gamma_mode


def draw_margin_and_support(ax, model, X):
    """Add ±1 margin contours and ring the support vectors on a classifier plot."""
    xx, yy = mesh_grid(X)
    grid = np.c_[xx.ravel(), yy.ravel()]
    df = model.decision_function(grid).reshape(xx.shape)
    ax.contour(
        xx, yy, df,
        levels=[-1, 0, 1],
        colors=["#444", "#222", "#444"],
        linestyles=["--", "-", "--"],
        linewidths=[1.0, 1.4, 1.0],
        alpha=0.9,
    )
    sv = model.support_vectors_
    ax.scatter(
        sv[:, 0], sv[:, 1],
        s=130, facecolors="none", edgecolors="#111",
        linewidths=1.4, zorder=5,
    )


def draw_epsilon_tube(ax, model, X, y, epsilon: float):
    x_flat = np.asarray(X).ravel()
    pad = 0.3 * (x_flat.max() - x_flat.min())
    xs = np.linspace(x_flat.min() - pad, x_flat.max() + pad, 600).reshape(-1, 1)
    ys = model.predict(xs)
    ax.fill_between(
        xs.ravel(), ys - epsilon, ys + epsilon,
        color="#e45756", alpha=0.15, label="ε-tube", zorder=2,
    )
    sv_idx = model.support_
    ax.scatter(
        np.asarray(X)[sv_idx, 0], np.asarray(y)[sv_idx],
        s=130, facecolors="none", edgecolors="#111",
        linewidths=1.4, zorder=5, label="support vector",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)


# ───────────────────────── Tabs ─────────────────────────
tab_cls, tab_reg = st.tabs(["Classification (SVC)", "Regression (SVR)"])


# ─────────────────────────────────────────────────────────────────────
#  Tab 1 — Classification
# ─────────────────────────────────────────────────────────────────────
with tab_cls:
    st.subheader("Hyperparameters")
    st.caption("Values you set **before** training. They control margin width, kernel shape, and fit.")

    c1, c2, c3, c4 = st.columns(4)
    dataset = c1.selectbox("Dataset (2 features)",
                           ["moons", "circles", "blobs", "classification"], index=0)
    kernel_cls = c2.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"], index=1)
    C_cls = c3.slider("C (inverse regularization)", 0.01, 100.0, 1.0, step=0.01)
    gamma_cls_mode = c4.selectbox("gamma", ["scale", "auto", "manual"], index=0,
                                  disabled=kernel_cls == "linear")

    c5, c6, c7, c8 = st.columns(4)
    gamma_cls_manual = c5.slider("gamma (manual)", 0.01, 10.0, 1.0, step=0.01,
                                 disabled=gamma_cls_mode != "manual" or kernel_cls == "linear")
    degree_cls = c6.slider("degree (poly)", 2, 6, 3, step=1,
                           disabled=kernel_cls != "poly")
    coef0_cls = c7.slider("coef0", -2.0, 2.0, 0.0, step=0.1,
                          disabled=kernel_cls not in ("poly", "sigmoid"))
    class_weight = c8.selectbox("class_weight", ["None", "balanced"])

    c9, c10 = st.columns(2)
    n_samples_cls = c9.slider("Samples", 80, 600, 240, step=20)
    noise_cls = c10.slider("Noise", 0.0, 0.6, 0.25, step=0.05)

    # ── Data + splits ──
    X_cls, y_cls = get_dataset(dataset, n_samples=n_samples_cls,
                               noise=noise_cls, seed=int(seed))
    X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
        X_cls, y_cls, test_size=1.0 - train_frac,
        random_state=int(seed), stratify=y_cls,
    )
    val_relative_c = val_frac / (val_frac + test_frac)
    X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
        X_temp_c, y_temp_c, test_size=1.0 - val_relative_c,
        random_state=int(seed), stratify=y_temp_c,
    )

    CW = None if class_weight == "None" else "balanced"

    def build_svc(kernel: str, C_value: float,
                  gamma_mode: str = gamma_cls_mode,
                  gamma_manual: float = gamma_cls_manual) -> SVC:
        return SVC(
            kernel=kernel,
            C=float(C_value),
            gamma=_resolve_gamma(gamma_mode, gamma_manual),
            degree=int(degree_cls),
            coef0=float(coef0_cls),
            class_weight=CW,
            random_state=int(seed),
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
            scores = model.decision_function(X_part)
            row["ROC-AUC"] = float(roc_auc_score(y_part, scores))
            row["Hinge"] = float(hinge_loss(y_part, scores, labels=[0, 1]))
        except Exception:
            row["ROC-AUC"] = float("nan")
            row["Hinge"] = float("nan")
        return row

    # ── Fit + render ──
    model_cls = build_svc(kernel_cls, C_cls).fit(X_train_c, y_train_c)

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        fig, ax = new_fig(6.5, 5.5)
        plot_decision(
            ax, model_cls, X_cls, y_cls,
            title=f"SVC — kernel={kernel_cls}, C={C_cls:g}",
            show_misclassified=False,
        )
        draw_margin_and_support(ax, model_cls, X_cls)
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Solid line = decision boundary. Dashed lines = ±1 margin "
            "(where `decision_function = ±1`). Black rings = support vectors."
        )

    with col2:
        st.markdown("**Metrics on each split**")
        rows = {
            "Train": metrics_row_cls(model_cls, X_train_c, y_train_c),
            "Val":   metrics_row_cls(model_cls, X_val_c,   y_val_c),
            "Test":  metrics_row_cls(model_cls, X_test_c,  y_test_c),
        }
        metrics_df = pd.DataFrame(rows).T[
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Hinge"]
        ]
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        st.caption(
            f"Split sizes — train: {len(X_train_c)}, val: {len(X_val_c)}, test: {len(X_test_c)}"
        )

        st.markdown("**SVM diagnostics**")
        n_sv = int(np.sum(model_cls.n_support_))
        sv_per_class = ", ".join(
            f"class {cls}: {int(n)}" for cls, n in zip(model_cls.classes_, model_cls.n_support_)
        )
        if kernel_cls == "linear":
            w = np.ravel(model_cls.coef_)
            margin_width = 2.0 / float(np.linalg.norm(w))
            margin_str = f"{margin_width:.4f}"
        else:
            margin_str = "n/a (non-linear kernel — ‖w‖ lives in feature space)"
        diag_df = pd.DataFrame(
            {
                "value": [
                    f"{n_sv}  ({n_sv / len(X_train_c):.1%} of train)",
                    sv_per_class,
                    margin_str,
                ]
            },
            index=["# support vectors", "by class", "margin width = 2/‖w‖"],
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

    with st.expander("Compare kernels on this dataset", expanded=False):
        fig_k, axes_k = grid_fig(4, ncols=4, cell=3.4)
        for ax_k, kk in zip(axes_k, ["linear", "rbf", "poly", "sigmoid"]):
            m_k = build_svc(kk, C_cls).fit(X_train_c, y_train_c)
            plot_decision(ax_k, m_k, X_cls, y_cls,
                          title=f"kernel={kk}", show_misclassified=False)
            draw_margin_and_support(ax_k, m_k, X_cls)
        st.pyplot(fig_k, use_container_width=True)
        st.caption(
            "All four use the same `C` and dataset. `linear` ignores `gamma`/`degree`; "
            "`rbf` uses `gamma`; `poly` uses `degree` and `coef0`; `sigmoid` uses `coef0`."
        )


# ─────────────────────────────────────────────────────────────────────
#  Tab 2 — Regression
# ─────────────────────────────────────────────────────────────────────
with tab_reg:
    st.subheader("Hyperparameters")
    st.caption("ε defines the 'free' zone around the prediction — only residuals outside it pay loss.")

    r1, r2, r3, r4 = st.columns(4)
    kernel_reg = r1.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"],
                              index=1, key="svr_kernel")
    C_reg = r2.slider("C (inverse regularization)", 0.01, 100.0, 1.0,
                      step=0.01, key="svr_C")
    epsilon = r3.slider("ε (insensitive zone)", 0.0, 1.0, 0.1, step=0.01)
    gamma_reg_mode = r4.selectbox("gamma", ["scale", "auto", "manual"], index=0,
                                  disabled=kernel_reg == "linear", key="svr_gamma_mode")

    r5, r6, r7, r8 = st.columns(4)
    gamma_reg_manual = r5.slider("gamma (manual)", 0.01, 10.0, 1.0, step=0.01,
                                 disabled=gamma_reg_mode != "manual" or kernel_reg == "linear",
                                 key="svr_gamma_manual")
    degree_reg = r6.slider("degree (poly)", 2, 6, 3, step=1,
                           disabled=kernel_reg != "poly", key="svr_degree")
    coef0_reg = r7.slider("coef0", -2.0, 2.0, 0.0, step=0.1,
                          disabled=kernel_reg not in ("poly", "sigmoid"),
                          key="svr_coef0")
    n_samples_reg = r8.slider("Samples", 30, 400, 150, step=10, key="svr_samples")

    r9, _ = st.columns([1, 3])
    noise_reg = r9.slider("Noise", 0.0, 1.0, 0.4, step=0.05, key="svr_noise")

    # ── Data + splits ──
    X_reg, y_reg, coef_true = get_svr_dataset(
        n_samples=n_samples_reg, noise=noise_reg, seed=int(seed)
    )
    X_train_r, X_temp_r, y_train_r, y_temp_r = train_test_split(
        X_reg, y_reg, test_size=1.0 - train_frac, random_state=int(seed)
    )
    val_relative_r = val_frac / (val_frac + test_frac)
    X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(
        X_temp_r, y_temp_r, test_size=1.0 - val_relative_r, random_state=int(seed)
    )

    def build_svr(kernel: str, C_value: float, eps_value: float,
                  gamma_mode: str = gamma_reg_mode,
                  gamma_manual: float = gamma_reg_manual) -> SVR:
        return SVR(
            kernel=kernel,
            C=float(C_value),
            epsilon=float(eps_value),
            gamma=_resolve_gamma(gamma_mode, gamma_manual),
            degree=int(degree_reg),
            coef0=float(coef0_reg),
        )

    def metrics_row_reg(model, X_part, y_part):
        pred = model.predict(X_part)
        mse = mean_squared_error(y_part, pred)
        return {
            "MSE":  float(mse),
            "RMSE": float(np.sqrt(mse)),
            "MAE":  float(mean_absolute_error(y_part, pred)),
            "R²":   float(r2_score(y_part, pred)) if len(y_part) > 1 else float("nan"),
        }

    # ── Fit + render ──
    model_reg = build_svr(kernel_reg, C_reg, epsilon).fit(X_train_r, y_train_r)

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        fig, ax = new_fig(7, 5)
        plot_regression(
            ax, model_reg, X_reg, y_reg,
            title=f"SVR — kernel={kernel_reg}, C={C_reg:g}, ε={epsilon:g}",
            line_label=f"SVR ({kernel_reg})",
        )
        draw_epsilon_tube(ax, model_reg, X_reg, y_reg, epsilon)
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Red shaded band = ε-tube around the prediction. Black-ringed points = "
            "support vectors (residual ≥ ε; the only points that shape the fit)."
        )

    with col2:
        st.markdown("**Metrics on each split**")
        rows = {
            "Train": metrics_row_reg(model_reg, X_train_r, y_train_r),
            "Val":   metrics_row_reg(model_reg, X_val_r,   y_val_r),
            "Test":  metrics_row_reg(model_reg, X_test_r,  y_test_r),
        }
        metrics_df = pd.DataFrame(rows).T[["MSE", "RMSE", "MAE", "R²"]]
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        st.caption(
            f"Split sizes — train: {len(X_train_r)}, val: {len(X_val_r)}, test: {len(X_test_r)}"
        )

        st.markdown("**SVR diagnostics**")
        n_sv = int(len(model_reg.support_))
        resid = np.abs(y_train_r - model_reg.predict(X_train_r))
        frac_inside = float(np.mean(resid <= epsilon))
        if kernel_reg == "linear":
            learned_coef = float(np.ravel(model_reg.coef_)[0])
            learned_int = float(np.ravel(model_reg.intercept_)[0])
            linear_str = f"coef={learned_coef:+.4f}  (true ≈ {coef_true:+.4f}),  intercept={learned_int:+.4f}"
        else:
            linear_str = "n/a (non-linear kernel)"
        diag_df = pd.DataFrame(
            {
                "value": [
                    f"{n_sv}  ({n_sv / len(X_train_r):.1%} of train)",
                    f"{frac_inside:.1%} of train",
                    linear_str,
                ]
            },
            index=["# support vectors", "fraction inside ε-tube", "linear params (learned vs true)"],
        )
        st.dataframe(diag_df, use_container_width=True)

    with st.expander("Compare kernels on this dataset", expanded=False):
        fig_k, axes_k = grid_fig(4, ncols=4, cell=3.4)
        for ax_k, kk in zip(axes_k, ["linear", "rbf", "poly", "sigmoid"]):
            m_k = build_svr(kk, C_reg, epsilon).fit(X_train_r, y_train_r)
            plot_regression(ax_k, m_k, X_reg, y_reg,
                            title=f"kernel={kk}", line_label=kk)
            draw_epsilon_tube(ax_k, m_k, X_reg, y_reg, epsilon)
        st.pyplot(fig_k, use_container_width=True)
        st.caption("All four use the same `C` and `ε`. Only `linear` ignores `gamma` and `degree`.")


# ───────────────────────── Explanation ─────────────────────────
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
A **support vector machine** finds the boundary (or curve) that maximizes the
*margin* — the empty corridor between itself and the nearest training points.
Only those nearest points, called **support vectors**, end up in the model;
the rest could be deleted without changing the fit.

**Classification (`SVC`).** For a linear kernel, the boundary is
`w·x + b = 0` and the margin is `2 / ‖w‖`. The ±1 dashed contours mark where
`w·x + b = ±1` — the two edges of the margin. Points inside the margin or on
the wrong side become support vectors.

- **Kernels** (`rbf`, `poly`, `sigmoid`) let the SVM draw curved boundaries
  by implicitly mapping features into a higher-dimensional space — without
  ever computing the mapping explicitly. `gamma` controls how local the
  influence of each support vector is (larger → tighter, wigglier).
- **`C`** is *inverse* regularization. Small `C` → wide margin, more
  support vectors, more bias. Large `C` → narrow margin, few support
  vectors, more variance.
- **Hinge loss** is SVMs' native loss: `max(0, 1 − y · f(x))`. It only
  penalizes points that aren't safely outside the margin, which is why a
  high-accuracy model can still show a non-zero hinge loss (margin
  violators), and why SVMs care so much about *margin*, not just *labels*.

**Regression (`SVR`).** Flip the idea inside-out: instead of pushing classes
apart, fit a curve and demand that residuals stay within an **ε-tube**.
Anything inside the tube costs nothing; only residuals outside ε pay loss
(this is **ε-insensitive loss**). Points exactly at or beyond ε become
support vectors and shape the fit.

- Shrinking **ε** forces the curve to track points more tightly and turns
  more of them into support vectors.
- Increasing **`C`** punishes excursions outside the tube harder — the curve
  bends more aggressively to keep points inside.

**Synthetic data.** Classification uses `make_moons` (crescent shapes —
non-linear), `make_circles` (concentric — only kernels can separate),
`make_blobs` (linearly separable Gaussians), and `make_classification`
(informative two-feature problem). Regression uses `make_regression`
(linear ground truth + Gaussian noise) so a `linear`-kernel SVR can
recover the slope while `rbf`/`poly` show off curved fits.
"""
    )
