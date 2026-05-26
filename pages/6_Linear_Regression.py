"""Linear Regression — least squares, Ridge & Lasso on a 1-D synthetic dataset."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils.data import get_linear_regression_dataset
from utils.plot import plot_regression, new_fig

st.set_page_config(page_title="Linear Regression", page_icon="📈", layout="wide")

st.title("📈 Linear Regression")
st.caption("Fit a line by minimizing squared error — with optional L2 / L1 regularization.")


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

h1, h2, h3, h4, h5 = st.columns(5)
model_kind = h1.selectbox("Model", ["LinearRegression (Plain)", "Ridge (L2)", "Lasso (L1)"])
fit_intercept = h2.toggle("fit_intercept", value=True)
alpha = h3.slider("alpha (regularization strength)", 0.0, 5.0, 0.5, step=0.05,
                  disabled=model_kind == "LinearRegression (Plain)")
n_samples = h4.slider("Samples", 30, 400, 150, step=10)
noise = h5.slider("Noise σ", 0.0, 2.0, 0.6, step=0.05)

# Hidden internal defaults for the animation modes (kept off the UI for clarity).
SGD_ETA0 = 0.02
ALPHA_SWEEP_MAX = 5.0


# ───────────────────────── Data + splits ─────────────────────────
X, y, a_true, b_true = get_linear_regression_dataset(
    n_samples=n_samples, noise=noise, seed=int(seed)
)

# train_test_split twice: first carve off train, then split temp into val/test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=1.0 - train_frac, random_state=int(seed)
)
val_relative = val_frac / (val_frac + test_frac)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1.0 - val_relative, random_state=int(seed)
)


# ───────────────────────── Model factory ─────────────────────────
SGD_PENALTY = {
    "LinearRegression (Plain)": None,
    "Ridge (L2)": "l2",
    "Lasso (L1)": "l1",
}


def build_closed_form(alpha_value: float):
    if model_kind == "LinearRegression (Plain)":
        return LinearRegression(fit_intercept=fit_intercept)
    if model_kind == "Ridge (L2)":
        return Ridge(alpha=alpha_value, fit_intercept=fit_intercept,
                     random_state=int(seed))
    return Lasso(alpha=max(alpha_value, 1e-4), fit_intercept=fit_intercept,
                 random_state=int(seed), max_iter=10000)


def build_sgd():
    return SGDRegressor(
        penalty=SGD_PENALTY[model_kind],
        alpha=alpha if SGD_PENALTY[model_kind] is not None else 1e-4,
        fit_intercept=fit_intercept,
        learning_rate="constant",
        eta0=SGD_ETA0,
        random_state=int(seed),
        warm_start=True,
        max_iter=1,
        tol=None,
        shuffle=True,
    )


def metrics_row(model, X_part, y_part):
    pred = model.predict(X_part)
    mse = mean_squared_error(y_part, pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_part, pred)),
        "R²": float(r2_score(y_part, pred)) if len(y_part) > 1 else float("nan"),
    }


# ───────────────────────── Render ─────────────────────────
placeholder = st.empty()


def render(model, *, frame_caption: str = ""):
    learned_a = float(np.ravel(model.coef_)[0])
    learned_b = float(model.intercept_ if np.ndim(model.intercept_) == 0
                      else model.intercept_[0])

    col1, col2 = placeholder.container().columns([1.1, 1], gap="large")

    with col1:
        fig, ax = new_fig(7, 5)
        title = f"Fitted line   y = {learned_a:.3f}·x + {learned_b:.3f}"
        if frame_caption:
            title = f"{frame_caption} — {title}"
        plot_regression(
            ax, model, X, y,
            title=title,
            line_label="linear fit",
        )
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**Parameters** (learned during training)")
        params_df = pd.DataFrame(
            {
                "slope (coef_)":     [a_true, learned_a],
                "intercept_":        [b_true, learned_b],
            },
            index=["ground truth", "learned"],
        )
        st.dataframe(params_df.style.format("{:+.4f}"),
                     use_container_width=True)

        st.markdown("**Metrics on each split**")
        rows = {
            "Train": metrics_row(model, X_train, y_train),
            "Val":   metrics_row(model, X_val,   y_val),
            "Test":  metrics_row(model, X_test,  y_test),
        }
        metrics_df = pd.DataFrame(rows).T[["MSE", "RMSE", "MAE", "R²"]]
        st.dataframe(metrics_df.style.format("{:.4f}"),
                     use_container_width=True)
        st.caption(
            f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
        )


# ───────────────────────── Drive the animation ─────────────────────────
if anim_mode == "Static":
    model = build_closed_form(alpha).fit(X_train, y_train.ravel())
    render(model)

elif anim_mode == "Iterations (SGD)":
    sgd = build_sgd()
    epochs = list(range(1, n_frames + 1))
    if play:
        for ep in epochs:
            sgd.partial_fit(X_train, y_train.ravel())
            render(sgd, frame_caption=f"SGD epoch {ep}/{n_frames}")
            time.sleep(speed)
    else:
        chosen_ep = st.slider("Epoch", 1, n_frames, n_frames, key="lr_iter_step")
        for _ in range(chosen_ep):
            sgd.partial_fit(X_train, y_train.ravel())
        render(sgd, frame_caption=f"SGD epoch {chosen_ep}/{n_frames}")

else:  # Regularization sweep
    alphas = np.logspace(-3, np.log10(ALPHA_SWEEP_MAX), n_frames)
    if play:
        for a in alphas:
            m = build_closed_form(float(a)).fit(X_train, y_train.ravel())
            render(m, frame_caption=f"alpha = {a:.4g}")
            time.sleep(speed)
    else:
        idx = st.slider("Sweep step", 1, n_frames, n_frames, key="lr_sweep_step")
        a = float(alphas[idx - 1])
        m = build_closed_form(a).fit(X_train, y_train.ravel())
        render(m, frame_caption=f"alpha = {a:.4g}")


# ───────────────────────── Explanation ─────────────────────────
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
**Linear regression** picks the slope and intercept that minimize the sum of
squared residuals on the training set. With just one feature, this is a line
through the cloud of training points.

- **Hyperparameters** are dials you turn *before* training: which loss
  variant (plain / Ridge / Lasso), the regularization strength `alpha`,
  whether to fit an intercept, sample size, and for SGD, the learning rate
  and number of epochs.
- **Parameters** are what training *learns*: here, exactly two numbers —
  the slope (`coef_`) and the intercept (`intercept_`). With Plain
  regression and `noise → 0`, they recover the ground-truth values used to
  generate the data.
- **Ridge (L2)** shrinks the slope toward zero as `alpha` grows; the line
  flattens.
- **Lasso (L1)** can drive the slope exactly to zero — useful when you
  suspect a feature is irrelevant.
- **Train / Val / Test**: the model only ever sees the **train** split.
  Validation lets you tune hyperparameters (`alpha`, `eta0`, epochs) without
  peeking at test. Test is the final, untouched estimate of generalization.
- **Metrics**: MSE is the loss being minimized; RMSE is in the same units
  as `y`; MAE is robust to outliers; R² is the fraction of variance
  explained — 1.0 is perfect, 0.0 is "no better than predicting the mean".
"""
    )
