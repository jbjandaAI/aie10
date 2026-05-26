"""XGBoost — gradient boosting on trees, visualized round by round."""
from __future__ import annotations

import time
import numpy as np
import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt

from utils.data import get_dataset
from utils.plot import plot_decision, new_fig

st.set_page_config(page_title="XGBoost", page_icon="⚡", layout="wide")

st.title("⚡ XGBoost — Gradient Boosted Trees")
st.caption("Each tree fits the gradient of the loss with respect to the running prediction.")

with st.sidebar:
    st.header("Controls")
    dataset = st.selectbox("Dataset", ["moons", "circles", "blobs"], index=0)
    n_samples = st.slider("Samples", 80, 400, 220, step=20)
    noise = st.slider("Noise", 0.0, 0.6, 0.3, step=0.05)
    n_rounds = st.slider("Boosting rounds", 1, 80, 30)
    max_depth = st.slider("Tree depth", 1, 6, 3)
    lr = st.slider("Learning rate (eta)", 0.05, 1.0, 0.3, step=0.05)
    reg_lambda = st.slider("L2 reg (lambda)", 0.0, 5.0, 1.0, step=0.1)
    play = st.toggle("▶ Auto-play", value=False)
    speed = st.slider("Frame delay (s)", 0.05, 1.0, 0.2, step=0.05, disabled=not play)


@st.cache_data(show_spinner=False)
def fit_xgb(dataset, n_samples, noise, n_rounds, max_depth, lr, reg_lambda):
    X, y = get_dataset(dataset, n_samples=n_samples, noise=noise)
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "max_depth": int(max_depth),
        "eta": float(lr),
        "lambda": float(reg_lambda),
        "verbosity": 0,
        "tree_method": "hist",
    }
    booster = xgb.train(params, dtrain, num_boost_round=n_rounds)

    # Per-round train log-loss using iteration_range.
    losses = []
    for k in range(1, n_rounds + 1):
        p = booster.predict(dtrain, iteration_range=(0, k))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        losses.append(float(ll))
    return X, y, booster, losses


X, y, booster, losses = fit_xgb(dataset, n_samples, noise, n_rounds, max_depth, lr, reg_lambda)


class XGBSlice:
    """Wraps a booster but only uses the first k trees."""
    def __init__(self, booster, k): self.booster, self.k = booster, int(k)
    def predict_proba(self, X):
        d = xgb.DMatrix(X)
        p = self.booster.predict(d, iteration_range=(0, self.k))
        return np.column_stack([1 - p, p])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


frame = st.slider("Trees used", 1, n_rounds, n_rounds if not play else 1)
placeholder = st.empty()


def render(k: int):
    container = placeholder.container()
    col1, col2 = container.columns([1.1, 1.0], gap="large")

    with col1:
        model = XGBSlice(booster, k)
        fig, ax = new_fig(6, 5)
        acc = (model.predict(X) == y).mean()
        plot_decision(ax, model, X, y, title=f"Trees: {k} — acc {acc:.2f} — loss {losses[k-1]:.3f}")
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor("white")
        xs = np.arange(1, n_rounds + 1)
        ax2.plot(xs, losses, color="#4c78a8", linewidth=1.6, alpha=0.5)
        ax2.plot(xs[:k], losses[:k], color="#4c78a8", linewidth=2.4)
        ax2.scatter([k], [losses[k - 1]], color="#e45756", zorder=5, s=60)
        ax2.set_xlabel("Boosting round")
        ax2.set_ylabel("Train log-loss")
        ax2.set_title("Loss curve")
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        st.pyplot(fig2, use_container_width=True)


if play:
    for k in range(1, n_rounds + 1):
        render(k)
        time.sleep(speed)
else:
    render(frame)

st.divider()
with st.expander("What's happening", expanded=True):
    st.markdown(
        f"""
**XGBoost** builds an additive model: at round *k*, the prediction is

`F_k(x) = F_{{k-1}}(x) + η · h_k(x)`

where `h_k` is a regression tree fit to the **negative gradient** (and curvature, via the Hessian)
of the log-loss at the previous round's predictions.

Key knobs you can play with:

- **Learning rate η** (`{lr}`): shrinks each tree's contribution. Smaller η → smoother fit, needs more rounds.
- **Tree depth** (`{max_depth}`): per-tree capacity. Deep trees model interactions but overfit faster.
- **L2 regularization λ** (`{reg_lambda}`): penalizes leaf weights, keeps individual trees from being too confident.

Watch the **loss curve** drop quickly at first and then flatten — diminishing returns are normal.
If the boundary starts to look noisy past a point, that's where early stopping would kick in on real data.
"""
    )
