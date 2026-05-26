"""AdaBoost — sequential weak learners on reweighted samples."""
from __future__ import annotations

import time
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

from utils.data import get_dataset
from utils.plot import plot_decision, new_fig

st.set_page_config(page_title="Boosting (AdaBoost)", page_icon="🚀", layout="wide")

st.title("🚀 Boosting — AdaBoost")
st.caption("Sequential weak learners. Each focuses on the previous round's mistakes.")

with st.sidebar:
    st.header("Controls")
    dataset = st.selectbox("Dataset", ["moons", "circles", "blobs"], index=0)
    n_samples = st.slider("Samples", 80, 400, 180, step=20)
    noise = st.slider("Noise", 0.0, 0.6, 0.3, step=0.05)
    n_rounds = st.slider("Boosting rounds", 1, 30, 15)
    stump_depth = st.slider("Weak learner depth", 1, 4, 1)
    play = st.toggle("▶ Auto-play", value=False)
    speed = st.slider("Frame delay (s)", 0.1, 1.5, 0.45, step=0.1, disabled=not play)


@st.cache_data(show_spinner=False)
def fit_adaboost(dataset, n_samples, noise, n_rounds, stump_depth):
    """Hand-rolled SAMME AdaBoost so we can return per-round state."""
    X, y = get_dataset(dataset, n_samples=n_samples, noise=noise)
    y_signed = np.where(y == 1, 1, -1)
    N = len(X)
    w = np.ones(N) / N

    snapshots = []  # list of (weights_before, learner, alpha, weights_after)
    for r in range(n_rounds):
        clf = DecisionTreeClassifier(max_depth=stump_depth, random_state=r)
        clf.fit(X, y, sample_weight=w)
        pred = np.where(clf.predict(X) == 1, 1, -1)
        miss = pred != y_signed
        err = np.sum(w[miss]) / np.sum(w)
        err = float(np.clip(err, 1e-10, 1 - 1e-10))
        alpha = 0.5 * np.log((1 - err) / err)
        w_before = w.copy()
        w = w * np.exp(-alpha * y_signed * pred)
        w = w / w.sum()
        snapshots.append((w_before, clf, alpha, w.copy(), err))
    return X, y, snapshots


X, y, snapshots = fit_adaboost(dataset, n_samples, noise, n_rounds, stump_depth)


class WeightedEnsemble:
    def __init__(self, members, alphas):
        self.members = members
        self.alphas = np.asarray(alphas, dtype=float)
    def decision_function(self, X):
        s = np.zeros(len(X))
        for clf, a in zip(self.members, self.alphas):
            pred = np.where(clf.predict(X) == 1, 1, -1)
            s += a * pred
        return s
    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


frame = st.slider("Round", 1, n_rounds, n_rounds if not play else 1)
placeholder = st.empty()


def render(k: int):
    container = placeholder.container()
    w_before, current, alpha_k, w_after, err_k = snapshots[k - 1]

    members = [s[1] for s in snapshots[:k]]
    alphas = [s[2] for s in snapshots[:k]]
    ens = WeightedEnsemble(members, alphas)

    col1, col2 = container.columns([1, 1], gap="large")
    with col1:
        fig, ax = new_fig(6, 5)
        plot_decision(ax, current, X, y, weights=w_before,
                      title=f"Round {k}: weak learner (err={err_k:.2f}, α={alpha_k:.2f})")
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig2, ax2 = new_fig(6, 5)
        acc = (ens.predict(X) == y).mean()
        plot_decision(ax2, ens, X, y, weights=w_after,
                      title=f"Ensemble after round {k} — acc {acc:.2f}")
        st.pyplot(fig2, use_container_width=True)

    # Compact metric strip
    cols = container.columns(3)
    cols[0].metric("Weighted error this round", f"{err_k:.3f}")
    cols[1].metric("Learner weight α", f"{alpha_k:.3f}")
    cols[2].metric("Train accuracy", f"{(ens.predict(X) == y).mean():.3f}")


if play:
    for k in range(1, n_rounds + 1):
        render(k)
        time.sleep(speed)
else:
    render(frame)

st.divider()
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
**AdaBoost** trains weak learners one at a time. After each round:

1. Compute the **weighted error** `ε` of the new learner on the current sample weights.
2. Give it a vote weight `α = ½ · ln((1−ε)/ε)`. Better learners get bigger α.
3. **Up-weight** points it got wrong, **down-weight** points it got right, then renormalize.
4. The next learner is forced to attend to the previously hard examples.

In the left plot, **bigger dots = higher current weight** — watch them concentrate on
the boundary. The right plot is the weighted vote of all learners so far.

Boosting reduces **bias** (unlike bagging, which reduces variance) by chaining
underfit models into a strong one.
"""
    )
