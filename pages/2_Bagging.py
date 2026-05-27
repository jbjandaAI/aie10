"""Bagging — bootstrap aggregation of independent trees."""
from __future__ import annotations

import time
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from utils.data import get_dataset
from utils.plot import plot_decision, new_fig, grid_fig

st.set_page_config(page_title="Bagging", page_icon="🪣", layout="wide")

st.title("🪣 Bagging — Bootstrap Aggregating")
st.caption("Many high-variance learners on different bootstrap samples; vote to reduce variance.")

with st.sidebar:
    st.header("Controls")
    dataset = st.selectbox("Dataset", ["moons", "circles", "blobs"], index=0)
    n_samples = st.slider("Samples", 80, 400, 200, step=20)
    noise = st.slider("Noise", 0.0, 0.6, 0.3, step=0.05)
    n_estimators = st.slider("Number of trees", 1, 12, 9)
    tree_depth = st.slider("Tree depth (high-variance base)", 1, 10, 6)
    play = st.toggle("▶ Auto-play", value=False)
    speed = st.slider("Frame delay (s)", 0.1, 1.5, 0.4, step=0.1, disabled=not play)

X, y = get_dataset(dataset, n_samples=n_samples, noise=noise)
rng = np.random.RandomState(0)

# Train trees on bootstrap samples; keep references for animation.
trees = []
samples = []
for i in range(n_estimators):
    idx = rng.randint(0, len(X), size=len(X))
    samples.append(idx)
    t = DecisionTreeClassifier(max_depth=tree_depth, random_state=i).fit(X[idx], y[idx])
    trees.append(t)


class Ensemble:
    """Majority-vote wrapper over a subset of fitted trees."""
    def __init__(self, members): self.members = members
    def predict(self, X):
        votes = np.mean([t.predict(X) for t in self.members], axis=0)
        return (votes >= 0.5).astype(int)
    def predict_proba(self, X):
        probs = np.mean([t.predict_proba(X) for t in self.members], axis=0)
        return probs


frame = st.slider("Trees included so far", 1, n_estimators, n_estimators if not play else 1)
placeholder = st.empty()


def render(k: int):
    members = trees[:k]
    container = placeholder.container()
    container.markdown(f"**Ensemble size: {k} / {n_estimators}**")

    # Left: each base tree (small). Right: aggregated decision.
    col1, col2 = container.columns([1.4, 1.0], gap="large")
    with col1:
        fig, axes = grid_fig(k, ncols=3, cell=2.6)
        for i, ax in enumerate(axes):
            idx = samples[i]
            # Show only the bootstrap sample for that tree.
            plot_decision(ax, trees[i], X[idx], y[idx],
                          title=f"Tree {i+1}", alpha_bg=0.6, show_misclassified=False)
        st.pyplot(fig, use_container_width=True)

    with col2:
        ens = Ensemble(members)
        fig2, ax2 = new_fig(5.5, 5)
        acc = (ens.predict(X) == y).mean()
        plot_decision(ax2, ens, X, y, title=f"Aggregated vote — acc {acc:.2f}")
        st.pyplot(fig2, use_container_width=True)


if play:
    for k in range(1, n_estimators + 1):
        render(k)
        time.sleep(speed)
else:
    render(frame)

st.divider()
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
Each tree sees a **bootstrap sample** — drawn *with replacement* from the training data,
so roughly 63% of the original points appear and the rest are out-of-bag. Because the
samples differ, the trees disagree in their boundaries. Their errors are largely
**uncorrelated**, so averaging cancels variance:

- A single deep tree overfits — its boundary is jagged.
- Many such trees, averaged, produce a **smoother** boundary that generalizes better.
- Bias is essentially unchanged; **variance** drops with more members.

Random Forest = bagging + an extra trick: at each split only a random subset of
features is considered, which **decorrelates** the trees further.
"""
    )
