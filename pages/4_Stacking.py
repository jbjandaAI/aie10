"""Stacking — base learners feed predictions into a meta-learner."""
from __future__ import annotations

import time
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

from utils.data import get_dataset
from utils.plot import plot_decision, new_fig, grid_fig

st.set_page_config(page_title="Stacking", page_icon="🧱", layout="wide")

st.title("🧱 Stacking — Stacked Generalization")
st.caption("Diverse base learners produce predictions; a meta-learner learns how to combine them.")

BASE_LIBRARY = {
    "Decision Tree (d=4)": lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
    "k-NN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": lambda: SVC(kernel="rbf", probability=True, random_state=0),
    "Logistic Regression": lambda: LogisticRegression(max_iter=500),
    "Naive Bayes": lambda: GaussianNB(),
}

with st.sidebar:
    st.header("Controls")
    dataset = st.selectbox("Dataset", ["moons", "circles", "blobs"], index=0)
    n_samples = st.slider("Samples", 80, 400, 200, step=20)
    noise = st.slider("Noise", 0.0, 0.6, 0.3, step=0.05)
    selected = st.multiselect(
        "Base learners",
        list(BASE_LIBRARY.keys()),
        default=["Decision Tree (d=4)", "k-NN (k=5)", "SVM (RBF)"],
    )
    play = st.toggle("▶ Auto-play", value=False)
    speed = st.slider("Frame delay (s)", 0.1, 1.5, 0.6, step=0.1, disabled=not play)

if not selected:
    st.warning("Pick at least one base learner.")
    st.stop()

X, y = get_dataset(dataset, n_samples=n_samples, noise=noise)

# Out-of-fold predictions to train the meta-learner honestly.
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof = np.zeros((len(X), len(selected)))
for tr, va in kf.split(X):
    for j, name in enumerate(selected):
        m = BASE_LIBRARY[name]()
        m.fit(X[tr], y[tr])
        if hasattr(m, "predict_proba"):
            oof[va, j] = m.predict_proba(X[va])[:, 1]
        else:
            oof[va, j] = m.decision_function(X[va])

# Fit each base learner on full data (for visualization)
base_models = [BASE_LIBRARY[n]() for n in selected]
for m in base_models:
    m.fit(X, y)

meta = LogisticRegression(max_iter=1000).fit(oof, y)


class Stacked:
    def __init__(self, bases, meta): self.bases, self.meta = bases, meta
    def _base_features(self, X):
        feats = []
        for m in self.bases:
            if hasattr(m, "predict_proba"):
                feats.append(m.predict_proba(X)[:, 1])
            else:
                feats.append(m.decision_function(X))
        return np.column_stack(feats)
    def predict(self, X): return self.meta.predict(self._base_features(X))
    def predict_proba(self, X): return self.meta.predict_proba(self._base_features(X))


stacked = Stacked(base_models, meta)

n_steps = len(selected) + 1  # show base learners one by one, then the stack
frame = st.slider("Step", 1, n_steps, n_steps if not play else 1)
placeholder = st.empty()


def render(step: int):
    container = placeholder.container()

    show_bases = min(step, len(selected))
    fig, axes = grid_fig(show_bases, ncols=min(3, len(selected)), cell=3.0)
    for j, ax in enumerate(axes):
        m = base_models[j]
        acc = (m.predict(X) == y).mean()
        plot_decision(ax, m, X, y, title=f"{selected[j]} — acc {acc:.2f}", alpha_bg=0.6)
    container.markdown("##### Base learners")
    container.pyplot(fig, use_container_width=True)

    if step > len(selected):
        container.markdown("##### Meta-learner combining base predictions")
        fig2, ax2 = new_fig(6, 5)
        acc = (stacked.predict(X) == y).mean()
        plot_decision(ax2, stacked, X, y,
                      title=f"Stacked ensemble — acc {acc:.2f}")
        container.pyplot(fig2, use_container_width=True)

        # Show meta-learner coefficients = how much each base contributes.
        coefs = meta.coef_.ravel()
        container.markdown("**Meta-learner coefficients** (logistic regression weights on base predictions):")
        cols = container.columns(len(selected))
        for j, c in enumerate(cols):
            c.metric(selected[j], f"{coefs[j]:+.2f}")


if play:
    for s in range(1, n_steps + 1):
        render(s)
        time.sleep(speed)
else:
    render(frame)

st.divider()
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
**Stacking** is a two-layer ensemble:

1. **Base learners** of different families produce predictions. Diversity matters more than individual strength.
2. A **meta-learner** (here a logistic regression) is trained on those predictions — it learns *when each base learner can be trusted*.

To avoid the meta-learner memorizing the bases' training errors, we generate **out-of-fold predictions**
with K-fold CV: each row's base feature is produced by a model that *did not see that row* during training.

The meta-coefficients shown above are exactly that "trust weighting" — large positive coefficients
mean the meta-learner relies heavily on that base; small or negative ones mean it largely ignores it.
"""
    )
