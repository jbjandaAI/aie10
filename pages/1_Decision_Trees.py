"""Decision Trees — grow a tree one split at a time (classification + regression)."""
from __future__ import annotations

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from utils.data import get_dataset, get_regression_dataset
from utils.plot import plot_decision, plot_regression, new_fig, simplify_tree_labels

st.set_page_config(page_title="Decision Trees", page_icon="🌳", layout="wide")

st.title("🌳 Decision Trees")
st.caption("Recursive partitioning of the feature space, one split at a time.")

with st.sidebar:
    st.header("Controls")
    max_depth = st.slider("Max depth (final tree)", 1, 8, 5)
    play = st.toggle("▶ Auto-play", value=False)
    speed = st.slider("Frame delay (s)", 0.1, 1.5, 0.5, step=0.1, disabled=not play)

tab_cls, tab_reg = st.tabs(["Classification", "Regression"])


# ───────────────────────── Classification ─────────────────────────
with tab_cls:
    c1, c2, c3 = st.columns(3)
    dataset = c1.selectbox("Dataset", ["moons", "circles", "blobs"], index=0,
                           key="cls_dataset")
    n_samples = c2.slider("Samples", 80, 400, 200, step=20, key="cls_n")
    noise = c3.slider("Noise", 0.0, 0.6, 0.25, step=0.05, key="cls_noise")
    st.markdown("**Hyperparameters** (set *before* training — also includes `Max depth` in the sidebar)")
    criterion = c1.selectbox("Split criterion", ["gini", "entropy"], key="cls_crit")

    X, y = get_dataset(dataset, n_samples=n_samples, noise=noise)
    frame_cls = st.slider("Depth (step)", 1, max_depth,
                          max_depth if not play else 1, key="cls_frame")
    placeholder_cls = st.empty()

    def render_cls(depth: int):
        model = DecisionTreeClassifier(
            max_depth=depth, criterion=criterion, random_state=0
        ).fit(X, y)
        col1, col2 = placeholder_cls.container().columns([1, 1], gap="large")
        with col1:
            fig, ax = new_fig(6, 5)
            acc = (model.predict(X) == y).mean()
            plot_decision(ax, model, X, y, title=f"Depth {depth} — train acc {acc:.2f}")
            st.pyplot(fig, use_container_width=True)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            fig2.patch.set_facecolor("white")
            plot_tree(
                model, filled=True, rounded=True, impurity=False,
                feature_names=["x1", "x2"], class_names=["0", "1"], ax=ax2,
            )
            simplify_tree_labels(ax2)
            ax2.set_title(f"Tree structure (depth {depth})", fontsize=11)
            st.pyplot(fig2, use_container_width=True)
            st.caption("**Parameters** (*learned* from training): each node's split feature & threshold, and each leaf's predicted class.")

    if play:
        for d in range(1, max_depth + 1):
            render_cls(d)
            time.sleep(speed)
    else:
        render_cls(frame_cls)

    with st.expander("What's happening", expanded=True):
        st.markdown(
            f"""
At each step the tree picks the **single feature + threshold** that maximizes
information gain using **{criterion}** impurity, and recursively repeats the
process on each child until the depth limit is reached.

- **Depth 1** = one global cut → a stump. Underfits.
- **Greater depth** → finer axis-aligned regions. Capacity grows fast.
- Past a point the boundary starts hugging individual training points: **overfitting**.

The right panel shows the literal tree — every internal node is a `x_i ≤ threshold`
question; every leaf predicts a class.
"""
        )


# ───────────────────────── Regression ─────────────────────────
with tab_reg:
    r1, r2, r3 = st.columns(3)
    reg_dataset = r1.selectbox("Target shape", ["sine", "cubic", "step"], index=0,
                               key="reg_dataset")
    reg_n = r2.slider("Samples", 40, 300, 120, step=10, key="reg_n")
    reg_noise = r3.slider("Noise", 0.0, 1.0, 0.25, step=0.05, key="reg_noise")
    st.markdown("**Hyperparameters** (set *before* training — also includes `Max depth` in the sidebar)")
    reg_crit = r1.selectbox("Split criterion",
                            ["squared_error", "friedman_mse", "absolute_error"],
                            key="reg_crit")

    Xr, yr = get_regression_dataset(reg_dataset, n_samples=reg_n, noise=reg_noise)
    frame_reg = st.slider("Depth (step)", 1, max_depth,
                          max_depth if not play else 1, key="reg_frame")
    placeholder_reg = st.empty()

    def render_reg(depth: int):
        model = DecisionTreeRegressor(
            max_depth=depth, criterion=reg_crit, random_state=0
        ).fit(Xr, yr)
        col1, col2 = placeholder_reg.container().columns([1, 1], gap="large")
        with col1:
            fig, ax = new_fig(6, 5)
            pred = model.predict(Xr)
            ss_res = float(np.sum((yr - pred) ** 2))
            ss_tot = float(np.sum((yr - yr.mean()) ** 2))
            r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            plot_regression(ax, model, Xr, yr,
                            title=f"Depth {depth} — R² {r2_score:.2f}")
            st.pyplot(fig, use_container_width=True)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            fig2.patch.set_facecolor("white")
            plot_tree(
                model, filled=True, rounded=True, impurity=False,
                feature_names=["x"], ax=ax2, precision=2,
            )
            # Regression leaves use `value` as the actual prediction — keep it,
            # drop only `samples` to declutter.
            simplify_tree_labels(ax2, drop_prefixes=("samples",))
            ax2.set_title(f"Tree structure (depth {depth})", fontsize=11)
            st.pyplot(fig2, use_container_width=True)
            st.caption("**Parameters** (*learned* from training): each node's split feature & threshold, and each leaf's predicted value.")

    if play:
        for d in range(1, max_depth + 1):
            render_reg(d)
            time.sleep(speed)
    else:
        render_reg(frame_reg)

    with st.expander("What's happening", expanded=True):
        st.markdown(
            """
A **regression tree** splits the input axis to minimize squared error within each region,
then predicts the **mean target value** of the samples that fall into each leaf.

That's why the red prediction curve is a **step function** — every flat segment is
one leaf of the tree. As depth grows:

- **Depth 1**: a single horizontal cut → two flat plateaus.
- **More depth**: more steps, the staircase approximates the true curve.
- **Too much depth**: each leaf contains very few points → the staircase starts
  fitting the noise instead of the signal.

Unlike linear regression, decision trees produce **piecewise-constant** predictions —
they never extrapolate beyond the training range, and they cannot represent smooth slopes.
"""
        )
