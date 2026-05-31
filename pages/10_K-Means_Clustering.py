"""K-Means Clustering — unsupervised partitioning by iterative centroid updates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)

from utils.data import get_blobs_dataset, get_dataset, mesh_grid
from utils.plot import new_fig, grid_fig

st.set_page_config(page_title="K-Means Clustering", page_icon="🎯", layout="wide")

st.title("🎯 K-Means Clustering")
st.caption("Unsupervised learning: split the data into k groups by repeatedly moving each centroid to the mean of the points nearest it.")


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("Controls")
    st.caption(
        "K-Means is **unsupervised** — there are no labels and no train/val/test "
        "split. The model is fit on all of the data at once."
    )
    seed = st.number_input("Random seed", value=7, step=1)


# ───────────────────────── Hyperparameters ─────────────────────────
st.subheader("Hyperparameters")
st.caption("Choose how many clusters to look for and how the centroids are seeded and refined.")

c1, c2, c3, c4 = st.columns(4)
dataset = c1.selectbox(
    "Dataset (2 features)",
    ["blobs (4)", "blobs (3)", "blobs (5)", "moons", "circles"],
    index=0,
)
k = c2.slider("k (n_clusters)", 1, 10, 4, step=1)
init = c3.selectbox("init", ["k-means++", "random"])
n_init = c4.slider("n_init", 1, 10, 10, step=1)

c5, c6 = st.columns(2)
max_iter = c5.slider("max_iter", 1, 50, 10, step=1)
n_samples = c6.slider("Samples", 80, 600, 300, step=20)


# ───────────────────────── Data ─────────────────────────
# K-Means uses only X; y_true (if any) is kept just to note the "true" cluster count.
_BLOB_CENTERS = {"blobs (4)": 4, "blobs (3)": 3, "blobs (5)": 5}
if dataset in _BLOB_CENTERS:
    X, y_true = get_blobs_dataset(
        n_samples=n_samples, centers=_BLOB_CENTERS[dataset], seed=int(seed)
    )
    true_k = _BLOB_CENTERS[dataset]
else:
    X, y_true = get_dataset(dataset, n_samples=n_samples, noise=0.08, seed=int(seed))
    true_k = 2

# k can't exceed the number of points we cluster.
k_eff = int(min(k, len(X)))

# Cohesive categorical palette for up to 10 clusters.
PALETTE = plt.get_cmap("tab10").colors


def build_kmeans(n_clusters: int) -> KMeans:
    return KMeans(
        n_clusters=int(min(n_clusters, len(X))),
        init=init,
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(seed),
    )


def plot_clusters(ax, Xp, labels, centroids, title: str = "",
                  show_regions: bool = True, ghosts=None):
    """Scatter points colored by assigned cluster, with centroids as black X's.

    If ``show_regions`` the nearest-centroid (Voronoi) partition is shaded behind
    the points. ``ghosts`` may be a previous centroid array to draw faintly so a
    step in the walkthrough shows how far each centroid just moved.
    """
    n_c = len(centroids)
    colors = [PALETTE[i % 10] for i in range(n_c)]

    if show_regions and n_c >= 1:
        xx, yy = mesh_grid(Xp)
        grid = np.c_[xx.ravel(), yy.ravel()]
        # Assign each grid cell to its nearest centroid.
        d = np.linalg.norm(grid[:, None, :] - centroids[None, :, :], axis=2)
        region = d.argmin(axis=1).reshape(xx.shape)
        cmap = ListedColormap(colors)
        ax.contourf(xx, yy, region, levels=np.arange(-0.5, n_c + 0.5, 1.0),
                    cmap=cmap, alpha=0.15)

    for ci in range(n_c):
        m = labels == ci
        ax.scatter(
            Xp[m, 0], Xp[m, 1],
            s=34, color=colors[ci], edgecolors="white", linewidths=0.7, zorder=3,
        )

    if ghosts is not None:
        ax.scatter(ghosts[:, 0], ghosts[:, 1], s=160, marker="X",
                   color="#bbb", edgecolors="white", linewidths=1.2, zorder=4)
        for old, new in zip(ghosts, centroids):
            ax.annotate("", xy=new, xytext=old,
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
                        zorder=4)

    ax.scatter(centroids[:, 0], centroids[:, 1], s=220, marker="X",
               color="#111", edgecolors="white", linewidths=1.6, zorder=5)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("x1", fontsize=10, color="#555")
    ax.set_ylabel("x2", fontsize=10, color="#555")
    if title:
        ax.set_title(title, fontsize=11)


def lloyd_history(Xp, n_clusters, how: str, max_it: int, rng_seed: int):
    """Run Lloyd's algorithm by hand, recording (centroids, labels) per step.

    Step 0 is the initial seeding (assignment to the seeded centroids, before any
    update). Each later step is one assign→update sweep. Stops early on
    convergence (no point changes cluster).
    """
    n_clusters = int(min(n_clusters, len(Xp)))
    Xp = np.asarray(Xp, dtype=float)
    if how == "k-means++":
        centroids, _ = kmeans_plusplus(Xp, n_clusters=n_clusters, random_state=rng_seed)
    else:
        rng = np.random.RandomState(rng_seed)
        centroids = Xp[rng.choice(len(Xp), n_clusters, replace=False)].copy()

    def assign(c):
        d = np.linalg.norm(Xp[:, None, :] - c[None, :, :], axis=2)
        return d.argmin(axis=1)

    labels = assign(centroids)
    history = [(centroids.copy(), labels.copy())]

    for _ in range(int(max_it)):
        new_c = centroids.copy()
        for ci in range(n_clusters):
            m = labels == ci
            if m.any():
                new_c[ci] = Xp[m].mean(axis=0)
        new_labels = assign(new_c)
        centroids = new_c
        history.append((centroids.copy(), new_labels.copy()))
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return history


# ───────────────────────── Fit + render ─────────────────────────
model = build_kmeans(k_eff).fit(X)

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    fig, ax = new_fig(6.5, 5.5)
    plot_clusters(
        ax, X, model.labels_, model.cluster_centers_,
        title=f"K-Means — k={k_eff}, init={init}, inertia={model.inertia_:.1f}",
    )
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Each point is colored by the cluster it was assigned to; the faint shading "
        "is the **Voronoi partition** (every location belongs to its nearest "
        "centroid). Black ✕ marks each centroid — the mean of its cluster."
    )

with col2:
    st.markdown("**Clustering metrics (fit on all data)**")
    n = len(X)
    if 2 <= k_eff <= n - 1:
        sil = float(silhouette_score(X, model.labels_))
        dbi = float(davies_bouldin_score(X, model.labels_))
        chi = float(calinski_harabasz_score(X, model.labels_))
    else:
        sil = dbi = chi = float("nan")
    metrics_df = pd.DataFrame(
        {
            "value": [
                f"{model.inertia_:.2f}",
                f"{sil:.4f}" if sil == sil else "—  (needs 2 ≤ k ≤ n−1)",
                f"{dbi:.4f}" if dbi == dbi else "—",
                f"{chi:.2f}" if chi == chi else "—",
            ]
        },
        index=[
            "Inertia (↓ within-cluster SSE)",
            "Silhouette (↑ best, −1..1)",
            "Davies–Bouldin (↓ better)",
            "Calinski–Harabasz (↑ better)",
        ],
    )
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        "Inertia always falls as k grows, so it can't pick k on its own — use the "
        "elbow plus silhouette below. Silhouette/DB/CH judge cluster quality without "
        "labels."
    )

    st.markdown("**Diagnostics**")
    diag_df = pd.DataFrame(
        {
            "value": [
                f"{k_eff}" + ("  (clamped to sample size)" if k_eff != k else ""),
                init,
                str(int(n_init)),
                f"{int(model.n_iter_)} (cap {int(max_iter)})",
                f"{true_k}",
            ]
        },
        index=["clusters k", "init", "n_init restarts", "converged in", "true clusters"],
    )
    st.dataframe(diag_df, use_container_width=True)


# ───────────────────────── Iteration walkthrough ─────────────────────────
with st.expander("Watch K-Means converge (Lloyd's algorithm)", expanded=False):
    history = lloyd_history(X, k_eff, init, max_iter, int(seed))
    last = len(history) - 1
    step = st.slider("Iteration", 0, last, last, key="kmeans_step")
    centroids_s, labels_s = history[step]
    prev = history[step - 1][0] if step > 0 else None

    # Inertia at this step (within-cluster SSE for the shown assignment).
    inertia_s = float(
        sum(
            ((X[labels_s == ci] - centroids_s[ci]) ** 2).sum()
            for ci in range(len(centroids_s))
            if (labels_s == ci).any()
        )
    )

    fig_s, ax_s = new_fig(6.5, 5.5)
    label = "seeded centroids" if step == 0 else f"after update {step}"
    plot_clusters(
        ax_s, X, labels_s, centroids_s,
        title=f"Step {step}/{last} — {label}, inertia={inertia_s:.1f}",
        ghosts=prev,
    )
    st.pyplot(fig_s, use_container_width=True)
    st.caption(
        "Step 0 shows the **seeded** centroids and the first assignment. Each later "
        "step does one **assign → update** sweep: recolor every point by its nearest "
        "centroid, then move each centroid (grey ✕ → black ✕) to the mean of its new "
        "members. Inertia decreases monotonically until no point changes cluster."
    )


# ───────────────────────── Elbow method ─────────────────────────
with st.expander("Elbow method (inertia vs k)", expanded=False):
    ks = list(range(1, 11))
    inertias = [build_kmeans(kk).fit(X).inertia_ for kk in ks]

    fig_e, ax_e = new_fig(7, 4)
    ax_e.plot(ks, inertias, color="#4c78a8", marker="o", markersize=4, linewidth=1.8)
    if 2 <= true_k <= 10:
        ax_e.axvline(true_k, color="#888", linestyle="--", linewidth=1.0)
        ax_e.annotate(f"true k={true_k}", xy=(true_k, inertias[true_k - 1]),
                      xytext=(6, 12), textcoords="offset points",
                      fontsize=9, color="#666")
    ax_e.set_xlabel("k (n_clusters)", fontsize=10, color="#555")
    ax_e.set_ylabel("inertia (within-cluster SSE)", fontsize=10, color="#555")
    for s in ("top", "right"):
        ax_e.spines[s].set_visible(False)
    ax_e.spines["left"].set_color("#bbb")
    ax_e.spines["bottom"].set_color("#bbb")
    ax_e.tick_params(colors="#888", labelsize=8)
    ax_e.set_title("Inertia keeps falling — look for the bend", fontsize=11)
    st.pyplot(fig_e, use_container_width=True)
    st.caption(
        "Inertia (total squared distance to the nearest centroid) only ever "
        "decreases as k rises — at k = n it hits zero. The **elbow**, where the "
        "curve stops dropping steeply, marks the point of diminishing returns and "
        "is a good guess for the true number of clusters."
    )


# ───────────────────────── Silhouette analysis ─────────────────────────
with st.expander("Silhouette analysis (score vs k)", expanded=False):
    ks_s = list(range(2, 11))
    sils = [float(silhouette_score(X, build_kmeans(kk).fit(X).labels_)) for kk in ks_s]
    best_i = int(np.argmax(sils))
    best_k = ks_s[best_i]

    fig_sa, ax_sa = new_fig(7, 4)
    ax_sa.plot(ks_s, sils, color="#e45756", marker="o", markersize=4, linewidth=1.8)
    ax_sa.axvline(best_k, color="#888", linestyle="--", linewidth=1.0)
    ax_sa.scatter([best_k], [sils[best_i]], s=90, facecolors="none",
                  edgecolors="#111", linewidths=1.6, zorder=5)
    ax_sa.set_xlabel("k (n_clusters)", fontsize=10, color="#555")
    ax_sa.set_ylabel("mean silhouette", fontsize=10, color="#555")
    for s in ("top", "right"):
        ax_sa.spines[s].set_visible(False)
    ax_sa.spines["left"].set_color("#bbb")
    ax_sa.spines["bottom"].set_color("#bbb")
    ax_sa.tick_params(colors="#888", labelsize=8)
    ax_sa.set_title(f"Best silhouette at k={best_k}", fontsize=11)
    st.pyplot(fig_sa, use_container_width=True)
    st.caption(
        "The silhouette score (−1 to 1) measures how much tighter a point sits with "
        "its own cluster than with the next-nearest one. Unlike inertia it has a "
        "real maximum, so its peak is a principled estimate of k. For the blob "
        f"datasets it should land on the true count ({true_k})."
    )


# ───────────────────────── Compare k ─────────────────────────
with st.expander("Compare values of k", expanded=False):
    k_grid = [2, 3, 4, 5]
    fig_g, axes_g = grid_fig(len(k_grid), ncols=4, cell=3.4)
    for ax_g, kk in zip(axes_g, k_grid):
        m_k = build_kmeans(kk).fit(X)
        plot_clusters(ax_g, X, m_k.labels_, m_k.cluster_centers_,
                      title=f"k={kk}", show_regions=True)
    st.pyplot(fig_g, use_container_width=True)
    st.caption(
        "The same data carved into a different number of clusters. Too few merges "
        "distinct groups; too many splits a single blob. The elbow and silhouette "
        "panels above turn this visual judgement into a number."
    )


# ───────────────────────── Explanation ─────────────────────────
with st.expander("What's happening", expanded=True):
    st.markdown(
        """
**K-Means** is the canonical *unsupervised* algorithm: there are no labels, so it
can't be told what the groups are — it has to discover structure in the geometry
of the data alone. You pick **`k`**, the number of clusters, and it finds `k`
centers that carve the space into compact, roughly-spherical regions.

It runs **Lloyd's algorithm**, a simple two-step loop repeated until nothing moves
(watch it in the *converge* panel):

1. **Assign** — color every point by its nearest centroid.
2. **Update** — move each centroid to the mean of the points just assigned to it.

Each sweep can only lower the **inertia** (the total squared distance from points
to their centroid), so the loop is guaranteed to converge — but only to a *local*
optimum that depends on where the centroids started.

- **`k` (n_clusters)** is the one structural choice. Inertia always falls as `k`
  grows, so you can't read `k` off inertia alone — use the **elbow** (where the
  drop flattens) together with the **silhouette** peak, which *does* have a real
  maximum.
- **`init`** seeds the centroids. **`k-means++`** spreads the initial centers far
  apart, which avoids the bad local optima plain **`random`** seeding can fall
  into — it usually converges faster and to a better solution.
- **`n_init`** runs the whole thing several times from different seeds and keeps
  the lowest-inertia result, a cheap hedge against unlucky starts.
- **`max_iter`** caps the assign/update sweeps; on these tidy datasets it
  converges in a handful.

**The spherical assumption.** Because clusters are defined purely by distance to a
center, K-Means can only find blobs of similar size and spread. Switch the dataset
to **moons** or **circles** and it fails visibly — it slices the crescents and
rings straight through, because no set of centroids can describe a curved,
non-convex shape. That's the headline limitation, and why density- or
connectivity-based methods exist.

**Synthetic data.** The blob datasets come from `get_blobs_dataset` (3–5 isotropic
Gaussians) and are standardized, so both axes contribute equally to Euclidean
distance — feature scaling matters as much here as it does for k-NN.
"""
    )
