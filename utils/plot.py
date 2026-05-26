"""Shared plotting helpers built on matplotlib."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .data import mesh_grid

# Cohesive palette — warm vs cool, with a neutral background gradient.
CMAP_BG = LinearSegmentedColormap.from_list(
    "bg", ["#4c78a8", "#f5f5f5", "#e45756"]
)
CLASS_COLORS = ["#1f77b4", "#d62728"]


def _predict_proba_or_decision(model, grid):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(grid)
        return p[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(grid)
        return 1.0 / (1.0 + np.exp(-d))
    pred = model.predict(grid)
    return pred.astype(float)


def plot_decision(
    ax,
    model,
    X,
    y,
    weights=None,
    title: str = "",
    show_misclassified: bool = True,
    alpha_bg: float = 0.7,
    xlabel: str = "x1",
    ylabel: str = "x2",
):
    xx, yy = mesh_grid(X)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = _predict_proba_or_decision(model, grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap=CMAP_BG, alpha=alpha_bg, vmin=0, vmax=1)
    ax.contour(xx, yy, zz, levels=[0.5], colors="#222", linewidths=1.0, alpha=0.6)

    # Point sizes — boosted samples get bigger dots.
    if weights is None:
        sizes = np.full(len(X), 36.0)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.max() if w.max() > 0 else w
        sizes = 18.0 + 130.0 * w

    for cls in (0, 1):
        m = y == cls
        ax.scatter(
            X[m, 0], X[m, 1],
            s=sizes[m], c=CLASS_COLORS[cls],
            edgecolors="white", linewidths=0.8, zorder=3,
        )

    if show_misclassified:
        try:
            pred = model.predict(X)
            wrong = pred != y
            if wrong.any():
                ax.scatter(
                    X[wrong, 0], X[wrong, 1],
                    s=sizes[wrong] + 60,
                    facecolors="none", edgecolors="#b30000",
                    linewidths=1.6, zorder=4,
                )
        except Exception:
            pass

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color="#555")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#555")
    if title:
        ax.set_title(title, fontsize=11)


def plot_regression(
    ax,
    model,
    X,
    y,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
):
    """Plot 1D regression data with the model's piecewise prediction overlay."""
    x_flat = np.asarray(X).ravel()
    pad = 0.3 * (x_flat.max() - x_flat.min())
    xs = np.linspace(x_flat.min() - pad, x_flat.max() + pad, 600).reshape(-1, 1)
    ys = model.predict(xs)

    ax.scatter(
        x_flat, y,
        s=34, c="#4c78a8", edgecolors="white", linewidths=0.7,
        zorder=3, label="training data",
    )
    ax.plot(xs.ravel(), ys, color="#e45756", linewidth=2.2,
            label="tree prediction", zorder=4)

    ax.set_xlabel(xlabel, fontsize=10, color="#555")
    ax.set_ylabel(ylabel, fontsize=10, color="#555")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbb")
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(colors="#888", labelsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)


def simplify_tree_labels(ax, drop_prefixes=("samples", "value")):
    """Strip lines like 'samples = ...' / 'value = ...' from plot_tree node text."""
    for child in ax.get_children():
        if not (hasattr(child, "get_text") and hasattr(child, "set_text")):
            continue
        raw = child.get_text()
        if not raw or "\n" not in raw:
            continue
        kept = [
            ln for ln in raw.split("\n")
            if not any(ln.strip().startswith(p) for p in drop_prefixes)
        ]
        child.set_text("\n".join(kept))


def new_fig(width: float = 6.0, height: float = 5.0):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    return fig, ax


def grid_fig(n: int, ncols: int = 3, cell: float = 3.2):
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows))
    fig.patch.set_facecolor("white")
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes[:n]
