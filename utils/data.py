"""Synthetic 2D datasets used across all pages."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import (
    make_moons, make_circles, make_classification, make_blobs, make_regression,
)


def get_dataset(name: str, n_samples: int = 200, noise: float = 0.25, seed: int = 7):
    rng = np.random.RandomState(seed)
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise * 0.4, factor=0.5, random_state=seed)
    elif name == "blobs":
        # `noise` (0..~0.6) controls cluster spread → cluster_std in roughly the same range.
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=2,
            cluster_std=0.6 + noise * 1.5,
            random_state=seed,
        )
    elif name == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.2,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    # Center and scale lightly so plots look consistent
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float32), y.astype(int)


def get_regression_dataset(name: str, n_samples: int = 120, noise: float = 0.25, seed: int = 7):
    """1D regression problems used by the Decision Trees page."""
    rng = np.random.RandomState(seed)
    X = np.sort(rng.uniform(-3.0, 3.0, n_samples)).reshape(-1, 1)
    x = X.ravel()
    if name == "sine":
        y = np.sin(1.6 * x) + 0.25 * x
    elif name == "step":
        y = np.where(x < -1, -1.0, np.where(x < 1, 0.5, 1.5))
    elif name == "cubic":
        y = 0.25 * (x ** 3) - x
    else:
        raise ValueError(f"Unknown regression dataset: {name}")
    y = y + rng.normal(0.0, noise, size=n_samples)
    return X.astype(np.float32), y.astype(np.float32)


def get_linear_regression_dataset(
    n_samples: int = 120,
    noise: float = 0.5,
    seed: int = 7,
    a_true: float = 1.5,
    b_true: float = -0.4,
):
    """1-D linear regression: y = a_true * x + b_true + Gaussian noise.

    Returns (X, y, a_true, b_true) so the page can compare the learned
    slope/intercept against the ground truth.
    """
    rng = np.random.RandomState(seed)
    X = np.sort(rng.uniform(-3.0, 3.0, n_samples)).reshape(-1, 1)
    y = a_true * X.ravel() + b_true + rng.normal(0.0, noise, size=n_samples)
    return X.astype(np.float32), y.astype(np.float32), float(a_true), float(b_true)


def get_svr_dataset(n_samples: int = 150, noise: float = 0.4, seed: int = 7):
    """1-D regression problem from `make_regression`, normalized to ~[-1, 1].

    Returns (X, y, coef_true) so the SVM page can compare the linear-kernel
    SVR's learned coefficient against the ground truth, mirroring how the
    Linear Regression page surfaces (a_true, b_true).
    """
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=1,
        noise=noise * 10.0,
        coef=True,
        random_state=seed,
    )
    # Sort by x so plotting the fitted line/tube is monotone.
    order = np.argsort(X.ravel())
    X = X[order]
    y = y[order]
    # Normalize y to a tame range so default epsilon=0.1 is meaningful.
    y_scale = max(abs(y).max(), 1e-8)
    y = y / y_scale
    coef_scaled = float(coef) / y_scale
    return X.astype(np.float32), y.astype(np.float32), float(coef_scaled)


def mesh_grid(X, pad: float = 0.5, step: float = 0.03):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    return xx, yy
