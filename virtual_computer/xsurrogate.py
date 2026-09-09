"""Cross-layer surrogate v1: ridge predictor of vertical outcomes.

The L1 blocker (docs/CRITIQUE.md) is the absence of any learned model
that predicts cross-layer measurement outcomes. This module is the
first such model, deliberately minimal: L2-regularized linear
regression (numpy only, like the layers' own prototypes) from
observable-side vertical features to measured deltas, evaluated by
leave-one-out skill vs the constant-mean baseline:

    skill = 1 - MAE(LOO preds) / MAE(constant mean)

Features use ONLY observable configuration (sizes, op counts, demand
anchors, control/sysctl settings) — never hidden workload demand or
oracle internals. A positive skill means the vertical outcomes have
learnable structure at this scale; zero/negative means they don't, and
is reported as such (a null result is evidence, not failure).
"""
from __future__ import annotations

import numpy as np


def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    # standardize columns (constant columns -> zeros, no NaN)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0.0] = 1.0
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + l2 * np.eye(d)
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    return {"w": w, "mu": mu, "sd": sd, "ymean": float(y.mean())}


def ridge_predict(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return model["ymean"] + ((X - model["mu"]) / model["sd"]) @ model["w"]


def loo_skill(X, y, l2: float = 1.0) -> dict:
    """Leave-one-out predictions, skill + R² vs constant baseline.

    Degrades gracefully (skill 0.0 + reason) when the sample cannot
    support LOO — measurement code reports, never crashes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    base = {"n": n, "l2": l2, "mae_model": None, "mae_const": None,
            "skill": 0.0, "r2": 0.0, "preds": []}
    if n < 3:
        return {**base, "reason": "insufficient-units"}
    if float(np.std(y)) == 0.0:
        return {**base, "reason": "constant-target"}
    preds = np.array([ridge_predict(ridge_fit(np.delete(X, i, 0),
                                              np.delete(y, i), l2), X[i:i + 1])[0]
                      for i in range(n)])
    mae_model = float(np.mean(np.abs(preds - y)))
    mae_const = float(np.mean(np.abs(y - y.mean())))
    r2_num = float(np.sum((y - preds) ** 2))
    r2_den = float(np.sum((y - y.mean()) ** 2))
    return {"n": n, "l2": l2,
            "mae_model": mae_model, "mae_const": mae_const,
            "skill": 1.0 - mae_model / mae_const if mae_const > 0 else 0.0,
            "r2": 1.0 - r2_num / r2_den if r2_den > 0 else 0.0,
            "preds": preds.tolist()}


def micro_features(n: int, mem_o0: int, mem_o3: int, control: str) -> list[float]:
    return [float(n), float(mem_o0), float(mem_o3), float(mem_o0 - mem_o3),
            1.0 if control == "streaming" else 0.0]


def kernel_features(demand: float, workload: str, n: int, gain: float,
                    steps: int) -> list[float]:
    return [float(demand), 1.0 if workload == "O0" else 0.0,
            float(n), float(gain), float(steps)]
