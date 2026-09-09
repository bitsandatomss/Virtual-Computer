"""Minimal numpy MLP stack with Adam (no torch dependency).

Supports the latent E/F/D surrogate: Linear -> tanh -> Linear blocks with
explicit parameter vector handling for checkpointing (lists, JSON-safe).
"""
from __future__ import annotations

import numpy as np


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1.0 - y * y


class MLP:
    def __init__(self, dims: list[int], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.dims = list(dims)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for n_in, n_out in zip(dims[:-1], dims[1:]):
            s = np.sqrt(2.0 / (n_in + n_out))
            self.W.append(rng.normal(0, s, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list]:
        h = x
        caches = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < len(self.W) - 1:
                h = tanh(z)
                caches.append((h, z))
            else:
                caches.append((z, z))
        return z, caches

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _ = self.forward(np.asarray(x, dtype=float))
        return out

    def parameters(self) -> dict:
        return {"dims": self.dims, "W": [w.tolist() for w in self.W],
                "b": [bb.tolist() for bb in self.b]}

    @classmethod
    def from_parameters(cls, p: dict) -> "MLP":
        m = cls.__new__(cls)
        m.dims = list(p["dims"])
        m.W = [np.asarray(w, dtype=float) for w in p["W"]]
        m.b = [np.asarray(bb, dtype=float) for bb in p["b"]]
        return m


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def mlp_gradients(mlp: MLP, x: np.ndarray, dout: np.ndarray) -> tuple[list, list]:
    """Backprop for MSE-style upstream grad dout (batch, out). Returns grads."""
    grads_W, grads_b, _ = mlp_backward(mlp, x, dout)
    return grads_W, grads_b


def mlp_backward(mlp: MLP, x: np.ndarray,
                 dout: np.ndarray) -> tuple[list, list, np.ndarray]:
    """Full backprop. Returns (grads_W, grads_b, dx) with exact input gradient."""
    x = np.asarray(x, dtype=float)
    _, caches = mlp.forward(x)
    grads_W: list[np.ndarray] = [None] * len(mlp.W)  # type: ignore
    grads_b: list[np.ndarray] = [None] * len(mlp.b)  # type: ignore
    delta = np.asarray(dout, dtype=float)
    h_prev = [x] + [c[0] for c in caches[:-1]]
    for i in reversed(range(len(mlp.W))):
        grads_W[i] = h_prev[i].T @ delta / max(len(x), 1)
        grads_b[i] = delta.mean(axis=0)
        dh = delta @ mlp.W[i].T
        if i > 0:
            delta = dh * dtanh(caches[i - 1][0])
        else:
            dx = dh
    return grads_W, grads_b, dx
