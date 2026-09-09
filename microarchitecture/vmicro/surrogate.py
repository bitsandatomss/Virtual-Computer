"""Learned timing surrogate: predicts interval cost without running the oracle.

The exact CPU (machine.py) is expensive to query at scale; the surrogate is a
cheap approximation T_hat(s, a) ~= cost of next K cycles under bundle a.
Trained with ridge regression on oracle traces. Ensemble of bootstraps gives
uncertainty (disagreement) for active reconstruction / search guidance.

Only numpy is required.
"""
from __future__ import annotations

import numpy as np

from vmicro.machine import ACTION_LIBRARY, CPU, FEATURE_NAMES


class SurrogateModel:
    def __init__(self, weights: np.ndarray, intercept: float, bundle_names: list[str]):
        self.w = np.asarray(weights, dtype=float)
        self.b = float(intercept)
        self.bundles = list(bundle_names)

    def predict(self, features: list[float] | np.ndarray, bundle: str) -> float:
        x = np.asarray(features, dtype=float)
        j = self.bundles.index(bundle.split(":")[0])
        # weight block per bundle: w has shape (n_bundles * n_features,)
        f = len(FEATURE_NAMES)
        seg = self.w[j * f:(j + 1) * f]
        return float(x @ seg + self.b)

    def uncertainty(self, features: list[float] | np.ndarray) -> float:
        # single model: heuristic spread from feature magnitudes
        x = np.asarray(features, dtype=float)
        preds = [self.predict(x, b) for b in self.bundles]
        return float(np.std(preds) / (abs(np.mean(preds)) + 1e-6))


class SurrogateEnsemble:
    def __init__(self, members: list[SurrogateModel]):
        assert members, "need at least one member"
        self.members = members
        self.bundles = members[0].bundles

    def predict(self, features, bundle: str) -> float:
        return float(np.mean([m.predict(features, bundle) for m in self.members]))

    def uncertainty(self, features) -> float:
        preds = np.array([m.predict(features, b) for m in self.members for b in self.bundles])
        return float(np.std(preds) / (abs(np.mean(preds)) + 1e-6))

    def predict_all(self, features) -> dict[str, float]:
        return {b: self.predict(features, b) for b in self.bundles}


def collect_trace(program, mem_init=None, reg_init=None, bundles: list[str] | None = None, interval: int = 8) -> list[tuple[list[float], str, float]]:
    """Run oracle once; emit (features, bundle_used, interval_cost) examples."""
    bundles = bundles or ["balanced", "streaming", "irregular", "control", "latency"]
    cpu = CPU(program, mem_init=mem_init, reg_init=reg_init)
    examples = []
    bi = 0
    while not cpu.halted and cpu.stats.cycles < cpu.max_cycles:
        bundle = bundles[bi % len(bundles)]
        cpu.set_control(ACTION_LIBRARY[bundle], controller_energy=0.01)
        feats = cpu.feature_vector()
        c0, e0, x0 = cpu.stats.cycles, cpu.stats.energy, cpu.stats.thermal_exposure
        for _ in range(interval):
            if cpu.halted or cpu.fetch_pc >= len(cpu.program):
                if not cpu.in_flight:
                    break
            try:
                cpu.step_cycle()
            except RuntimeError:
                break
        cost = (cpu.stats.cycles - c0) + 0.08 * ((cpu.stats.energy - e0) - 0.0) + 0.6 * (cpu.stats.thermal_exposure - x0)
        examples.append((feats, bundle, cost))
        bi += 1
        if cpu.fetch_pc >= len(cpu.program) and not cpu.in_flight:
            cpu.halted = True
            break
    return examples


def train_surrogate(examples: list[tuple[list[float], str, float]], l2: float = 1e-3, seed: int = 0, n_members: int = 3) -> SurrogateEnsemble:
    bundles = sorted({b for _, b, _ in examples})
    f = len(FEATURE_NAMES)
    rng = np.random.default_rng(seed)
    members = []
    n = len(examples)
    for m in range(n_members):
        idx = rng.integers(0, n, n)
        rows, targets = [], []
        for i in idx:
            feats, b, cost = examples[int(i)]
            j = bundles.index(b)
            row = np.zeros(len(bundles) * f)
            row[j * f:(j + 1) * f] = np.asarray(feats)
            rows.append(row)
            targets.append(cost)
        X = np.stack(rows)
        y = np.asarray(targets)
        # ridge: (X'X + l2 I)^-1 X'y
        A = X.T @ X + l2 * np.eye(X.shape[1])
        w = np.linalg.solve(A, X.T @ y)
        b0 = float(y.mean() - (X.mean(axis=0) @ w) * 0.0)  # keep intercept 0-centered; fold bias via feature
        members.append(SurrogateModel(w, 0.0, bundles))
    return SurrogateEnsemble(members)
