"""dynamics.py — learned latent state-transition model of kernel behaviour.

Architecture (context.txt lines 126-139, 311-315):
    z_t     = E(x_t)            encoder: KIR -> latent
    z_{t+1} = F(z_t, a_t)       transition: (latent, action) -> latent
    x^_{t+1} = D(z_{t+1})       decoder: latent -> observable prediction

NOT  f(x, g) = y  (single perturbation prediction).
BUT  F(z, a) = z' (persistent environment with sequential interventions).

Implementation: small numpy MLPs (no torch dependency, repo is stdlib+pydantic).
Observables predicted (OBS_DIM=4):
    [mean_util, runnable_norm, lat_norm, switch_rate_norm]
Action encoding (ACTION_DIM=1):
    log-scaled target_latency_us in [0, 1].
Latent (LATENT_DIM=8): linear projection of the 6-D KIR feature vector.

Training data: (f_t, a_t, y_{t+1}) tuples collected from KernelSimulator,
which plays the role of the "exact oracle" (cf. Virtual Chess exact rules).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from learned_kernel.policy.schemas import (
    CPUMetrics,
    KernelIntermediateRepresentation,
    PolicyAction,
    SchedulerState,
)
from learned_kernel.trainer.rl_trainer import extract_features

LATENT_DIM = 8
OBS_DIM = 4
ACTION_DIM = 1

_MIN_LAT_US = 1_000.0
_MAX_LAT_US = 24_000.0
_MAX_RUNNABLE = 64.0
_NORM_LAT_MS = 20.0
_MAX_SWITCH_RATE = 30_000.0
_T_REF_US = 6_000.0

_HIDDEN1 = 32
_HIDDEN2 = 16


def encode_action(action: Optional[PolicyAction]) -> np.ndarray:
    """Log-scale target_latency_us -> [0, 1]. None/default -> T_REF position."""
    t = _T_REF_US
    if action is not None and action.scheduler is not None \
            and action.scheduler.target_latency_us is not None:
        t = float(action.scheduler.target_latency_us)
    t = min(_MAX_LAT_US, max(_MIN_LAT_US, t))
    a = (math.log(t) - math.log(_MIN_LAT_US)) / (math.log(_MAX_LAT_US) - math.log(_MIN_LAT_US))
    return np.array([a], dtype=np.float64)


def encode_kir(kir: KernelIntermediateRepresentation) -> np.ndarray:
    """KIR -> 6-D normalized feature vector (same as OfflineTrainer)."""
    return np.array(extract_features(kir), dtype=np.float64)


def kir_observables(kir: KernelIntermediateRepresentation, prev_switches: int,
                    dt_s: float = 0.5) -> np.ndarray:
    """KIR (+ switch delta) -> 4-D supervised target in [0, 1]-ish range."""
    cpus = list(kir.scheduler.cpus.values())
    mean_util = sum(c.utilization for c in cpus) / len(cpus) if cpus else 0.0
    runnable_norm = min(sum(c.runnable_tasks for c in cpus), _MAX_RUNNABLE) / _MAX_RUNNABLE
    lat_norm = min(kir.scheduler.avg_latency_ms, _NORM_LAT_MS) / _NORM_LAT_MS
    rate = max(0.0, (kir.scheduler.total_context_switches - prev_switches) / max(dt_s, 1e-6))
    rate_norm = min(rate, _MAX_SWITCH_RATE) / _MAX_SWITCH_RATE
    return np.array([mean_util, runnable_norm, lat_norm, rate_norm], dtype=np.float64)


def decode_latent(z: np.ndarray, prev_switches: int, n_cpus: int = 2,
                  timestamp: float = 1000.0, dt_s: float = 0.5) -> KernelIntermediateRepresentation:
    """Fallback decoder used only when a model has no learned D matrix yet."""
    o = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64).ravel()[:4]))
    mean_util = float(np.clip(o[0], 0.02, 1.0))
    runnable = int(round(float(np.clip(o[1], 0.0, 1.0)) * _MAX_RUNNABLE / max(n_cpus, 1)))
    lat_ms = float(np.clip(o[2], 0.0025, 1.0) * _NORM_LAT_MS)
    rate = float(np.clip(o[3], 0.0, 1.0) * _MAX_SWITCH_RATE)
    cpus = {i: CPUMetrics(utilization=mean_util, runnable_tasks=runnable, irq_time=0.02)
            for i in range(n_cpus)}
    return KernelIntermediateRepresentation(
        timestamp=timestamp,
        scheduler=SchedulerState(
            cpus=cpus,
            total_context_switches=int(prev_switches + rate * dt_s),
            avg_latency_ms=round(max(0.05, lat_ms), 4),
        ),
    )


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class TransitionModel:
    """E + F + D with residual transition. Pure numpy, seeded, serializable."""

    def __init__(self, seed: int = 0, latent_dim: int = LATENT_DIM):
        self.latent_dim = latent_dim
        rng = np.random.default_rng(seed)
        s = 0.5
        self.E = rng.normal(0, s / math.sqrt(6), size=(latent_dim, 6))
        self.eb = np.zeros(latent_dim)
        # F: [z(8) + a(1)] -> h1(32) -> h2(16) -> dz(8)
        self.W1 = rng.normal(0, s / math.sqrt(9), size=(_HIDDEN1, latent_dim + 1))
        self.b1 = np.zeros(_HIDDEN1)
        self.W2 = rng.normal(0, s / math.sqrt(_HIDDEN1), size=(_HIDDEN2, _HIDDEN1))
        self.b2 = np.zeros(_HIDDEN2)
        self.W3 = rng.normal(0, s / math.sqrt(_HIDDEN2), size=(latent_dim, _HIDDEN2))
        self.b3 = np.zeros(latent_dim)
        # D: z(8) -> obs(4), plus a QUADRATIC action skip V (OBS_DIM x 2).
        # The true scheduler latency is U-shaped in log target-latency
        # (queueing vs preemption overhead -> interior optimum; see
        # learned_kernel/simulator/env.py). A linear skip can only express a
        # monotone trend and actively fights the left arm, collapsing the
        # surrogate toward "tightest is always best". The [a, a^2] path gives
        # the decoder the right structure class while staying fully learned.
        self.D = rng.normal(0, s / math.sqrt(latent_dim), size=(OBS_DIM, latent_dim))
        self.V = np.zeros((OBS_DIM, 2))
        self.db = np.array([0.3, 0.1, 0.2, 0.05])
        self.seed = seed

    @staticmethod
    def _action_basis(a: np.ndarray) -> np.ndarray:
        a1 = np.asarray(a, dtype=np.float64).ravel()
        return np.stack([a1, a1 ** 2], axis=1)  # (n, 2)

    # -- forward -- #
    def encode(self, f: np.ndarray) -> np.ndarray:
        return _tanh(self.E @ np.asarray(f, dtype=np.float64) + self.eb)

    def transition(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        x = np.concatenate([np.asarray(z).ravel(), np.asarray(a).ravel()])
        h1 = _tanh(self.W1 @ x + self.b1)
        h2 = _tanh(self.W2 @ h1 + self.b2)
        dz = self.W3 @ h2 + self.b3
        return _tanh(z + 0.5 * dz)  # residual + bounded latent

    def decode(self, z: np.ndarray, a: Optional[np.ndarray] = None) -> np.ndarray:
        zr = np.asarray(z, dtype=np.float64).ravel()
        if a is None:
            ab = np.zeros((1, 2))
        else:
            ab = self._action_basis(a)
        return 1.0 / (1.0 + np.exp(-(self.D @ zr + (self.V @ ab[0]) + self.db)))

    def predict_obs(self, kir: KernelIntermediateRepresentation,
                    action: Optional[PolicyAction]) -> np.ndarray:
        z = self.encode(encode_kir(kir))
        a = encode_action(action)
        z2 = self.transition(z, a)
        return self.decode(z2, a)

    def predict_kir(self, kir: KernelIntermediateRepresentation,
                    action: Optional[PolicyAction], dt_s: float = 0.5,
                    n_cpus: int = 2) -> KernelIntermediateRepresentation:
        o = self.predict_obs(kir, action)
        mean_util = float(np.clip(o[0], 0.02, 1.0))
        per_cpu = max(0, int(round(float(np.clip(o[1], 0.0, 1.0)) * _MAX_RUNNABLE / max(n_cpus, 1))))
        lat_ms = float(np.clip(o[2], 0.0025, 1.0) * _NORM_LAT_MS)
        rate = float(np.clip(o[3], 0.0, 1.0) * _MAX_SWITCH_RATE)
        cpus = {i: CPUMetrics(utilization=mean_util, runnable_tasks=per_cpu, irq_time=0.02)
                for i in range(max(n_cpus, len(kir.scheduler.cpus)))}
        return KernelIntermediateRepresentation(
            timestamp=kir.timestamp + dt_s,
            scheduler=SchedulerState(
                cpus=cpus,
                total_context_switches=int(kir.scheduler.total_context_switches + rate * dt_s),
                avg_latency_ms=round(max(0.05, lat_ms), 4),
            ),
        )

    # -- training: single-step + optional unrolled multi-step loss (see fit below) -- #

    def _forward_backward(self, F: np.ndarray, A: np.ndarray, Y: np.ndarray):
        n = len(F)
        # forward with caches
        Z = _tanh(F @ self.E.T + self.eb)                      # (n, L)
        X = np.concatenate([Z, A.reshape(n, 1)], axis=1)       # (n, L+1)
        H1pre = X @ self.W1.T + self.b1
        H1 = _tanh(H1pre)
        H2pre = H1 @ self.W2.T + self.b2
        H2 = _tanh(H2pre)
        DZ = H2 @ self.W3.T + self.b3                          # (n, L)
        Z2 = _tanh(Z + 0.5 * DZ)
        A2 = self._action_basis(A)                       # (n, 2)
        Opre = Z2 @ self.D.T + A2 @ self.V.T + self.db
        O = 1.0 / (1.0 + np.exp(-Opre))
        err = O - Y
        loss = float(np.mean(err ** 2))
        # backward
        dOpre = (2.0 / (n * OBS_DIM)) * err * O * (1 - O)       # (n, 4)
        gD = dOpre.T @ Z2
        gV = dOpre.T @ A2
        gdb = dOpre.sum(axis=0)
        dZ2 = dOpre @ self.D * (1 - Z2 ** 2)
        dDZ = dZ2 * 0.5
        dZ_via_res = dZ2
        gW3 = dDZ.T @ H2
        gb3 = dDZ.sum(axis=0)
        dH2 = dDZ @ self.W3 * (1 - H2 ** 2)
        gW2 = dH2.T @ H1
        gb2 = dH2.sum(axis=0)
        dH1 = dH2 @ self.W2 * (1 - H1 ** 2)
        gW1 = dH1.T @ X
        gb1 = dH1.sum(axis=0)
        dX = dH1 @ self.W1
        dZ = dX[:, :self.latent_dim] + dZ_via_res
        dZpre = dZ * (1 - Z ** 2)
        gE = dZpre.T @ F
        geb = dZpre.sum(axis=0)
        return loss, [gE, geb, gW1, gb1, gW2, gb2, gW3, gb3, gD, gV, gdb]

    # -- unrolled (multi-step) training: planning-relevant fidelity -- #
    #
    # Single-step MSE rewards matching average outcomes; planners consume
    # COMPOUNDED outcomes, where per-step shape errors (e.g. underpredicted
    # tight-window latency) accumulate into inverted rankings. The unrolled
    # loss rolls the model open-loop for H steps from each window start and
    # penalizes all H predicted observables — backprop through time over the
    # residual transition (MuZero-style: train the model for the use it gets).

    def _unrolled_forward_backward(self, Fw: np.ndarray, Aw: np.ndarray,
                                   Yw: np.ndarray):
        N, H, _ = Fw.shape
        L = self.latent_dim
        # forward with per-step caches
        Z = [None] * (H + 1)
        Z[0] = _tanh(Fw[:, 0, :] @ self.E.T + self.eb)          # (N, L)
        H1, H1p, H2, H2p, DZ, A2, O, Op = [], [], [], [], [], [], [], []
        for h in range(H):
            a = Aw[:, h]
            A2h = self._action_basis(a)                          # (N, 2)
            A2.append(A2h)
            x = np.concatenate([Z[h], a.reshape(N, 1)], axis=1)  # (N, L+1)
            h1p = x @ self.W1.T + self.b1
            h1 = _tanh(h1p)
            h2p = h1 @ self.W2.T + self.b2
            h2 = _tanh(h2p)
            dz = h2 @ self.W3.T + self.b3
            z2 = _tanh(Z[h] + 0.5 * dz)
            op = z2 @ self.D.T + A2h @ self.V.T + self.db
            o = 1.0 / (1.0 + np.exp(-op))
            H1p.append(h1p); H1.append(h1); H2p.append(h2p); H2.append(h2)
            DZ.append(dz); O.append(o); Op.append(op); Z[h + 1] = z2
        Oall = np.stack(O, axis=1)                               # (N, H, 4)
        err = Oall - Yw
        loss = float(np.mean(err ** 2))
        scale = 2.0 / (N * H * OBS_DIM)
        # backward through time
        gE = np.zeros_like(self.E); geb = np.zeros_like(self.eb)
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
        gW3 = np.zeros_like(self.W3); gb3 = np.zeros_like(self.b3)
        gD = np.zeros_like(self.D); gV = np.zeros_like(self.V)
        gdb = np.zeros_like(self.db)
        carry = np.zeros((N, L))  # dL/dz_{h+1} from later steps
        for h in range(H - 1, -1, -1):
            d_op = scale * err[:, h, :] * O[h] * (1 - O[h])      # (N, 4)
            gD += d_op.T @ Z[h + 1]
            gV += d_op.T @ A2[h]
            gdb += d_op.sum(axis=0)
            d_znext = d_op @ self.D + carry          # dL/dz_{h+1} pre-tanh-gate
            d_pre = d_znext * (1 - Z[h + 1] ** 2)     # through z'=tanh(pre)
            d_dz = d_pre * 0.5
            d_z_direct = d_pre
            gb3 += d_dz.sum(axis=0)
            gW3 += d_dz.T @ H2[h]
            d_h2 = d_dz @ self.W3 * (1 - H2[h] ** 2)
            gb2 += d_h2.sum(axis=0)
            gW2 += d_h2.T @ H1[h]
            d_h1 = d_h2 @ self.W2 * (1 - H1[h] ** 2)
            gb1 += d_h1.sum(axis=0)
            gW1 += d_h1.T @ np.concatenate(
                [Z[h], Aw[:, h].reshape(N, 1)], axis=1)
            d_x = d_h1 @ self.W1
            carry = d_z_direct + d_x[:, :L]
        d_z0 = carry * (1 - Z[0] ** 2)
        gE += d_z0.T @ Fw[:, 0, :]
        geb += d_z0.sum(axis=0)
        return loss, [gE, geb, gW1, gb1, gW2, gb2, gW3, gb3, gD, gV, gdb]

    def fit_unrolled(self, Fw: np.ndarray, Aw: np.ndarray, Yw: np.ndarray,
                     epochs: int = 100, lr: float = 0.02,
                     verbose: bool = False) -> List[float]:
        return self._run_adam(
            lambda: self._unrolled_forward_backward(Fw, Aw, Yw),
            epochs, lr, verbose)

    def fit(self, feats: np.ndarray, acts: np.ndarray, targets: np.ndarray,
            epochs: int = 300, lr: float = 0.02, verbose: bool = False,
            windows=None, unroll_weight: float = 0.0) -> List[float]:
        """Single-step fit, optionally mixed with an unrolled multi-step term.

        windows = (Fw, Aw, Yw); total grad = g_single + unroll_weight*g_unrolled.
        windows=None or weight 0 reproduces the legacy single-step path exactly.
        """
        F = np.asarray(feats, dtype=np.float64)
        A = np.asarray(acts, dtype=np.float64)
        Y = np.asarray(targets, dtype=np.float64)
        n = len(F)
        batch = min(n, 256)
        rng = np.random.default_rng(self.seed ^ 0xABCD)
        losses: List[float] = []

        def one_round():
            idx = rng.permutation(n)[:batch]
            return self._forward_backward(F[idx], A[idx], Y[idx])

        if windows is None or not unroll_weight:
            return self._run_adam(one_round, epochs, lr, verbose)

        Fw, Aw, Yw = (np.asarray(w, dtype=np.float64) for w in windows)

        def mixed_round():
            l1, g1 = one_round()
            l2, g2 = self._unrolled_forward_backward(Fw, Aw, Yw)
            losses.append(l1 + unroll_weight * l2)
            return None, [g + unroll_weight * gu for g, gu in zip(g1, g2)]

        self._run_adam(mixed_round, epochs, lr, verbose)
        return losses

    def _run_adam(self, grad_fn, epochs: int, lr: float,
                  verbose: bool) -> List[float]:
        params = [self.E, self.eb, self.W1, self.b1, self.W2, self.b2,
                  self.W3, self.b3, self.D, self.V, self.db]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        losses: List[float] = []
        for ep in range(1, epochs + 1):
            out = grad_fn()
            if isinstance(out, tuple) and out[0] is None:
                grads = out[1]
            else:
                loss, grads = out
                losses.append(loss)
            t = ep
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                mh = m[i] / (1 - b1 ** t)
                vh = v[i] / (1 - b2 ** t)
                p -= lr * mh / (np.sqrt(vh) + eps)
            if verbose and ep % 100 == 0:
                shown = losses[-1] if losses else float("nan")
                print(f"    [dynamics] epoch {ep}/{epochs} loss={shown:.5f}")
        return losses

    # -- persistence -- #
    def to_dict(self) -> dict:
        return {"seed": self.seed, "latent_dim": self.latent_dim,
                "format": "vk-transition-v3",
                **{k: getattr(self, k).tolist() for k in
                   ("E", "eb", "W1", "b1", "W2", "b2", "W3", "b3", "D", "V", "db")}}

    def save(self, path: str) -> str:
        import json
        import os
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        return path

    @classmethod
    def load(cls, path: str) -> "TransitionModel":
        import json
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "TransitionModel":
        m = cls(seed=d.get("seed", 0), latent_dim=d.get("latent_dim", LATENT_DIM))
        for k in ("E", "eb", "W1", "b1", "W2", "b2", "W3", "b3", "D", "db"):
            setattr(m, k, np.array(d[k], dtype=np.float64))
        if "V" in d:  # tolerant loader: v1 has no V, v2 has (4,1), v3 has (4,2)
            V = np.array(d["V"], dtype=np.float64)
            if V.shape == (OBS_DIM, 1):
                m.V = np.concatenate([V, np.zeros((OBS_DIM, 1))], axis=1)
            elif V.shape == (OBS_DIM, 2):
                m.V = V
        return m

    def param_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for k in ("E", "W1", "W2", "W3", "D", "V"):
            h.update(memoryview(getattr(self, k).tobytes()))
        return h.hexdigest()[:16]

    # -- evaluation on a held-out transition set -- #
    def evaluate(self, feats: np.ndarray, acts: np.ndarray,
                 targets: np.ndarray) -> dict:
        """Single-step MSE/MAE + per-output MSE (no oracle needed)."""
        F = np.asarray(feats, dtype=np.float64)
        A = np.asarray(acts, dtype=np.float64).ravel()
        Y = np.asarray(targets, dtype=np.float64)
        Z = _tanh(F @ self.E.T + self.eb)
        X = np.concatenate([Z, A.reshape(len(F), 1)], axis=1)
        H1 = _tanh(X @ self.W1.T + self.b1)
        H2 = _tanh(H1 @ self.W2.T + self.b2)
        Z2 = _tanh(Z + 0.5 * (H2 @ self.W3.T + self.b3))
        A2 = self._action_basis(A)
        O = 1.0 / (1.0 + np.exp(-(Z2 @ self.D.T + A2 @ self.V.T + self.db)))
        err = O - Y
        names = ("mean_util", "runnable_norm", "lat_norm", "switch_rate_norm")
        return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))),
                "per_output_mse": {n: float(np.mean(err[:, j] ** 2))
                                   for j, n in enumerate(names)},
                "n": len(F)}


@dataclass
class DynamicsDataset:
    feats: np.ndarray
    acts: np.ndarray
    targets: np.ndarray
    seeds: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.feats)


def collect_dataset(n_episodes: int = 24, steps_per_episode: int = 40,
                    n_cpus: int = 2, base_seed: int = 7000,
                    lat_choices: Tuple[float, ...] = (1000, 2000, 4000, 6000, 12000, 24000),
                    dt_s: float = 0.5) -> DynamicsDataset:
    """Sample (s, a, s') tuples from the exact oracle (KernelSimulator).

    Random-action rollouts with log-uniform-ish latency choices + occasional
    action repeats, so the dataset covers the (state, action) product space
    instead of one observational trajectory (the bug the old trainer had).
    """
    from learned_kernel.simulator.env import KernelSimulator
    from learned_kernel.policy.schemas import PolicyAction, SchedulerAction

    rng = np.random.default_rng(base_seed)
    F, A, Y, seeds = [], [], [], []
    for ep in range(n_episodes):
        sim = KernelSimulator(seed=base_seed + ep, n_cpus=n_cpus, dt_s=dt_s)
        kir = sim.current_kir()
        prev_sw = kir.scheduler.total_context_switches
        cur_lat = float(rng.choice(np.array(lat_choices, dtype=float)))
        for _ in range(steps_per_episode):
            if rng.random() < 0.35:
                cur_lat = float(rng.choice(np.array(lat_choices, dtype=float)))
            act = PolicyAction(policy_id="probe",
                               scheduler=SchedulerAction(target_latency_us=int(cur_lat)))
            F.append(encode_kir(kir))
            A.append(encode_action(act)[0])
            nxt = sim.step(act)
            Y.append(kir_observables(nxt, prev_sw, dt_s))
            prev_sw = nxt.scheduler.total_context_switches
            kir = nxt
        seeds.append(base_seed + ep)
    return DynamicsDataset(feats=np.array(F), acts=np.array(A), targets=np.array(Y), seeds=seeds)


def train_model(dataset: DynamicsDataset, seed: int = 0,
                epochs: int = 300, lr: float = 0.02,
                verbose: bool = False) -> TransitionModel:
    model = TransitionModel(seed=seed)
    model.fit(dataset.feats, dataset.acts, dataset.targets,
              epochs=epochs, lr=lr, verbose=verbose)
    return model
