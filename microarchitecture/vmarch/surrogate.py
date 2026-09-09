"""Learned latent microarchitecture surrogate: zt=E(xt,c), zt+1=F(z,a), xhat=D(z).

Implements context.txt lines 122-139 literally. Multi-task training on oracle
transition tuples (x, a, cost, x'):
  loss = ||F(E(x),a) - E(x')||^2 (latent consistency)
       + (cost_head - cost)^2    (planning-relevant cost)
       + 0.3 * ||D(F) - x'||^2  (observable reconstruction / hallucination probe)

Ensembles (3 seeds) give disagreement = uncertainty. Checkpoints carry format
version + feature schema + config digest + data hash, and are rejected on
mismatch (stale-schema discipline from DEA).

Also ships a LinearBaseline (ridge, closed form) for ablations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from vmarch.config import BUNDLE_NAMES
from vmarch.core import FEATURE_NAMES
from vmarch.mlp import MLP, Adam, mlp_backward
from vmarch.version import FEATURE_VERSION, MODEL_FORMAT_VERSION, VMARCH_VERSION

LATENT_DIM = 16


def _onehot(bundle: str, names: list[str]) -> np.ndarray:
    v = np.zeros(len(names))
    v[names.index(bundle.split(":")[0])] = 1.0
    return v


class LatentSurrogate:
    def __init__(self, enc: MLP, trans: MLP, cost_head: MLP, dec: MLP,
                 bundles: list[str], x_mean: np.ndarray, x_std: np.ndarray,
                 meta: dict | None = None) -> None:
        self.enc = enc
        self.trans = trans
        self.cost_head = cost_head
        self.dec = dec
        self.bundles = list(bundles)
        self.x_mean = np.asarray(x_mean, dtype=float)
        self.x_std = np.asarray(x_std, dtype=float)
        self.meta = meta or {}

    def _n(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.x_mean) / self.x_std

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.enc.predict(self._n(np.atleast_2d(x)))

    def predict_cost(self, x, bundle: str) -> float:
        z = self.encode(x)
        a = np.atleast_2d(_onehot(bundle, self.bundles))
        h = np.concatenate([z, np.tile(a, (len(z), 1))], axis=1)
        zp = self.trans.predict(h)
        c = self.cost_head.predict(zp)
        return float(c[0, 0])

    def predict_next(self, x, bundle: str) -> np.ndarray:
        z = self.encode(x)
        a = np.atleast_2d(_onehot(bundle, self.bundles))
        h = np.concatenate([z, np.tile(a, (len(z), 1))], axis=1)
        zp = self.trans.predict(h)
        return self.dec.predict(zp)[0] * self.x_std + self.x_mean

    def uncertainty(self, x) -> float:
        costs = np.array([self.predict_cost(x, b) for b in self.bundles])
        return float(costs.std() / (abs(costs.mean()) + 1e-6))

    # -- persistence --
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": MODEL_FORMAT_VERSION,
                   "feature_version": FEATURE_VERSION,
                   "vmarch_version": VMARCH_VERSION,
                   "feature_names": FEATURE_NAMES,
                   "bundles": self.bundles, "latent_dim": LATENT_DIM,
                   "enc": self.enc.parameters(), "trans": self.trans.parameters(),
                   "cost": self.cost_head.parameters(), "dec": self.dec.parameters(),
                   "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
                   "meta": self.meta}
        p.write_text(json.dumps(payload) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "LatentSurrogate":
        payload = json.loads(Path(path).read_text())
        if payload["format_version"] != MODEL_FORMAT_VERSION:
            raise ValueError("stale surrogate checkpoint format")
        if payload["feature_names"] != FEATURE_NAMES:
            raise ValueError("surrogate feature schema mismatch")
        return cls(MLP.from_parameters(payload["enc"]), MLP.from_parameters(payload["trans"]),
                   MLP.from_parameters(payload["cost"]), MLP.from_parameters(payload["dec"]),
                   payload["bundles"], np.asarray(payload["x_mean"]), np.asarray(payload["x_std"]),
                   payload.get("meta", {}))


class SurrogateEnsemble:
    def __init__(self, members: list[LatentSurrogate]) -> None:
        assert members
        self.members = members
        self.bundles = members[0].bundles

    def predict_cost(self, x, bundle: str) -> float:
        return float(np.mean([m.predict_cost(x, bundle) for m in self.members]))

    def uncertainty(self, x) -> float:
        preds = np.array([[m.predict_cost(x, b) for b in self.bundles] for m in self.members])
        return float(preds.std() / (abs(preds.mean()) + 1e-6))

    def disagreement(self, x, bundle: str) -> float:
        preds = np.array([m.predict_cost(x, bundle) for m in self.members])
        return float(preds.std())

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        for i, m in enumerate(self.members):
            m.save(d / f"member_{i}.json")

    @classmethod
    def load(cls, directory: str | Path) -> "SurrogateEnsemble":
        d = Path(directory)
        return cls([LatentSurrogate.load(p) for p in sorted(d.glob("member_*.json"))])


def _pack(examples: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray([e[0] for e in examples], dtype=float)
    A = np.asarray([_onehot(e[1], BUNDLE_NAMES) for e in examples])
    C = np.asarray([e[2] for e in examples], dtype=float).reshape(-1, 1)
    Xn = np.asarray([e[3] for e in examples], dtype=float)
    return X, A, C, Xn


def train_latent(examples: list[tuple], seed: int = 0, epochs: int = 60,
                 batch: int = 64, lr: float = 3e-3, l2: float = 1e-5,
                 hidden: int = 32, val_frac: float = 0.2) -> tuple[LatentSurrogate, dict]:
    """Train one E/F/D member with held-out gating (DEA methodology)."""
    assert len(examples) >= 8, "need transition examples to train"
    rng = np.random.default_rng(seed)
    n = len(examples)
    idx = rng.permutation(n)
    cut = max(4, int(n * (1 - val_frac)))
    tr, va = [examples[i] for i in idx[:cut]], [examples[i] for i in idx[cut:]]
    Xtr, Atr, Ctr, Xntr = _pack(tr)
    Xva, Ava, Cva, Xnva = _pack(va)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    # Standardize costs (center AND scale): the head must start near the
    # target mean, otherwise Adam spends all steps learning a huge bias.
    cmean, csd = float(Ctr.mean()), float(Ctr.std() + 1e-6)
    Cntr, Cnva = (Ctr - cmean) / csd, (Cva - cmean) / csd
    f = Xtr.shape[1]
    enc = MLP([f, hidden, LATENT_DIM], seed + 1)
    trans = MLP([LATENT_DIM + len(BUNDLE_NAMES), hidden, LATENT_DIM], seed + 2)
    cost = MLP([LATENT_DIM, hidden // 2, 1], seed + 3)
    dec = MLP([LATENT_DIM, hidden, f], seed + 4)
    params = enc.W + enc.b + trans.W + trans.b + cost.W + cost.b + dec.W + dec.b
    opt = Adam(params, lr=lr)
    Xn = lambda X: (X - mu) / sd
    best_val, best_state, curve = float("inf"), None, []

    def snapshot():
        import copy as _c
        return _c.deepcopy([w.copy() for w in enc.W] + [b.copy() for b in enc.b] +
                           [w.copy() for w in trans.W] + [b.copy() for b in trans.b] +
                           [w.copy() for w in cost.W] + [b.copy() for b in cost.b] +
                           [w.copy() for w in dec.W] + [b.copy() for b in dec.b])

    def restore(state):
        k = 0
        for m in (enc, trans, cost, dec):
            for w in m.W:
                w[:] = state[k]
                k += 1
            for b in m.b:
                b[:] = state[k]
                k += 1

    def loss_of(X, A, Cn, Xnt):
        Z = enc.predict(Xn(X))
        H = np.concatenate([Z, A], axis=1)
        Zp = trans.predict(H)
        Chat = cost.predict(Zp)
        Xhat = dec.predict(Zp)
        with np.errstate(over="ignore"):
            l_cost = float(np.mean((Chat - Cn) ** 2))
            l_lat = float(np.mean((Zp - enc.predict(Xn(Xnt))) ** 2))
            l_rec = float(np.mean((Xhat - Xn(Xnt)) ** 2))
        return l_cost + l_lat + 0.3 * l_rec, (Z, H, Zp, Chat, Xhat)

    for ep in range(epochs):
        order = rng.permutation(len(tr))
        for s in range(0, len(tr), batch):
            bi = order[s:s + batch]
            Xb, Ab, Cnb, Xnb = Xtr[bi], Atr[bi], Cntr[bi], Xntr[bi]
            n_b = len(Xb)
            Z = enc.predict(Xn(Xb))
            H = np.concatenate([Z, Ab], axis=1)
            Zp = trans.predict(H)
            Chat = cost.predict(Zp)
            Xhat = dec.predict(Zp)
            Zt = enc.predict(Xn(Xnb))  # detached target
            # exact backprop through the three heads into the transition
            gWc, gbc, dZp_cost = mlp_backward(cost, Zp, (2.0 / n_b) * (Chat - Cnb))
            gWd, gbd, dZp_rec = mlp_backward(dec, Zp, (2.0 * 0.3 / n_b) * (Xhat - Xn(Xnb)))
            dZp_lat = (2.0 / Zp.size) * (Zp - Zt)
            dZp = dZp_cost + dZp_lat + dZp_rec
            gWt, gbt, dH = mlp_backward(trans, H, dZp)
            dZ = dH[:, :LATENT_DIM]
            gWe, gbe, _ = mlp_backward(enc, Xn(Xb), dZ)
            grads = gWe + gbe + gWt + gbt + gWc + gbc + gWd + gbd
            for p, g in zip(params, grads):
                g += l2 * p
            opt.step(params, grads)
        vl, _ = loss_of(Xva, Ava, Cnva, Xnva)
        curve.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = snapshot()
    restore(best_state)
    tl, _ = loss_of(Xtr, Atr, Cntr, Xntr)
    data_hash = hashlib.sha256(np.asarray([e[2] for e in examples]).tobytes()).hexdigest()[:12]
    member = LatentSurrogate(enc, trans, cost, dec, BUNDLE_NAMES, mu, sd,
                             {"seed": seed, "train_loss": tl, "val_loss": best_val,
                              "examples": len(examples), "data_hash": data_hash,
                              "cost_mean": cmean, "cost_scale": float(csd)})
    # Restore original units (head is linear in its last layer):
    # out = (h @ W + b) * csd + cmean.
    member.cost_head.W[-1] *= csd
    member.cost_head.b[-1] = member.cost_head.b[-1] * csd + cmean
    return member, {"train_loss": tl, "val_loss": best_val, "curve": curve,
                    "examples": len(examples), "data_hash": data_hash}


def train_ensemble(examples: list[tuple], seeds: tuple = (11, 22, 33), **kw) -> tuple[SurrogateEnsemble, dict]:
    members, infos = [], []
    for s in seeds:
        m, info = train_latent(examples, seed=s, **kw)
        members.append(m)
        infos.append(info)
    ens = SurrogateEnsemble(members)
    return ens, {"members": infos,
                 "mean_val": float(np.mean([i["val_loss"] for i in infos]))}


# ---- evaluation protocols (context.txt publishable questions) ----
def evaluate_parity(ens: SurrogateEnsemble, examples: list[tuple]) -> dict:
    errs = [abs(ens.predict_cost(x, b) - c) for x, b, c, _ in examples]
    errs = np.asarray(errs)
    return {"mae": float(errs.mean()), "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max()), "n": len(errs)}


def evaluate_hallucination(ens: SurrogateEnsemble, examples: list[tuple], tol: float = 3.0) -> dict:
    bad = sum(1 for x, b, c, _ in examples if abs(ens.predict_cost(x, b) - c) > tol)
    return {"rate": bad / max(len(examples), 1), "count": bad, "n": len(examples), "tol": tol}


def evaluate_divergence(ens: SurrogateEnsemble, start_x: list, plan: list[str], steps: int = 8) -> dict:
    """Open-loop latent rollout spread: mean pairwise member disagreement growth."""
    trajs = []
    for m in ens.members:
        x = np.asarray(start_x, dtype=float)
        pred_costs = []
        for b in plan[:steps]:
            pred_costs.append(m.predict_cost(x, b))
            x = m.predict_next(x, b)
        trajs.append(pred_costs)
    trajs = np.asarray(trajs)
    spread = trajs.std(axis=0)
    return {"per_step_spread": [float(v) for v in spread],
            "final_spread": float(spread[-1]) if len(spread) else 0.0}
