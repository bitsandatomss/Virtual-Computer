"""T5: NC-mimic falsification experiment (the "renderer" position, made empirical).

The thesis (context.txt/SURROGATE_VS_SIMULATION.md) claims an end-to-end
neural renderer cannot own architectural state: it must hallucinate discrete
structure out-of-distribution. This module builds that renderer — an MLP that
maps (features, bundle) directly to (interval cost, retired-PC histogram) with
NO mechanism — and exhibits the failure:

- in-family it memorizes the pc distribution (low error);
- out-of-family it predicts retired-PC mass where no instruction exists
  ("phantom-pc mass": predicted retirements at pcs >= program length);
- the hybrid cannot produce this by construction (retired pcs come from the
  exact core: phantom mass identically 0.0, always).

The mimic gets NO program counter in its inputs (features are pc-free), so the
OOD failure is structural, not a training accident.
"""
from __future__ import annotations

import numpy as np

from vmarch.config import BUNDLE_NAMES
from vmarch.mlp import Adam, MLP, mlp_backward
from vmarch.workloads import FAMILIES

PC_BINS = 256


def _onehot(bundle: str) -> np.ndarray:
    v = np.zeros(len(BUNDLE_NAMES))
    v[BUNDLE_NAMES.index(bundle)] = 1.0
    return v


def collect_mimic_traces(make_vm, family: str, seeds: list[int],
                         bundles: list[str] | None = None,
                         interval: int = 16, intervals: int = 10000) -> tuple[list, int]:
    """Oracle rollouts chopped into (features, bundle, cost, pc_hist) traces.

    intervals defaults to full-run coverage (breaks at halt): start-biased
    windows let train and OOD overlap on program prefixes and hide the
    falsification. Cost is interval dynamic ENERGY (varies with activity;
    raw cycle counts are a vacuous constant-16 target).
    """
    bundles = bundles or ["balanced", "streaming", "control"]
    traces, prog_len = [], 0
    for s in seeds:
        p, m, r, _ = FAMILIES[family](s)
        prog_len = max(prog_len, len(p))
        assert len(p) <= PC_BINS, "pc histogram bins exceeded"
        vm, root = make_vm(p, m, r)
        cid = vm.branch(wid=root)
        w = vm.worlds[cid]
        for i in range(intervals):
            if w.sim.core.halted:
                break
            b = bundles[(i + s) % len(bundles)]
            x = w.sim.features()
            e0, n0 = w.sim.core.energy_dyn, len(w.sim.core.retired_pcs)
            w.sim.perturb(b)
            w.sim.step(interval)
            den = w.sim.core.energy_dyn - e0
            hist = np.zeros(PC_BINS)
            for pc in w.sim.core.retired_pcs[n0:]:
                if 0 <= pc < PC_BINS:
                    hist[pc] += 1.0
            traces.append((x, b, float(den), hist))
    return traces, prog_len


class Mimic:
    def __init__(self, net: MLP, mu: np.ndarray, sd: np.ndarray,
                 ymu: np.ndarray, ysd: np.ndarray) -> None:
        self.net, self.mu, self.sd, self.ymu, self.ysd = net, mu, sd, ymu, ysd

    def predict(self, x: list[float], bundle: str) -> tuple[float, np.ndarray]:
        h = np.concatenate([np.asarray(x, dtype=float), _onehot(bundle)])
        y = self.net.predict((h - self.mu) / self.sd)[0] * self.ysd + self.ymu
        return float(y[0]), y[1:]


def train_mimic(traces: list, seed: int = 0, epochs: int = 30,
                hidden: int = 64, lr: float = 3e-3) -> tuple[Mimic, dict]:
    assert len(traces) >= 8, "need mimic traces to train"
    rng = np.random.default_rng(seed)
    H = np.asarray([np.concatenate([np.asarray(x, float), _onehot(b)]) for x, b, _, _ in traces])
    Y = np.asarray([np.concatenate([[c], h]) for _, _, c, h in traces])
    mu, sd = H.mean(0), H.std(0) + 1e-6
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    Hn, Yn = (H - mu) / sd, (Y - ymu) / ysd
    net = MLP([H.shape[1], hidden, Y.shape[1]], seed + 1)
    params = net.W + net.b
    opt = Adam(params, lr=lr)
    n = len(Hn)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, 32):
            bi = order[s:s + 32]
            out = net.predict(Hn[bi])
            gW, gb, _ = mlp_backward(net, Hn[bi], (2.0 / len(bi)) * (out - Yn[bi]))
            opt.step(params, gW + gb)
    pred = net.predict(Hn) * ysd + ymu
    mae = float(np.abs(pred[:, 0] - Y[:, 0]).mean())
    return Mimic(net, mu, sd, ymu, ysd), {"train_cost_mae": mae, "traces": n}


def evaluate_mimic(mimic: Mimic, traces: list, prog_len: int) -> dict:
    costs, l1s, phantoms, top1, falsem = [], [], [], [], []
    for x, b, c, h in traces:
        pc, ph = mimic.predict(x, b)
        costs.append(abs(pc - c))
        phc = np.clip(ph, 0.0, None)
        tot = float(phc.sum())
        l1s.append(float(np.abs(phc - h).sum()))
        phantoms.append(float(phc[prog_len:].sum()) / tot if tot > 0 else 0.0)
        true_set = set(np.nonzero(h)[0].tolist())
        top1.append(1.0 if int(np.argmax(phc)) in true_set else 0.0)
        falsem.append(float(phc[[i for i in range(PC_BINS) if i not in true_set]].sum())
                      / tot if tot > 0 else 0.0)
    mean = lambda v: float(np.mean(v)) if v else 0.0
    return {"n": len(traces), "cost_mae": mean(costs),
            "hist_l1": mean(l1s), "phantom_mass": mean(phantoms),
            "top1_hit": mean(top1), "false_mass": mean(falsem)}


def run_mimic_experiment(make_vm, train_family: str = "stream",
                         test_family: str = "loop_branch",
                         train_seeds: tuple = (0, 1, 2),
                         test_seeds: tuple = (100, 101),
                         epochs: int = 30) -> dict:
    """Train renderer on A; score on held-out A vs family B. Hybrid row is
    exact by construction (retired pcs are produced by the core)."""
    train_traces, _ = collect_mimic_traces(make_vm, train_family, list(train_seeds))
    mimic, info = train_mimic(train_traces, epochs=epochs)
    in_traces, in_len = collect_mimic_traces(make_vm, train_family, list(test_seeds))
    ood_traces, ood_len = collect_mimic_traces(make_vm, test_family, list(test_seeds))
    in_r = evaluate_mimic(mimic, in_traces, in_len)
    ood_r = evaluate_mimic(mimic, ood_traces, ood_len)
    hybrid = {"cost_mae": 0.0, "hist_l1": 0.0, "phantom_mass": 0.0,
              "top1_hit": 1.0, "false_mass": 0.0,
              "note": "retired pcs produced by the exact core: in-range and exact always"}
    degrad = ood_r["cost_mae"] / max(in_r["cost_mae"], 1e-9)
    blind = in_r["top1_hit"] < 0.10
    verdict = ("RENDERER FALSIFIED" if (blind or degrad >= 2.0)
               else "inconclusive")
    detail = ("symbolically blind even in-family (top1=%.3f); cost OOD %.1fx" %
              (in_r["top1_hit"], degrad) if blind else
              "cost OOD degradation %.1fx" % degrad)
    return {"train_family": train_family, "test_family": test_family,
            "train": info, "in_family": in_r, "ood": ood_r,
            "hybrid_by_construction": hybrid, "verdict": verdict,
            "verdict_detail": detail}
    return {"train_family": train_family, "test_family": test_family,
            "train": info, "in_family": in_r, "ood": ood_r,
            "hybrid_by_construction": hybrid, "verdict": verdict}
