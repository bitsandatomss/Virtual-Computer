"""Phase 2 — from predictor to state-transition model (context.txt Phase 2).

Instead of f(x, g) = y, build F(z, a) = z' with persistent latent state, and
show the payoff: open-loop multi-step rollouts that a single-step predictor
used autoregressively cannot match without a transition objective. Compares
divergence curves of the transition model vs a naive repeat-last-observation
baseline on held-out workloads.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(full: bool = False) -> dict:
    import numpy as np
    from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
    from virtual_kernel import EnsembleDynamics, TruthOracle, fast_config
    from virtual_kernel.datasets import _reseed, to_transitions
    from virtual_kernel.dynamics import encode_kir
    from virtual_kernel.metrics import divergence_stats

    cfg = fast_config("phase2")
    n_tr, steps = (12, 25) if full else (4, 10)
    tr = to_transitions(_reseed([], list(range(7000, 7000 + n_tr)), steps,
                                cfg.arms, cfg.n_cpus, cfg.dt_s), cfg.dt_s)
    ens = EnsembleDynamics.train(n_members=3 if full else 2, epochs=150 if full else 30,
                                 n_cpus=cfg.n_cpus, dataset=tr)
    oracle = TruthOracle(seed=9000, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)
    kir0 = _reseed([], [9000], steps, cfg.arms, cfg.n_cpus, cfg.dt_s)[0].steps[0].kir
    acts = [PolicyAction(policy_id="p2", scheduler=SchedulerAction(
        target_latency_us=a)) for a in (2000, 6000, 12000, 6000)]
    rep = oracle.evaluate(ens, kir0, acts, divergence_threshold=cfg.divergence_threshold)
    # naive baseline: predict no-change from kir0
    oracle.reset(kir0)
    f0 = encode_kir(kir0)
    naive = []
    for a in acts:
        ko = oracle.step(a)
        naive.append(float(np.abs(encode_kir(ko) - f0).mean()))
    print(f"[phase2] transition-model drift: {rep.rollout_mae_curve}")
    print(f"[phase2] naive no-change drift:  {[round(v, 4) for v in naive]}")
    print(f"[phase2] {divergence_stats(rep.rollout_mae_curve, cfg.divergence_threshold)}")
    return {"model_curve": rep.rollout_mae_curve, "naive_curve": naive}


if __name__ == "__main__":
    run(full="--full" in sys.argv)
