"""Phase 1 — perturbation-response predictor (context.txt Phase 1).

The 'challenge entry' analog: given unperturbed kernel state + a sysctl
intervention, predict the post-intervention state. Scored on HELD-OUT
workloads (zero-shot), the brutally clean generalization test. No platform,
no swarm — just the best single-step model within the constraints.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(full: bool = False) -> dict:
    from virtual_kernel import EnsembleDynamics, fast_config
    from virtual_kernel.datasets import _reseed, dataset_hash, to_transitions

    cfg = fast_config("phase1")
    n_tr, n_te, steps = (12, 6, 25) if full else (4, 3, 10)
    tr = to_transitions(_reseed([], list(range(7000, 7000 + n_tr)), steps,
                                cfg.arms, cfg.n_cpus, cfg.dt_s), cfg.dt_s)
    te = to_transitions(_reseed([], list(range(9000, 9000 + n_te)), steps,
                                cfg.arms, cfg.n_cpus, cfg.dt_s), cfg.dt_s)
    ens = EnsembleDynamics.train(n_members=3 if full else 2, epochs=150 if full else 30,
                                 n_cpus=cfg.n_cpus, dataset=tr)
    print(f"[phase1] train_mse={ens.evaluate(tr)['mse']:.5f} "
          f"heldout_mse={ens.evaluate(te)['mse']:.5f} "
          f"(train {dataset_hash(tr)}, test {dataset_hash(te)})")
    return {"train_mse": ens.evaluate(tr)["mse"], "heldout_mse": ens.evaluate(te)["mse"]}


if __name__ == "__main__":
    run(full="--full" in sys.argv)
