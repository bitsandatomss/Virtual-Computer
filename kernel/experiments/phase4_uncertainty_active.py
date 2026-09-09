"""Phase 4 — uncertainty + active experimentation (context.txt Phase 4).

Virtual Kernel -> which physical probe should we run? Runs the active loop:
rank probes by disagreement, execute on the exact oracle, retrain, and show
mse_before -> mse_after per round. Uncertainty must PAY in oracle efficiency.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(full: bool = False) -> dict:
    from learned_kernel.simulator.env import KernelSimulator
    from virtual_kernel import EnsembleDynamics, TruthOracle, active_loop, fast_config

    cfg = fast_config("phase4")
    ens = EnsembleDynamics.train(
        n_members=3 if full else 2, n_episodes=8 if full else 4,
        steps_per_episode=20 if full else 10,
        epochs=100 if full else 30, n_cpus=cfg.n_cpus)
    kir0 = KernelSimulator(seed=11, n_cpus=cfg.n_cpus).current_kir()
    res = active_loop(ens, kir0, TruthOracle(seed=11, n_cpus=cfg.n_cpus),
                      rounds=3 if full else 2, probes_per_round=4,
                      epochs=40 if full else 15)
    for r in res.rounds:
        print(f"[phase4] round {r['round']}: probed={r['probed']} "
              f"mse {r['mse_before']:.5f} -> {r['mse_after']:.5f}")
    print(f"[phase4] oracle calls: {res.n_oracle_calls}")
    return {"rounds": res.rounds, "calls": res.n_oracle_calls}


if __name__ == "__main__":
    run(full="--full" in sys.argv)
