"""Phase 5 — agents inside the virtual kernel (context.txt Phase 5).

Research/engineering agents operate OVER the virtual kernel: hypothesis,
perturbation search, analysis, adversarial challenge with revision, robust
selection across demand scenarios, and oracle validation. The virtual
laboratory: millions of virtual trajectories in, a handful of physical
probes out, results feeding back into the model.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(full: bool = False) -> dict:
    from learned_kernel.simulator.env import KernelSimulator
    from virtual_kernel import (
        EnsembleDynamics, TruthOracle, VirtualKernel, VirtualLab,
        fast_config, robust_search,
    )
    from virtual_kernel.datasets import capture_traces

    cfg = fast_config("phase5")
    ens = EnsembleDynamics.train(
        n_members=3 if full else 2, n_episodes=8 if full else 4,
        steps_per_episode=20 if full else 10,
        epochs=120 if full else 30, n_cpus=cfg.n_cpus)
    kir0 = KernelSimulator(seed=11, n_cpus=cfg.n_cpus).current_kir()
    vk = VirtualKernel(ens, kir0, dt_s=cfg.dt_s)
    lab = VirtualLab(vk, oracle=TruthOracle(seed=11, n_cpus=cfg.n_cpus))
    rep = lab.run(horizon=3 if full else 2, beam_width=6 if full else 3,
                  validate_k=1, debate_rounds=2 if full else 1)
    for f in rep.findings:
        print(f"[phase5][{f.agent}] {f.claim}")
    # robust selection across demand scenarios (workload as adversary)
    eps = capture_traces(4 if full else 2, 6, base_seed=5000,
                         arms=cfg.arms, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)
    scenarios = vk.scenario_kirs(eps)
    cands = [rep.best_latencies, [6000] * len(rep.best_latencies),
             [2000] * len(rep.best_latencies)]
    rob = robust_search(ens, scenarios, cands)
    print(f"[phase5] robust selection: {rob['selected']}")
    print(f"[phase5] {rep.summary()}")
    return {"lab": rep.to_dict(), "robust": rob}


if __name__ == "__main__":
    run(full="--full" in sys.argv)
