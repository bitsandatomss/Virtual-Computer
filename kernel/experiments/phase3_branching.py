"""Phase 3 — branching + sequential interventions (context.txt Phase 3).

X0 -> {X1..Xn} counterfactual worlds, then sequential chains X0->X1->X2->X3
with compare() scoring. Demonstrates the environment (not predictor) claim:
multiple futures coexist and are scored against each other virtually.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(full: bool = False) -> dict:
    from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
    from learned_kernel.simulator.env import KernelSimulator
    from virtual_kernel import EnsembleDynamics, VirtualKernel, fast_config

    cfg = fast_config("phase3")
    ens = EnsembleDynamics.train(
        n_members=3 if full else 2, n_episodes=8 if full else 4,
        steps_per_episode=20 if full else 10,
        epochs=120 if full else 30, n_cpus=cfg.n_cpus)
    kir0 = KernelSimulator(seed=11, n_cpus=cfg.n_cpus).current_kir()
    vk = VirtualKernel(ens, kir0, dt_s=cfg.dt_s)
    seqs = {"tight": [2000, 2000, 2000], "default": [6000, 6000, 6000],
            "loose": [12000, 12000, 12000]}
    for name, seq in seqs.items():
        vk.branch(name)
        for lat in seq:
            vk.intervene(PolicyAction(policy_id="p3", scheduler=SchedulerAction(
                target_latency_us=lat)), branch=name)
    for a in ("tight", "loose"):
        print(f"[phase3] main-vs-{a}: {vk.compare('main', a)}")
    print(f"[phase3] tight-vs-loose: {vk.compare('tight', 'loose')}")
    return {"comparisons": {a: vk.compare("main", a) for a in seqs}}


if __name__ == "__main__":
    run(full="--full" in sys.argv)
