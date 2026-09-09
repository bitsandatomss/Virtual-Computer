"""Phase 6 — the closed scientific loop (context.txt lines 445-479, 603-623).

The file's endpoint: simulation + surrogate + reality in one loop —

    observe -> learn -> simulate -> identify uncertainty -> choose experiment
        -> observe -> (surrogate improves) -> repeat

The SIMULATOR provides known structure (exact scheduler dynamics). The
SURROGATE provides learned structure cheap to query. REALITY (here: the
exact oracle standing in for physical measurement) provides ground truth.
Each loop iteration must PROVE its value: the surrogate's held-out error
must fall, and the manifest must record every number's provenance.

This is the L5 behavior assessed in isolation: the surrogate as an
epistemic instrument that decides what reality needs to be queried next.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks._common import ensure_dir, log_manifest, save_json


def run(full: bool = False) -> dict:
    from learned_kernel.simulator.env import KernelSimulator
    from virtual_kernel import (
        EnsembleDynamics, TruthOracle, VKConfig, active_loop, fast_config,
    )
    from virtual_kernel.datasets import dataset_hash, train_holdout_split
    from virtual_kernel.levels import assess_L1, assess_L3
    from virtual_kernel.ood import OODDetector

    cfg = VKConfig(name="closed-loop-full") if full else fast_config("closed-loop")
    n_ep, steps = (16, 30) if full else (5, 12)
    rounds = 3 if full else 2

    # observe: zero-shot split = "reality" divided into seen / unseen workloads
    train_ds, test_ds, train_eps, test_eps = train_holdout_split(
        cfg.train_seeds if full else list(range(7000, 7005)),
        cfg.test_seeds if full else list(range(9000, 9002)),
        steps, cfg.arms, cfg.n_cpus, cfg.dt_s)
    # learn: surrogate from seen reality
    ens = EnsembleDynamics.train(n_members=3 if full else 2,
                                 epochs=150 if full else 40,
                                 dataset=train_ds)
    kir0 = test_eps[0].steps[0].kir
    oracle = TruthOracle(seed=cfg.seed, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)

    pre = {"L1": assess_L1(ens, train_eps),
           "L3": assess_L3(ens, [ep.steps[len(ep.steps) // 2].kir for ep in test_eps[:4]],
                           seed=cfg.seed, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)}
    det = OODDetector().fit(train_ds.feats)
    ood_pre = det.report(train_ds.feats, test_ds.feats)

    # simulate -> identify uncertainty -> choose experiment -> observe
    loop = active_loop(ens, kir0, oracle, rounds=rounds, probes_per_round=4,
                       epochs=60 if full else 20, dt_s=cfg.dt_s)

    post = {"L1": assess_L1(ens, train_eps),
            "L3": assess_L3(ens, [ep.steps[len(ep.steps) // 2].kir for ep in test_eps[:4]],
                            seed=cfg.seed, n_cpus=cfg.n_cpus, dt_s=cfg.dt_s)}
    out = {
        "config_hash": cfg.hash,
        "pre": {k: {"passed": v["passed"], "metrics": v["metrics"]} for k, v in pre.items()},
        "post": {k: {"passed": v["passed"], "metrics": v["metrics"]} for k, v in post.items()},
        "active_rounds": loop.rounds,
        "active_oracle_calls": loop.n_oracle_calls,
        "ood_test_flag_rate": ood_pre["test_flag_rate"],
        "closed_loop_gain": {
            "L3_spearman_pre": pre["L3"]["metrics"]["spearman"],
            "L3_spearman_post": post["L3"]["metrics"]["spearman"],
        },
    }
    rep_dir = ensure_dir(os.path.join(cfg.report_dir, "closed-loop"))
    save_json(os.path.join(rep_dir, "results.json"), {"config": cfg.to_dict(), **out})
    md = ("# Closed-loop experiment — surrogate as epistemic instrument\n\n"
          f"config `{cfg.hash}`\n\n"
          f"L1 replay: {pre['L1']['metrics']['train_mse']:.5f} -> "
          f"{post['L1']['metrics']['train_mse']:.5f}\n\n"
          f"L3 spearman: {out['closed_loop_gain']['L3_spearman_pre']:.3f} -> "
          f"{out['closed_loop_gain']['L3_spearman_post']:.3f}\n\n"
          f"active oracle calls: {loop.n_oracle_calls}, "
          f"OOD test flag rate: {ood_pre['test_flag_rate']:.2f}\n")
    with open(os.path.join(rep_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        log_manifest("closed-loop", cfg, dataset_hash(ens.dataset),
                     ens.hashes(), {"gain": out["closed_loop_gain"]})
    except Exception as e:
        out["manifest_warning"] = str(e)
    print(md)
    return out


if __name__ == "__main__":
    run(full="--full" in sys.argv)
