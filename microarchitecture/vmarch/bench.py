"""Evaluation protocols: the five trustworthiness requirements + maturity gates.

Implements context.txt 579–601 as runnable protocols and 662–684 as gates:
1. calibration      — disagreement-vs-error correlation + binned reliability
2. conservation     — architectural invariants (P0: retirement/R0/ordering/safety)
3. ood_detection    — cross-family and cross-config generalization gaps
4. uncertainty_prop — open-loop divergence horizon (spread vs rollout steps)
5. active_selection — validator precision (do top-ranked queries reduce error most?)

Plus: held_out() end-to-end (train → parity → killer benchmark → artifact),
maturity_gate() L1–L5 verdicts, and versioned artifact writing with overlap
refusal (disjoint train/select/test seed ranges).
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from vmarch.config import BUNDLE_NAMES
from vmarch.version import BENCH_PROTOCOL_VERSION, VMARCH_VERSION
from vmarch.workloads import FAMILIES, check_disjoint

# T7: pre-registered primary metric. All policy comparisons are descriptive;
# the confirmatory claim of a bench run is the value of this one entry.
PRIMARY_METRIC = "delta_surrogate_mcts"


def skill_scores(examples: list[tuple], ensemble) -> dict:
    """T4: skill vs naive baselines. Absolute MAE gates are vacuous (a
    constant predictor passes them); these fail constants by construction:
    mean-predictor scores exactly 0.0, last-value degrades off-distribution.
    """
    c = np.asarray([t[2] for t in examples], dtype=float)
    pred = np.asarray([ensemble.predict_cost(x, b) for x, b, _, _ in examples])
    mae = float(np.abs(pred - c).mean()) if len(c) else 0.0
    mae_mean = float(np.abs(c - c.mean()).mean()) if len(c) else 0.0
    mae_last = float(np.abs(c[1:] - c[:-1]).mean()) if len(c) > 1 else mae_mean
    sse = float(((c - pred) ** 2).sum())
    sst = float(((c - c.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return {"n": len(c), "mae": mae, "r2": r2,
            "mae_mean_baseline": mae_mean, "mae_last_baseline": mae_last,
            "skill_vs_mean": 1.0 - mae / mae_mean if mae_mean > 0 else 0.0,
            "skill_vs_last": 1.0 - mae / mae_last if mae_last > 0 else 0.0}


def calibration(examples: list[tuple], ensemble) -> dict:
    errs, dis = [], []
    for x, b, c, _ in examples:
        errs.append(abs(ensemble.predict_cost(x, b) - c))
        dis.append(ensemble.disagreement(x, b))
    errs, dis = np.asarray(errs), np.asarray(dis)
    corr = float(np.corrcoef(dis, errs)[0, 1]) if len(errs) > 2 and dis.std() > 0 else 0.0
    order = np.argsort(dis)
    k = max(1, len(order) // 4)
    quartiles = [float(errs[order[i * k:(i + 1) * k]].mean()) for i in range(4)]
    return {"disagreement_error_corr": corr, "error_by_disagreement_quartile": quartiles,
            "n": len(errs), "monotone": bool(all(b >= a for a, b in zip(quartiles, quartiles[1:])))}


def conservation_suite(make_vm, seeds: list[int]) -> dict:
    """P0 invariants. Any failure is a simulator bug, never 'model error'."""
    from vmarch.isa import assemble
    checks = []
    # R0 hardwired + ALU + store/load chain
    prog = assemble(["ADDI R0, R0, 99", "ADDI R1, R0, 5", "ADDI R2, R0, 7",
                     "ADD R3, R1, R2", "STORE R0, R3, 100", "LOAD R4, R0, 100", "HALT"])
    vm, root = make_vm(prog, {}, {})
    cid = vm.branch(wid=root)
    vm.run_world(cid, ["balanced"])
    core = vm.worlds[cid].sim.core
    checks.append(("R0_zero", core.regs[0] == 0))
    checks.append(("alu_chain", core.regs[3] == 12))
    checks.append(("store_load_forwarding_path", core.regs[4] == 12 and core.mem[100] == 12))
    # branch taken + predictor learns
    prog2 = assemble(["ADDI R1, R0, 1", "ADDI R2, R0, 1", "BEQ R1, R2, 2",
                      "ADDI R3, R0, 111", "ADDI R3, R0, 42", "HALT"])
    vm2, root2 = make_vm(prog2, {}, {})
    cid2 = vm2.branch(wid=root2)
    vm2.run_world(cid2, ["balanced"])
    checks.append(("branch_taken", vm2.worlds[cid2].sim.core.regs[3] == 42))
    # thermal safety: turbo requested at the limit must be governed
    from vmarch.workloads import stream_like
    p, m, r, _ = stream_like(0, 30)
    vm3, root3 = make_vm(p, m, r)
    w3 = vm3.worlds[root3]
    w3.sim.core.temperatures = [41.9] * 4
    reasons = w3.sim.perturb("latency")
    checks.append(("governor_fires", bool(reasons) and
                   w3.sim.core.control.power_mode.value != "turbo"))
    # determinism: fork parity
    for s in seeds[:3]:
        for fam in ("stream", "branch"):
            p, m, r, _ = FAMILIES[fam](s)
            va, ra = make_vm(p, m, r)
            a = va.branch(wid=ra)
            resa = va.run_world(a, ["balanced", "streaming"])
            vb, rb = make_vm(p, m, r)
            b = vb.branch(wid=rb)
            resb = vb.run_world(b, ["balanced", "streaming"])
            checks.append((f"determinism_{fam}_{s}",
                           resa["objective"] == resb["objective"]))
    failed = [n for n, ok in checks if not ok]
    return {"passed": len(checks) - len(failed), "total": len(checks),
            "failed": failed, "ok": not failed}


def ood_gaps(train_examples: list[tuple], test_by_domain: dict[str, list[tuple]],
             ensemble) -> dict:
    from vmarch.surrogate import evaluate_parity
    out = {}
    for domain, examples in test_by_domain.items():
        out[domain] = evaluate_parity(ensemble, examples)
    return out


def divergence_horizon(ensemble, start_x: list, plan: list[str], steps: int = 8) -> dict:
    from vmarch.surrogate import evaluate_divergence
    return evaluate_divergence(ensemble, start_x, plan, steps)


def validator_precision(vm, wid: int, ensemble, candidates: list[list[str]],
                        interval: int = 16) -> dict:
    from vmarch.agents import validator_queries
    order = validator_queries(vm, wid, ensemble, candidates)["query_order"]
    feats = vm.worlds[wid].sim.features()
    scored = []
    for q in order:
        cid = vm.branch(label="valcheck", wid=wid)
        actual = vm.run_world(cid, q["plan"], interval=interval)["objective"]
        pred = sum(ensemble.predict_cost(feats, b) for b in q["plan"])
        scored.append({"plan": q["plan"], "rank_score": q["score"],
                       "error": abs(pred - actual)})
    top_err = np.mean([s["error"] for s in scored[:2]]) if len(scored) >= 2 else 0.0
    bot_err = np.mean([s["error"] for s in scored[-2:]]) if len(scored) >= 2 else 0.0
    return {"ranked": scored,
            "top2_mean_error": float(top_err), "bottom2_mean_error": float(bot_err),
            "precise": bool(top_err >= bot_err)}


def held_out(make_vm, family: str, train_seeds: range, test_seeds: range,
             train_kwargs: dict | None = None, budget: int = 4,
             horizon: int = 4, interval: int = 16, cfg=None,
             ood_families: list[str] | None = None,
             train_reps: int = 1) -> dict:
    """End-to-end Phase-1-style protocol on one family.

    T6: ood_families cross-evaluates the trained ensemble on other families
    (train-on-A/test-on-B context transfer). T7: train_reps repeats training
    with rotated seeds and reports parity/skill spread across reps.
    """
    from vmarch.search import killer_benchmark
    from vmarch.surrogate import (evaluate_hallucination, evaluate_parity,
                                  train_ensemble)
    from vmarch.virtual import VirtualMicroarchitecture
    check_disjoint((train_seeds.start, train_seeds.stop),
                   (test_seeds.start, test_seeds.stop))
    train_kwargs = train_kwargs or {}
    # train data from oracle trajectories (the "reality" leg)
    examples = []
    for s in train_seeds:
        p, m, r, _ = FAMILIES[family](s)
        probe_vm, probe_root = make_vm(p, m, r)
        examples += probe_vm.collect_dataset(
            probe_root, intervals=12,
            bundles=["balanced", "streaming", "irregular", "control", "latency"],
            interval=interval)

    def _mk(seed: int):
        p, m, r, _ = FAMILIES[family](seed)
        vm = VirtualMicroarchitecture(p, cfg=cfg, mem_init=m, reg_init=r)
        return vm, 0

    # parity on held-out seeds
    test_examples = []
    for s in test_seeds:
        p, m, r, _ = FAMILIES[family](s)
        probe_vm, probe_root = make_vm(p, m, r)
        test_examples += probe_vm.collect_dataset(
            probe_root, intervals=6, bundles=["balanced", "streaming", "control"],
            interval=interval)
    rep_parity, rep_skill, ensemble, train_info = [], [], None, {}
    for rep in range(max(1, train_reps)):
        seeds = tuple(1000 + rep * 97 + j * 13 for j in range(3))
        ensemble, train_info = train_ensemble(examples, seeds=seeds, **train_kwargs)
        rep_parity.append(evaluate_parity(ensemble, test_examples))
        rep_skill.append(skill_scores(test_examples, ensemble))
    parity = rep_parity[0]
    skill = rep_skill[0]
    spread = {}
    if len(rep_parity) > 1:
        spread = {"mae_std": float(np.std([p["mae"] for p in rep_parity])),
                  "mae_min": float(min(p["mae"] for p in rep_parity)),
                  "mae_max": float(max(p["mae"] for p in rep_parity)),
                  "skill_mean_min": float(min(s["skill_vs_mean"] for s in rep_skill)),
                  "skill_mean_max": float(max(s["skill_vs_mean"] for s in rep_skill)),
                  "r2_min": float(min(s["r2"] for s in rep_skill)),
                  "r2_max": float(max(s["r2"] for s in rep_skill)),
                  "reps": len(rep_parity)}
    hall = evaluate_hallucination(ensemble, test_examples)
    calib = calibration(test_examples, ensemble)
    kb = killer_benchmark(_mk, ensemble, list(test_seeds)[:8],
                          horizon=horizon, interval=interval, budget=budget)
    ood = {}
    for of in (ood_families or []):
        ood_ex = []
        for s in test_seeds:
            p, m, r, _ = FAMILIES[of](s)
            probe_vm, probe_root = make_vm(p, m, r)
            ood_ex += probe_vm.collect_dataset(
                probe_root, intervals=6,
                bundles=["balanced", "streaming", "control"], interval=interval)
        ood[of] = {"parity": evaluate_parity(ensemble, ood_ex),
                   "skill": skill_scores(ood_ex, ensemble)}
    return {"family": family, "train": train_info, "parity": parity,
            "skill": skill, "parity_spread": spread,
            "hallucination": hall, "calibration": calib,
            "killer_benchmark": kb, "ood": ood,
            "primary_metric": PRIMARY_METRIC,
            "maturity": maturity_gate(parity, hall, calib, kb, skill)}


def maturity_gate(parity: dict, hall: dict, calib: dict, kb: dict,
                  skill: dict | None = None) -> dict:
    """T4: L1/L2 are skill gates (fail constants by construction).
    skill=None keeps the call valid but yields False (no evidence)."""
    if skill is None:
        l1 = l2 = False
    else:
        l1 = skill["skill_vs_mean"] >= 0.10 and skill["r2"] >= 0.10
        l2 = skill["skill_vs_last"] >= 0.05
    sb = kb.get("delta_surrogate_beam", {})
    l3 = bool(sb and sb.get("mean", 0) is not None and sb["mean"] > 0)
    l4 = bool(hall["rate"] < 0.25 and calib.get("monotone", False))
    l5 = bool(l4 and sb and (sb.get("mean", 0) - sb.get("ci95", 10**9)) > 0)
    return {"L1_exemplar": l1, "L2_interpolative": l2, "L3_counterfactual": l3,
            "L4_mechanistic": l4, "L5_scientific": l5}


def save_artifact(path: str, payload: dict, extra: dict | None = None) -> dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"bench_protocol": BENCH_PROTOCOL_VERSION, "vmarch_version": VMARCH_VERSION,
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           **(extra or {}), **payload}
    p.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return doc
