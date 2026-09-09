"""Agent harness over the shared virtual microarchitecture.

Roster from context.txt (cell §8 + chess swarm): Researcher (hypotheses),
Perturber (chooses interventions), BottleneckAnalyst (interprets response),
Adversary (finds refutations), Designer (searches combinations/configs),
Validator (selects physical — here oracle — validation), SurrogateCritic
(identifies where the learned model may be wrong). All operate on the same
virtual environment; synthesize_report() merges findings into a lab report.

Agents are deterministic planners over exact counters, not LLM theatre: every
claim cites measured numbers and the world lineage that produced them.
"""
from __future__ import annotations


def bottleneck_analyst(vm, wid: int) -> dict:
    m = vm.measure(wid)
    s = m["stalls"]
    cands = {
        "frontend_icache": s["icache_stall"] + s["fetch_starve"] * 0.5,
        "rob_capacity_ilp": s["rob_full"],
        "dependence_raw": s["dep_stall"],
        "memory_mshr": s["mshr_stall"] + m.get("l1d_miss_rate", 0) * m["cycles"] * 0.1,
        "branch_squash": s["squashes"] * 4 + s["flush_bubbles"],
        "rs_capacity": s["rs_full"] + s["lsq_full"],
    }
    top = max(cands, key=lambda k: cands[k])
    advice = {
        "frontend_icache": "streaming (prefetch) or wider fetch; check I-footprint",
        "rob_capacity_ilp": "latency (turbo issue width) to expose ILP",
        "dependence_raw": "no control bundle fixes RAW chains; consider unrolling at compile time",
        "memory_mshr": "streaming for sequential, irregular (bypass+no-prefetch) for pointer",
        "branch_squash": "control is wrong here only if predictor weak; else irregular/spec-off for chaos",
        "rs_capacity": "latency to drain queues faster",
    }
    return {"agent": "BottleneckAnalyst", "world": wid, "scores": cands,
            "bottleneck": top, "recommendation": advice[top],
            "evidence": {"ipc": m["ipc"], "mpki": m["mpki"],
                         "l1d_miss": m["l1d_miss_rate"], "squashes": m["squashes"]}}


def perturber(vm, wid: int, surrogate, horizon: int = 4, interval: int = 16) -> dict:
    from vmarch.search import beam_search_plan
    out = beam_search_plan(vm, wid, surrogate, horizon, interval,
                           beam=4, oracle_budget=3)
    return {"agent": "Perturber", "world": wid,
            "plan": out["best"]["plan"] if out["best"] else None,
            "objective": out["best"]["objective"] if out["best"] else None,
            "oracle_calls": out["oracle_calls"],
            "surrogate_calls": out["surrogate_calls"]}


def researcher(vm, wid: int, hypothesis: str, plan_a: list, plan_b: list,
               interval: int = 16) -> dict:
    ca = vm.branch(label="hypo-A", wid=wid)
    cb = vm.branch(label="hypo-B", wid=wid)
    ra = vm.run_world(ca, plan_a, interval=interval)
    rb = vm.run_world(cb, plan_b, interval=interval)
    delta = rb["objective"] - ra["objective"]
    return {"agent": "Researcher", "hypothesis": hypothesis,
            "plan_a": plan_a, "plan_b": plan_b,
            "objective_a": ra["objective"], "objective_b": rb["objective"],
            "delta_b_minus_a": delta,
            "verdict": "confirmed" if delta < 0 else "refuted",
            "worlds": [ca, cb]}


def adversary(vm, wid: int, surrogate, candidates: list[list[str]],
              interval: int = 16, budget: int = 4) -> dict:
    """Spend oracle budget where surrogate predictions look most suspicious:
    validate candidates, report max |predicted - actual| as refutations."""
    scored = []
    for plan in candidates[:budget]:
        feats = vm.worlds[wid].sim.features()
        pred = sum(surrogate.predict_cost(feats, b) for b in plan)
        cid = vm.branch(label="adversarial", wid=wid)
        actual = vm.run_world(cid, plan, interval=interval)["objective"]
        scored.append({"plan": plan, "predicted": pred, "actual": actual,
                       "gap": abs(pred - actual), "world": cid})
    scored.sort(key=lambda d: d["gap"], reverse=True)
    return {"agent": "Adversary", "world": wid, "refutations": scored,
            "worst_gap": scored[0]["gap"] if scored else 0.0}


def designer(make_vm_prog_cfg, workload_seed: int, configs: list) -> dict:
    """Constrained design-space comparison: same program, rival tape-outs."""
    rows = []
    for cfg in configs:
        vm, root = make_vm_prog_cfg(workload_seed, cfg)
        cid = vm.branch(label=f"design-{cfg.name}", wid=root)
        res = vm.run_world(cid, ["balanced"] * 6, interval=16)
        rows.append({"config": cfg.name, "digest": cfg.digest(),
                     "cycles": res["cycles"], "energy": res["energy"],
                     "objective": res["objective"], "ipc": res["ipc"]})
    rows.sort(key=lambda r: r["objective"])
    return {"agent": "Designer", "rows": rows, "recommended": rows[0]["config"]}


def validator_queries(vm, wid: int, surrogate, candidates: list[list[str]]) -> dict:
    """Rank candidate oracle queries by information-gain proxy:
    ensemble disagreement x predicted-cost spread."""
    feats = vm.worlds[wid].sim.features()
    ranked = []
    for plan in candidates:
        dis = sum(surrogate.disagreement(feats, b) for b in plan)
        costs = [surrogate.predict_cost(feats, b) for b in plan]
        spread = max(costs) - min(costs) if costs else 0.0
        ranked.append({"plan": plan, "score": dis * (1.0 + spread),
                       "disagreement": dis, "spread": spread})
    ranked.sort(key=lambda d: d["score"], reverse=True)
    return {"agent": "Validator", "world": wid, "query_order": ranked}


def surrogate_critic(surrogate, parity: dict, divergence: dict,
                     hallucination: dict) -> dict:
    deployable = (hallucination["rate"] < 0.25 and parity["mae"] < 5.0
                  and divergence["final_spread"] < 20.0)
    return {"agent": "SurrogateCritic", "parity_mae": parity["mae"],
            "hallucination_rate": hallucination["rate"],
            "divergence_spread": divergence["final_spread"],
            "verdict": "deployable with budget" if deployable else "needs data",
            "prescription": ("collect max-disagreement intervals" if not deployable
                             else "proceed to killer benchmark")}


def synthesize_report(title: str, findings: list[dict]) -> str:
    lines = [f"# Lab report: {title}", ""]
    for f in findings:
        lines.append(f"## {f.get('agent', 'finding')}")
        for k, v in f.items():
            if k == "agent":
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)
