"""Counterfactual search: beam + MCTS + killer benchmark.

Implements the Virtual Chess program (context.txt 1158–1403) for computation:
- beam search and MCTS over control-bundle plans, scored by the surrogate,
  validated *only* on the exact oracle under a finite budget B;
- expert online heuristic (phase-reactive, DEA lineage);
- killer_benchmark: fixed / best-fixed / expert / exact-search(B) /
  learned-alone / surrogate-MCTS(B) with paired deltas + 95% CIs.
  Metric: objective achieved per oracle interaction.

Budget unit (auditable): 1 oracle call = 1 full-plan validation run
(`run_world`). Surrogate calls are unlimited and counted separately.
"""
from __future__ import annotations

import math
import random

from vmarch.config import BUNDLE_NAMES

# Feature index coupling (vmarch.core.FEATURE_NAMES): mem_frac=14, branch=16,
# vec=17, sequentiality=11, rob_occ=1, headroom=12.
_I_MEM, _I_BR, _I_VEC, _I_SEQ, _I_ROB = 14, 16, 17, 11, 1


def expert_bundle(features: list[float], headroom: float) -> str:
    if headroom < 3.0:
        return "thermal_recovery"
    if features[_I_MEM] > 0.55 and features[_I_SEQ] > 0.45:
        return "streaming"
    if features[_I_VEC] > 0.30:
        return "latency"
    if features[_I_MEM] > 0.40 and features[_I_SEQ] < 0.30:
        return "irregular"
    if features[_I_BR] > 0.35:
        return "control"
    if features[_I_ROB] > 0.70:
        return "latency"
    return "balanced"


def run_expert(vm, wid: int, interval: int = 8) -> dict:
    """Online phase-reactive policy, run to HALT (full trajectory, like plans).

    NOTE (audit fix): this previously stepped a bounded number of cycles and
    returned a partial-run objective, which made expert-vs-baseline deltas
    meaningless. It now retires the whole program; cost = 1 oracle run.
    """
    cid = vm.branch(label="expert", wid=wid)
    w = vm.worlds[cid]
    decisions = 0
    while not w.sim.core.halted and w.sim.core.cycle < w.sim.core.cfg.max_cycles:
        if w.sim.core.cycle % interval == 0:
            w.sim.perturb(expert_bundle(w.sim.features(), w.sim.core.headroom()))
            decisions += 1
        w.sim.step(1)
    res = w.sim.result()
    res["world"] = cid
    res["decisions"] = decisions
    return res


def _validate(vm, root_wid: int, plans: list[list[str]], interval: int,
              budget: int) -> tuple[list[dict], int]:
    """Validate candidate plans on the oracle. Returns (results, oracle_spent)."""
    out, spent = [], 0
    for plan in plans[:budget]:
        cid = vm.branch(label="validate", wid=root_wid)
        res = vm.run_world(cid, plan, interval=interval)
        out.append({"objective": res["objective"], "plan": plan, "result": res})
        spent += 1
    out.sort(key=lambda d: d["objective"])
    return out, spent


def beam_search_plan(vm, root_wid: int, surrogate, horizon: int = 6,
                     interval: int = 16, beam: int = 4,
                     oracle_budget: int = 4) -> dict:
    names = list(surrogate.bundles)
    beams: list[tuple[float, list[str]]] = [(0.0, [])]
    surrogate_calls = 0
    probe_steps = 0  # oracle cycles spent on feature probes (not validations)
    for _ in range(horizon):
        cands = []
        for cost, plan in beams:
            probe = vm.branch(label="probe", wid=root_wid)
            pw = vm.worlds[probe]
            for p in plan:
                pw.sim.perturb(p)
                pw.sim.step(interval)
                probe_steps += interval
            feats = pw.sim.features()
            for b in names:
                cands.append((cost + surrogate.predict_cost(feats, b), plan + [b]))
                surrogate_calls += 1
        cands.sort(key=lambda t: t[0])
        beams = cands[:beam]
    plans = [p for _, p in beams]
    validated, spent = _validate(vm, root_wid, plans, interval, oracle_budget)
    return {"best": validated[0] if validated else None, "validated": validated,
            "beam": [{"predicted": c, "plan": p} for c, p in beams],
            "surrogate_calls": surrogate_calls, "oracle_calls": spent,
            "probe_steps": probe_steps}


class _MCTSNode:
    __slots__ = ("wid", "plan", "visits", "total", "children")

    def __init__(self, wid: int, plan: list[str]) -> None:
        self.wid = wid
        self.plan: list[str] = plan
        self.visits = 0
        self.total = 0.0
        self.children: dict[str, "_MCTSNode"] = {}

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else float("inf")


def _surrogate_rollout_value(surrogate, feats, depth: int) -> float:
    names = list(surrogate.bundles)
    total, x = 0.0, feats
    for _ in range(depth):
        costs = [(surrogate.predict_cost(x, b), b) for b in names]
        c, b = min(costs, key=lambda t: t[0])
        total += c
        try:
            x = surrogate.members[0].predict_next(x, b).tolist()
        except Exception:
            break
    return total


def mcts_plan(vm, root_wid: int, surrogate, horizon: int = 4, interval: int = 16,
               iterations: int = 40, c: float = 2.0,
               oracle_budget: int = 4, seed: int = 0) -> dict:
    names = list(surrogate.bundles)
    rng = random.Random(seed)
    root = _MCTSNode(root_wid, [])
    surrogate_calls = 0
    probe_steps = 0

    def uct_score(child: _MCTSNode, parent_visits: int) -> float:
        if child.visits == 0:
            return float("-inf")  # unvisited first (we minimize cost)
        return child.mean - c * math.sqrt(math.log(max(parent_visits, 1)) / child.visits)

    for _ in range(iterations):
        node = root
        path = [root]
        # select
        while len(node.children) == len(names) and len(node.plan) < horizon:
            node = min(node.children.values(),
                       key=lambda ch: uct_score(ch, node.visits))
            path.append(node)
        # expand (seeded order so rival hypotheses get fair coverage)
        if len(node.plan) < horizon:
            tried = set(node.children)
            order = [b for b in names if b not in tried]
            rng.shuffle(order)
            for b in order:
                cid = vm.branch(label="mcts", wid=node.wid)
                cw = vm.worlds[cid]
                cw.sim.perturb(b)
                cw.sim.step(interval)
                probe_steps += interval
                feats = cw.sim.features()
                step_cost = surrogate.predict_cost(feats, b)
                surrogate_calls += 1
                leaf = _MCTSNode(cid, node.plan + [b])
                node.children[b] = leaf
                remainder = _surrogate_rollout_value(
                    surrogate, feats, horizon - len(leaf.plan))
                surrogate_calls += max(0, horizon - len(leaf.plan)) * len(names)
                value = step_cost + remainder
                # backprop along the full selection path (root ... parent, leaf)
                for n in path:
                    n.visits += 1
                    n.total += value
                leaf.visits += 1
                leaf.total += value
                break
            continue
        # terminal node: re-evaluate
        w = vm.worlds[node.wid]
        v = _surrogate_rollout_value(surrogate, w.sim.features(), 0)
        for n in path:
            n.visits += 1
            n.total += v
    # gather candidate plans: prefer full-horizon leaves (comparable predicted
    # totals); shallow leaves only if the tree never reached full depth.
    full: list[tuple[float, list[str]]] = []
    shallow: list[tuple[float, list[str]]] = []

    def collect(n: _MCTSNode) -> None:
        if not n.children or len(n.plan) >= horizon:
            if n.plan:
                (full if len(n.plan) >= horizon else shallow).append((n.mean, n.plan))
            return
        for ch in n.children.values():
            collect(ch)

    collect(root)
    leaves = full or shallow
    leaves.sort(key=lambda t: t[0])
    plans = [p for _, p in leaves]
    if not plans:  # degenerate: fall back to greedy single plan
        feats = vm.worlds[root_wid].sim.features()
        plans = [[min(names, key=lambda b: surrogate.predict_cost(feats, b))]]
    validated, spent = _validate(vm, root_wid, plans, interval, oracle_budget)
    return {"best": validated[0] if validated else None, "validated": validated,
            "leaves": [{"predicted": c0, "plan": p} for c0, p in leaves[:8]],
            "surrogate_calls": surrogate_calls, "oracle_calls": spent,
            "probe_steps": probe_steps, "iterations": iterations}


def random_plans(seed: int, n: int, horizon: int,
                 names: list[str] | None = None) -> list[list[str]]:
    rng = random.Random(seed)
    names = names or BUNDLE_NAMES
    return [[rng.choice(names) for _ in range(horizon)] for _ in range(n)]


def _paired_ci(deltas: list[float]) -> tuple[float, float]:
    import statistics
    n = len(deltas)
    if n < 2:
        return (deltas[0] if deltas else 0.0), 0.0
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    return m, 1.96 * sd / math.sqrt(n)


def killer_benchmark(make_vm, surrogate, seeds: list[int], horizon: int = 4,
                     interval: int = 16, budget: int = 4) -> dict:
    """Full killer-benchmark protocol. make_vm(seed) -> (vm, root_wid).

    Contestants: balanced | best_fixed (retrospective) | expert |
    exact-search(B random plans) | learned-alone (surrogate argmin, 1 run) |
    surrogate-beam(B) | surrogate-mcts(B). Paired deltas vs balanced + 95% CI.
    """
    per_seed, oracle_use = [], {}
    for s in seeds:
        vm, root = make_vm(s)
        r_bal = vm.run_world(vm.branch(label="c-balanced", wid=root),
                             ["balanced"] * horizon, interval=interval)
        fixed = {}
        for b in BUNDLE_NAMES:
            cid = vm.branch(label=f"c-{b}", wid=root)
            fixed[b] = vm.run_world(cid, [b] * horizon, interval=interval)["objective"]
        best_fixed = min(fixed.values())
        r_exp = run_expert(vm, root, interval=interval)
        exact_cands = random_plans(1000 + s, budget, horizon)
        exact_val, exact_spent = _validate(vm, root, exact_cands, interval, budget)
        beam = beam_search_plan(vm, root, surrogate, horizon, interval,
                                beam=4, oracle_budget=budget)
        mcts = mcts_plan(vm, root, surrogate, horizon, interval,
                         iterations=30, oracle_budget=budget, seed=5000 + s)
        names = list(surrogate.bundles)
        feats = vm.worlds[root].sim.features()
        argmin_plan = [min(names, key=lambda b: surrogate.predict_cost(feats, b))] * horizon
        learned_val, learned_spent = _validate(vm, root, [argmin_plan], interval, 1)
        per_seed.append({
            "seed": s, "balanced": r_bal["objective"], "fixed": fixed,
            "best_fixed": best_fixed, "expert": r_exp["objective"],
            "exact_search": exact_val[0]["objective"] if exact_val else None,
            "learned_alone": learned_val[0]["objective"] if learned_val else None,
            "surrogate_beam": beam["best"]["objective"] if beam["best"] else None,
            "surrogate_mcts": mcts["best"]["objective"] if mcts["best"] else None,
            "beam_plan": beam["best"]["plan"] if beam["best"] else None,
            "mcts_plan": mcts["best"]["plan"] if mcts["best"] else None,
        })
        oracle_use = {"balanced": 1, "fixed_each": 1, "expert": 1,
                      "exact_search": exact_spent, "learned_alone": learned_spent,
                      "surrogate_beam": beam["oracle_calls"],
                      "surrogate_mcts": mcts["oracle_calls"],
                      "surrogate_calls_beam": beam["surrogate_calls"],
                      "surrogate_calls_mcts": mcts["surrogate_calls"],
                      # honesty accounting: oracle cycles spent on feature
                      # probes (world forking views), separate from the
                      # validation budget B they do not consume
                      "probe_steps_beam": beam.get("probe_steps", 0),
                      "probe_steps_mcts": mcts.get("probe_steps", 0)}
    out = {"per_seed": per_seed, "oracle_use_per_seed": oracle_use,
           "budget": budget, "horizon": horizon, "interval": interval}
    for key in ("best_fixed", "expert", "exact_search", "learned_alone",
                "surrogate_beam", "surrogate_mcts"):
        deltas = [r["balanced"] - r[key] for r in per_seed if r[key] is not None]
        m, ci = _paired_ci(deltas)
        out[f"delta_{key}"] = {"mean": m, "ci95": ci, "n": len(deltas)}
    return out
