"""planning.py — MCTS + robust search over the learned kernel dynamics.

context.txt core (Virtual Chess section):
- MuZero-style: plan with the learned model, consult the exact oracle only
  to validate ("how much useful search can the learned environment perform
  before consulting the exact oracle?").
- Generalized objective: don't just find the best continuation — find the
  configuration that collapses the future risk space. Here the "opponent" is
  the workload: an adversarial demand regime that punishes fragile tunings.
  robust_search() maximizes the WORST-case reward over demand scenarios
  (opponent-constrained configuration selection).
- Risk-aware planning: every virtual step pays an uncertainty penalty, so
  the planner avoids regions where the surrogate is unreliable ("when does
  model uncertainty become strategically dangerous?").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from learned_kernel.policy.schemas import (
    KernelIntermediateRepresentation,
    PolicyAction,
    SchedulerAction,
)

from .search import CANDIDATE_LATENCIES, _reward_of


def _action_for(lat: int) -> PolicyAction:
    return PolicyAction(policy_id="plan",
                        scheduler=SchedulerAction(target_latency_us=int(lat)))


@dataclass
class MCTSResult:
    best_latencies: List[int]
    value: float
    visits: Dict[int, int]
    n_sims: int


class _Node:
    __slots__ = ("kir", "depth", "visits", "total", "children", "untried")

    def __init__(self, kir: KernelIntermediateRepresentation, depth: int,
                 arms: Tuple[int, ...]):
        self.kir = kir
        self.depth = depth
        self.visits = 0
        self.total = 0.0
        self.children: Dict[int, "_Node"] = {}
        self.untried: List[int] = list(arms)

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0


def mcts_search(ensemble, kir0: KernelIntermediateRepresentation,
                horizon: int = 4, sims: int = 200, c: float = 1.4,
                uncertainty_penalty: float = 0.5,
                arms: Tuple[int, ...] = CANDIDATE_LATENCIES,
                dt_s: float = 0.5, seed: int = 0,
                discount: float = 0.95) -> MCTSResult:
    """UCT search over virtual sysctl trajectories. Zero oracle calls."""
    rng = np.random.default_rng(seed)
    root = _Node(kir0.model_copy(deep=True), 0, arms)

    def step_value(kir, lat: int):
        act = _action_for(lat)
        nxt = ensemble.predict_kir(kir, act, dt_s=dt_s)
        r = _reward_of(kir, nxt)
        u = ensemble.uncertainty(kir, act)
        return nxt, r - uncertainty_penalty * u

    for _ in range(sims):
        node = root
        path: List[Tuple[_Node, int]] = []
        # selection
        while not node.untried and node.children and node.depth < horizon:
            total = node.visits
            log_total = np.log(max(total, 1))
            best_lat, best_score = -1, float("-inf")
            for lat, child in node.children.items():
                exploit = child.mean
                explore = c * np.sqrt(log_total / max(child.visits, 1))
                s = exploit + explore
                if s > best_score:
                    best_score, best_lat = s, lat
            node = node.children[best_lat]
            path.append((node, best_lat))
        # expansion
        if node.depth < horizon and node.untried:
            lat = node.untried[int(rng.integers(len(node.untried)))]
            node.untried.remove(lat)
            nxt, r = step_value(node.kir, lat)
            child = _Node(nxt, node.depth + 1, arms)
            # rollout with random policy to horizon
            ret = r
            kir, disc = nxt, discount
            for _ in range(node.depth + 1, horizon):
                rl = int(arms[int(rng.integers(len(arms)))])
                kir, rr = step_value(kir, rl)
                ret += disc * rr
                disc *= discount
            child.visits, child.total = 1, ret
            node.children[lat] = child
            node.visits += 1
            node.total += ret
            for anc, _ in path:
                anc.visits += 1
                anc.total += ret
        else:
            # terminal node: backprop its mean
            for anc, _ in path:
                anc.visits += 1
                anc.total += node.mean

    visits = {lat: ch.visits for lat, ch in root.children.items()}
    # greedy descent along most-visited children for the full sequence
    seq: List[int] = []
    node = root
    while node.children and len(seq) < horizon:
        lat = max(node.children, key=lambda l: node.children[l].visits)
        seq.append(lat)
        node = node.children[lat]
    return MCTSResult(best_latencies=seq, value=root.mean, visits=visits,
                      n_sims=sims)


def rollout_sequence_reward(ensemble, kir0: KernelIntermediateRepresentation,
                            latencies: List[int],
                            uncertainty_penalty: float = 0.0,
                            dt_s: float = 0.5) -> Tuple[float, KernelIntermediateRepresentation]:
    kir = kir0.model_copy(deep=True)
    total = 0.0
    for lat in latencies:
        act = _action_for(lat)
        nxt = ensemble.predict_kir(kir, act, dt_s=dt_s)
        total += _reward_of(kir, nxt) - uncertainty_penalty * ensemble.uncertainty(kir, act)
        kir = nxt
    return total, kir


def robust_search(ensemble, scenario_kirs: List[KernelIntermediateRepresentation],
                  candidates: List[List[int]],
                  uncertainty_penalty: float = 0.0,
                  dt_s: float = 0.5) -> Dict:
    """Maximin over demand scenarios: workload as adversary.

    For each candidate latency sequence, score it from EVERY scenario start
    state (different workload regimes) and keep the worst case. Select the
    candidate with the best worst-case — the opponent-constrained config.
    """
    rows = []
    for seq in candidates:
        worst, per = float("inf"), []
        for kir0 in scenario_kirs:
            r, _ = rollout_sequence_reward(ensemble, kir0, seq,
                                           uncertainty_penalty, dt_s)
            per.append(r)
            worst = min(worst, r)
        rows.append({"sequence": seq, "worst_case": worst,
                     "mean": float(np.mean(per)), "per_scenario": per})
    rows.sort(key=lambda d: d["worst_case"], reverse=True)
    return {"ranking": rows, "selected": rows[0] if rows else None,
            "n_scenarios": len(scenario_kirs)}
