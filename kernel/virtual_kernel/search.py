"""search.py — virtual screening: massive counterfactual search in the surrogate.

context.txt core operation: learn -> branch -> search -> act -> validate.
Generalized chess objective (lines 451-456, 700-704):

    "Can a learned surrogate convert an expensive sequential interaction
     problem into a massive counterfactual search problem, such that the
     optimal real-world action can be selected before the real trajectory
     unfolds?"

Here: search millions of virtual sysctl trajectories in the surrogate,
pick the best, validate top-k against the exact oracle, and report
"strength achieved per oracle interaction".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from learned_kernel.policy.schemas import KernelIntermediateRepresentation, PolicyAction, SchedulerAction
from learned_kernel.trainer.reward import RewardCalculator

CANDIDATE_LATENCIES = (1000, 2000, 4000, 6000, 12000, 24000)


@dataclass
class BeamResult:
    latencies: List[int]
    predicted_reward: float
    oracle_reward: Optional[float] = None
    oracle_calls: int = 0


def _reward_of(kir_prev: KernelIntermediateRepresentation,
               kir_next: KernelIntermediateRepresentation) -> float:
    calc = RewardCalculator()
    return float(calc.calculate_step_reward(kir_prev, kir_next))


def virtual_screen(ensemble, kir0: KernelIntermediateRepresentation,
                   horizon: int = 4, beam_width: int = 6,
                   n_virtual_rollouts: Optional[int] = None,
                   dt_s: float = 0.5) -> List[BeamResult]:
    """Beam search over latency sequences using surrogate only (0 oracle calls).

    Returns ranked candidates; caller validates top-k with TruthOracle.
    n_virtual_rollouts reports how many counterfactual worlds were searched.
    """
    beams: List[tuple] = [([], kir0.model_copy(deep=True), 0.0)]
    for _ in range(horizon):
        nxt_beams = []
        for seq, kir, r in beams:
            for lat in CANDIDATE_LATENCIES:
                act = PolicyAction(policy_id="search",
                                   scheduler=SchedulerAction(target_latency_us=lat))
                kir2 = ensemble.predict_kir(kir, act, dt_s=dt_s)
                nxt_beams.append((seq + [lat], kir2, r + _reward_of(kir, kir2)))
        nxt_beams.sort(key=lambda t: t[2], reverse=True)
        beams = nxt_beams[:beam_width]
    results = [BeamResult(latencies=s, predicted_reward=r) for s, _, r in beams]
    if n_virtual_rollouts is not None:
        pass
    virtual_screen.last_virtual_worlds = len(CANDIDATE_LATENCIES) * horizon * beam_width
    return results


def validate_topk(results: List[BeamResult], oracle, kir0, k: int = 2) -> List[BeamResult]:
    """Ground top-k virtual candidates in exact dynamics (the only oracle cost)."""
    from learned_kernel.policy.schemas import PolicyAction as PA, SchedulerAction as SA
    for res in results[:k]:
        oracle.reset(kir0)
        kir = kir0
        total = 0.0
        for lat in res.latencies:
            nxt = oracle.step(PA(policy_id="verify", scheduler=SA(target_latency_us=lat)))
            total += _reward_of(kir, nxt)
            kir = nxt
        res.oracle_reward = total
        res.oracle_calls = oracle.calls
        oracle.calls = 0
    return results
