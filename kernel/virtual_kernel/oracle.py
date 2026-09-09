"""oracle.py — exact-dynamics truth oracle + surrogate scoring.

Analogous to Virtual Chess's exact-rules oracle (context.txt lines 628-650):
every virtual claim can be checked against exact dynamics. Here the exact
dynamics are KernelSimulator's analytic scheduler model; the surrogate is
the learned EnsembleDynamics. Metrics: single-step MSE, open-loop rollout
divergence ("how far can it roll out before divergence?"), and the
"strength per oracle interaction" efficiency ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from learned_kernel.policy.core import HeuristicLatencyPolicy
from learned_kernel.policy.schemas import (
    KernelIntermediateRepresentation,
    PolicyAction,
    SchedulerAction,
)
from learned_kernel.simulator.env import KernelSimulator
from learned_kernel.trainer.reward import RewardCalculator

from .dynamics import encode_kir, kir_observables


@dataclass
class OracleReport:
    single_step_mse: float
    rollout_mae_curve: List[float]
    rollout_horizon: int
    divergence_step: Optional[int]  # first step where MAE > threshold
    n_oracle_calls: int

    def summary(self) -> dict:
        return {"single_step_mse": self.single_step_mse,
                "final_mae": self.rollout_mae_curve[-1] if self.rollout_mae_curve else 0.0,
                "divergence_step": self.divergence_step,
                "n_oracle_calls": self.n_oracle_calls}


class TruthOracle:
    """Thin wrapper making KernelSimulator explicit as the validation oracle.

    Paired-measurement discipline (see THEORY.md "oracle discipline"): the
    hidden workload demand is re-seeded deterministically from
    (oracle seed, start state) on every reset. Consequences:

    - Repeated rollouts from the SAME start state face the SAME demand
      trajectory, so differences between candidate sequences are purely
      action-driven (paired design — the workload luck cancels).
    - Results are order-independent: benchmarking sequence A before B gives
      the same numbers as B before A. (KernelSimulator.reset restores
      observables but leaves the workload RNG running, which previously
      contaminated every sequential comparison with rollout history.)
    - learned_kernel/ itself is untouched; only this harness's instance is
      re-seeded, which the class explicitly supports via its constructor.
    """

    def __init__(self, seed: int = 1234, n_cpus: int = 2, dt_s: float = 0.5):
        self.base_seed = int(seed)
        self.sim = KernelSimulator(seed=seed, n_cpus=n_cpus, dt_s=dt_s)
        self.n_cpus = n_cpus
        self.dt_s = dt_s
        self.calls = 0

    @staticmethod
    def _workload_seed(base_seed: int,
                       kir: Optional[KernelIntermediateRepresentation]) -> int:
        import random
        if kir is None:
            return base_seed & 0xFFFFFFFF
        state_key = (int(kir.timestamp * 1000)
                     + 9176 * kir.scheduler.total_context_switches)
        return (int(base_seed) + 1009 * state_key) & 0xFFFFFFFF

    def reset(self, kir: Optional[KernelIntermediateRepresentation] = None) -> None:
        import random
        from learned_kernel.simulator.env import WorkloadProfile
        self.sim.reset(from_state=kir)
        self.sim.workload = WorkloadProfile(
            rng=random.Random(self._workload_seed(self.base_seed, kir)),
            dt_s=self.dt_s)
        self.calls = 0

    def step(self, action: Optional[PolicyAction]) -> KernelIntermediateRepresentation:
        self.calls += 1
        return self.sim.step(action)

    def current(self) -> KernelIntermediateRepresentation:
        return self.sim.current_kir()

    def evaluate(self, ensemble, kir0: KernelIntermediateRepresentation,
                 actions: List[Optional[PolicyAction]],
                 divergence_threshold: float = 0.15) -> OracleReport:
        """Score surrogate against exact dynamics over one action sequence."""
        from .ensemble import EnsembleDynamics
        assert isinstance(ensemble, EnsembleDynamics)
        # single-step MSE over the sequence
        self.reset(kir0)
        kir = kir0
        prev_sw = kir.scheduler.total_context_switches
        sq = []
        for a in actions:
            nxt_exact = self.step(a)
            o_pred = ensemble.predict_mean_obs(kir, a)
            o_true = kir_observables(nxt_exact, prev_sw, self.dt_s)
            sq.append(float(np.mean((o_pred - o_true) ** 2)))
            prev_sw = nxt_exact.scheduler.total_context_switches
            kir = nxt_exact
        # open-loop rollout divergence: surrogate feeds itself, oracle is truth
        self.reset(kir0)
        kir_v = kir0.model_copy(deep=True)
        mae_curve = []
        div_step = None
        for i, a in enumerate(actions):
            kir_o = self.step(a)
            kir_v = ensemble.predict_kir(kir_v, a, dt_s=self.dt_s)
            mae = float(np.abs(encode_kir(kir_o) - encode_kir(kir_v)).mean())
            mae_curve.append(mae)
            if div_step is None and mae > divergence_threshold:
                div_step = i
        calls = self.calls
        return OracleReport(single_step_mse=float(np.mean(sq)) if sq else 0.0,
                            rollout_mae_curve=mae_curve,
                            rollout_horizon=len(actions),
                            divergence_step=div_step,
                            n_oracle_calls=calls)

    def sequence_reward(self, kir0: KernelIntermediateRepresentation,
                        latencies: List[int]) -> float:
        """Exact cumulative reward of a latency sequence (counts oracle calls)."""
        calc = RewardCalculator()
        self.reset(kir0)
        kir = kir0
        total = 0.0
        for lat in latencies:
            nxt = self.step(PolicyAction(
                policy_id="ladder",
                scheduler=SchedulerAction(target_latency_us=int(lat))))
            total += calc.calculate_step_reward(kir, nxt)
            kir = nxt
        return total


@dataclass
class LadderRung:
    name: str
    latencies: List[int]
    oracle_reward: float
    oracle_calls: int
    predicted_reward: Optional[float] = None


@dataclass
class LadderReport:
    """The 4-rung comparison ladder (context.txt Virtual Chess §'compare four
    systems'): exact search under oracle budget vs learned approaches."""
    rungs: List[LadderRung] = field(default_factory=list)
    budget: int = 0

    def to_dict(self) -> Dict:
        return {"budget": self.budget,
                "rungs": [vars(r) for r in self.rungs]}

    def winner(self) -> LadderRung:
        return max(self.rungs, key=lambda r: r.oracle_reward)


def run_ladder(ensemble, kir0: KernelIntermediateRepresentation,
               horizon: int = 3, budget: int = 20,
               arms=(1000, 2000, 4000, 6000, 12000, 24000),
               mcts_sims: int = 200, beam_width: int = 6,
               seed: int = 0, n_cpus: int = 2,
               dt_s: float = 0.5) -> LadderReport:
    """Compare four systems under a fixed oracle-interaction budget.

    Rung 1 — heuristic: HeuristicLatencyPolicy rolled out on the EXACT oracle.
    Rung 2 — exact search: exhaustive sequence eval on the oracle, capped at
             `budget` oracle calls (truncated arm set if needed).
    Rung 3 — surrogate beam: virtual_screen (0 oracle calls) + 1 validation.
    Rung 4 — surrogate MCTS: mcts_search (0 oracle calls) + 1 validation.
    """
    from .planning import mcts_search
    from .search import virtual_screen

    oracle = TruthOracle(seed=seed, n_cpus=n_cpus, dt_s=dt_s)
    rungs: List[LadderRung] = []

    # Rung 1: heuristic on exact dynamics
    oracle.reset(kir0)
    pol = HeuristicLatencyPolicy()
    kir, calls, rew = kir0, 0, 0.0
    calc = RewardCalculator()
    for _ in range(horizon):
        a = pol.decide(kir)
        nxt = oracle.step(a)
        calls += 1
        rew += calc.calculate_step_reward(kir, nxt)
        kir = nxt
    rungs.append(LadderRung("1-heuristic-exact", [], rew, calls))

    # Rung 2: exact exhaustive search within budget
    import itertools
    best_seq, best_rew, used = [], float("-inf"), 0
    for seq in itertools.product(arms, repeat=horizon):
        if used + horizon > budget:
            break
        r = oracle.sequence_reward(kir0, list(seq))
        used += horizon
        if r > best_rew:
            best_rew, best_seq = r, list(seq)
    rungs.append(LadderRung("2-exact-search-budgeted", best_seq, best_rew, used))

    # Rung 3: surrogate beam + single oracle validation
    results = virtual_screen(ensemble, kir0, horizon=horizon, beam_width=beam_width,
                             dt_s=dt_s)
    top = results[0].latencies
    r3 = oracle.sequence_reward(kir0, top)
    rungs.append(LadderRung("3-surrogate-beam", top, r3, horizon,
                            predicted_reward=results[0].predicted_reward))

    # Rung 4: surrogate MCTS + single oracle validation
    m = mcts_search(ensemble, kir0, horizon=horizon, sims=mcts_sims, seed=seed,
                    arms=tuple(arms), dt_s=dt_s)
    r4 = oracle.sequence_reward(kir0, m.best_latencies)
    rungs.append(LadderRung("4-surrogate-mcts", m.best_latencies, r4, horizon,
                            predicted_reward=m.value))

    return LadderReport(rungs=rungs, budget=budget)
