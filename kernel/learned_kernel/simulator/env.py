"""
simulator/env.py — Causal kernel-behaviour model.

Why this exists
---------------
The previous trainer evaluated candidate policies against *observational*
trajectories: reward(prev, curr) depended only on recorded telemetry, never on
the action the candidate would have taken, so every gradient estimate was ~0
and training was a no-op. Offline RL fundamentally requires a model of
(state, action) -> next state. This module provides an explicit, documented,
seeded dynamics model of scheduler behaviour so that

    reward(candidate policy) = rollout(candidate policy through KernelSimulator)

is a genuine counterfactual evaluation.

Honesty scope
-------------
These dynamics are first-order approximations for a prototype. Real-kernel
evaluation requires replaying logged (s, a, s', r) tuples or live shadow
deployment; both paths are noted in README/LIMITATIONS.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from ..policy.schemas import (
    CPUMetrics,
    KernelIntermediateRepresentation,
    PolicyAction,
    SchedulerState,
)

# ─────────────────────────────────────────────────────────────── #
# Workload model                                                   #
# ─────────────────────────────────────────────────────────────── #

@dataclass
class WorkloadProfile:
    """
    Demand intensity D(t) in [0, 1]. Regimes switch via a two-state Markov
    chain (idle-ish vs saturated); within a regime D follows a seeded random
    walk, giving realistic autocorrelated load instead of i.i.d. noise.
    """
    rng: random.Random = field(repr=False)
    dt_s: float = 0.5
    p_rise: float = 0.08          # P(idle -> busy) per step
    p_fall: float = 0.08          # P(busy -> idle) per step
    demand: float = 0.3           # current intensity
    regime: float = 0.3           # regime anchor the walk reverts toward

    def advance(self) -> float:
        if self.rng.random() < self.p_rise:
            self.regime = 0.95
        elif self.rng.random() < self.p_fall:
            self.regime = 0.25
        # mean-reverting walk inside [0.05, 1]
        self.demand += 0.4 * (self.regime - self.demand) + self.rng.gauss(0, 0.05)
        self.demand = min(1.0, max(0.05, self.demand))
        return self.demand


# ─────────────────────────────────────────────────────────────── #
# Dynamics constants                                               #
# ─────────────────────────────────────────────────────────────── #
#
# Physics of the two opposing terms (validated by tests):
#   queueing  ∝ D · (T/T_ref)^ALPHA_Q      — longer windows make each task
#                                            wait longer before its slice
#   overhead  ∝ ω · (T_ref/T)^BETA_OH      — shorter windows preempt more,
#                                            burning cycles on switching
#
# latency(D,T) = queueing + overhead has an INTERIOR optimum
#   T* = T_ref · (ω·β·(0.5+D) / (D·Q_SCALE·α))^(1/(α+β))
# so idle systems favour long targets and busy systems tight ones.

_T_REF_US = 6_000          # CFS default target latency (reference point)
_Q_SCALE_MS = 8.0          # queueing-delay scale at D=1, T=T_ref
_ALPHA_Q = 1.0             # queueing growth w.r.t. window length
_BETA_OH = 1.2             # preemption-overhead growth as target shrinks
_OMEGA_OH_MS = 0.8         # overhead scale (ms) at T_ref
_GAMMA_SW = 0.5            # switch-rate elasticity w.r.t. target latency
_SR_BASE_SW_S = 400.0      # switches/s at D=1 with default target
_MAX_LAT_MS = 40.0


class KernelSimulator:
    """
    Seedable scheduler-dynamics environment.

        latency_ms(D, T) ≈ D·Qs·(T/T_ref)^α + ω·(T_ref/T)^β·(0.5+D)  (clamped)
        switch_rate(D, T) ≈ sr₀·D·(T_ref/T)^γ                        (capped)

    The two opposing terms make the latency/response optimum INTERIOR:
    lengthening T reduces preemption overhead but increases queueing delay —
    exactly the trade-off a real CFS tunable exhibits and exactly the signal
    a learnable policy needs.
    """

    def __init__(
        self,
        seed: int = 1234,
        n_cpus: int = 2,
        workload: Optional[WorkloadProfile] = None,
        dt_s: float = 0.5,
    ):
        self._rng = random.Random(seed)
        self.n_cpus = n_cpus
        self.dt_s = dt_s
        self.workload = workload or WorkloadProfile(
            rng=random.Random(seed ^ 0xBEEF), dt_s=dt_s)
        self.reset()

    # ------------------------------------------------------------------ #

    def reset(self, from_state: Optional[KernelIntermediateRepresentation] = None) -> None:
        """Start (or restart) an episode; optionally seed internal stats."""
        self._t = 1000.0
        self._switches = 0
        self._latency_ms = 4.0
        self._util = 0.3
        self._runnable = 2
        self._irq = 0.02
        if from_state is not None:
            s = from_state.scheduler
            self._switches = s.total_context_switches
            self._latency_ms = s.avg_latency_ms
            if s.cpus:
                utils = [c.utilization for c in s.cpus.values()]
                runnables = [c.runnable_tasks for c in s.cpus.values()]
                irqs = [c.irq_time for c in s.cpus.values()]
                self._util = sum(utils) / len(utils)
                self._runnable = int(sum(runnables) / len(runnables))
                self._irq = sum(irqs) / len(irqs)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _effective_latency(demand: float, target_us: float) -> float:
        t = max(float(target_us), 1.0)
        ratio_up = t / _T_REF_US          # >1 when window longer than default
        queueing = demand * _Q_SCALE_MS * (ratio_up ** _ALPHA_Q)
        overhead = (_OMEGA_OH_MS * (ratio_up ** -_BETA_OH)) * (0.5 + demand)
        return min(_MAX_LAT_MS, max(0.05, queueing + overhead))

    @staticmethod
    def _switch_rate(demand: float, target_us: float) -> float:
        t = max(float(target_us), 1.0)
        return min(30_000.0, _SR_BASE_SW_S * demand * ((t / _T_REF_US) ** -_GAMMA_SW))

    def current_kir(self) -> KernelIntermediateRepresentation:
        cpus = {
            i: CPUMetrics(utilization=self._util, runnable_tasks=self._runnable,
                          irq_time=self._irq)
            for i in range(self.n_cpus)
        }
        return KernelIntermediateRepresentation(
            timestamp=self._t,
            scheduler=SchedulerState(
                cpus=cpus,
                total_context_switches=int(self._switches),
                avg_latency_ms=round(self._latency_ms, 4),
            ),
        )

    # ------------------------------------------------------------------ #

    def step(self, action: Optional[PolicyAction]) -> KernelIntermediateRepresentation:
        """Advance one tick, applying the actuated target latency if present."""
        target_us = _T_REF_US
        if action is not None and action.scheduler is not None \
                and action.scheduler.target_latency_us is not None:
            target_us = float(action.scheduler.target_latency_us)

        d = self.workload.advance()
        self._t += self.workload.dt_s

        new_lat = self._effective_latency(d, target_us)
        rate = self._switch_rate(d, target_us)
        self._switches += rate * self.workload.dt_s

        # Overhead inflates effective utilisation (cycles burned switching).
        overhead_ms = (_OMEGA_OH_MS * (target_us / _T_REF_US) ** -_BETA_OH) * (0.5 + d)
        oh_frac = overhead_ms / max(new_lat, 0.05)
        self._util = min(1.0, d * (1.0 + 0.5 * oh_frac))
        self._runnable = max(0, int(round(d * 12)))
        self._irq = min(1.0, 0.01 + 0.04 * d)
        self._latency_ms = new_lat
        return self.current_kir()
