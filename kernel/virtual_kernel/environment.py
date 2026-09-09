"""environment.py — the Virtual Kernel as an executable environment.

Direct port of context.txt's Virtual Cell API (lines 72-94) to the kernel:

    cell.observe()                    -> kernel.observe()
    cell.perturb(type, target)        -> kernel.intervene(action)
    cell.branch()                     -> kernel.branch(name)
    cell.rollout(time)                -> kernel.rollout(policy, steps)
    cell.measure(kind)                -> kernel.measure(kind)
    cell.uncertainty()                -> kernel.uncertainty(action)
    cell.compare(branch_a, branch_b)  -> kernel.compare(a, b)

Plus kernel-specific additions:
    fork() alias, sequential intervene chains X0->X1->X2, reset(), history,
    validate_against_oracle() (virtual prediction vs exact KernelSimulator).

Branching is copy-on-write over (kir, latent-vector, switch-counter):
each branch is an independent counterfactual world sharing the ensemble.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from learned_kernel.policy.core import PolicyBase
from learned_kernel.policy.schemas import KernelIntermediateRepresentation, PolicyAction

from .dynamics import encode_action, encode_kir
from .ensemble import EnsembleDynamics


@dataclass
class BranchState:
    name: str
    kir: KernelIntermediateRepresentation
    latent: np.ndarray
    history: List[KernelIntermediateRepresentation] = field(default_factory=list)
    actions: List[Optional[str]] = field(default_factory=list)


def _kir_distance(a: KernelIntermediateRepresentation,
                  b: KernelIntermediateRepresentation) -> float:
    fa = encode_kir(a)
    fb = encode_kir(b)
    return float(np.sqrt(((fa - fb) ** 2).mean()))


class VirtualKernel:
    """Branchable learned surrogate of kernel dynamics."""

    def __init__(self, ensemble: EnsembleDynamics,
                 initial_kir: KernelIntermediateRepresentation,
                 dt_s: float = 0.5):
        self.ensemble = ensemble
        self.dt_s = dt_s
        self.n_cpus = len(initial_kir.scheduler.cpus)
        z0 = ensemble.members[0].encode(encode_kir(initial_kir))
        self._branches: Dict[str, BranchState] = {
            "main": BranchState(name="main", kir=initial_kir, latent=np.array(z0),
                                history=[initial_kir])
        }
        self._active = "main"

    # -- core API (mirrors context.txt cell.*) -- #
    def observe(self, branch: Optional[str] = None) -> KernelIntermediateRepresentation:
        return self._branches[branch or self._active].kir

    def intervene(self, action: Optional[PolicyAction],
                  branch: Optional[str] = None) -> KernelIntermediateRepresentation:
        """Apply one virtual intervention: z' = F(z, a), x^ = D(z')."""
        key = branch or self._active
        bs = self._branches[key]
        a = encode_action(action)
        zs = [m.transition(bs.latent if i == 0 else m.encode(encode_kir(bs.kir)), a)
              for i, m in enumerate(self.ensemble.members)]
        # keep the mean-member latent as canonical persistent state
        all_o = np.array([m.decode(z, a) for m, z in zip(self.ensemble.members, zs)])
        best = int(np.argmin(((all_o - all_o.mean(axis=0)) ** 2).sum(axis=1)))
        bs.latent = zs[best]
        nxt = self.ensemble.members[best].predict_kir(bs.kir, action, dt_s=self.dt_s,
                                                      n_cpus=self.n_cpus)
        # keep latent consistent with what decoder actually rendered
        bs.kir = nxt
        bs.history.append(nxt)
        bs.actions.append(action.policy_id if action else "none")
        return nxt

    def branch(self, name: str, source: Optional[str] = None) -> str:
        src = self._branches[source or self._active]
        self._branches[name] = BranchState(
            name=name, kir=src.kir.model_copy(deep=True),
            latent=np.array(src.latent, copy=True),
            history=list(src.history), actions=list(src.actions))
        return name

    fork = branch  # alias

    def switch(self, name: str) -> None:
        if name not in self._branches:
            raise KeyError(f"unknown branch {name!r}")
        self._active = name

    def rollout(self, policy: PolicyBase, steps: int,
                branch: Optional[str] = None) -> List[KernelIntermediateRepresentation]:
        key = branch or self._active
        out = []
        for _ in range(steps):
            kir = self.observe(key)
            try:
                action = policy.decide(kir)
            except Exception:
                action = None
            out.append(self.intervene(action, key))
        return out

    def measure(self, kind: str = "kernel_state",
                branch: Optional[str] = None) -> dict:
        kir = self.observe(branch)
        cpus = list(kir.scheduler.cpus.values())
        base = {
            "mean_util": sum(c.utilization for c in cpus) / len(cpus),
            "avg_latency_ms": kir.scheduler.avg_latency_ms,
            "total_context_switches": kir.scheduler.total_context_switches,
            "timestamp": kir.timestamp,
        }
        if kind == "expression":
            return base  # compat alias for cell.measure("expression")
        if kind == "cell_state":
            return {**base, "latent_norm": float(np.linalg.norm(
                self._branches[branch or self._active].latent))}
        return base

    def uncertainty(self, action: Optional[PolicyAction] = None,
                    branch: Optional[str] = None) -> float:
        kir = self.observe(branch)
        return self.ensemble.uncertainty(kir, action)

    def compare(self, a: str, b: str) -> dict:
        ka, kb = self.observe(a), self.observe(b)
        return {
            "branches": (a, b),
            "kir_distance": _kir_distance(ka, kb),
            "latency_delta_ms": kb.scheduler.avg_latency_ms - ka.scheduler.avg_latency_ms,
            "steps_a": len(self._branches[a].history),
            "steps_b": len(self._branches[b].history),
        }

    # -- helpers -- #
    @property
    def branches(self) -> List[str]:
        return list(self._branches)

    @property
    def active_branch(self) -> str:
        return self._active

    def reset(self, kir: KernelIntermediateRepresentation,
              branch: Optional[str] = None) -> None:
        key = branch or self._active
        z0 = self.ensemble.members[0].encode(encode_kir(kir))
        self._branches[key] = BranchState(name=key, kir=kir, latent=np.array(z0),
                                          history=[kir])

    def reset_from_episode(self, episode, at_step: int = 0,
                           branch: Optional[str] = None) -> None:
        """Reset a branch to a recorded trace state (replayable scenarios)."""
        self.reset(episode.steps[at_step].kir, branch=branch)

    def scenario_kirs(self, episodes, at_step: int = 0):
        """One start-state per episode: demand-scenario set for robust search."""
        return [ep.steps[min(at_step, len(ep.steps) - 1)].kir for ep in episodes]

    def trajectory_uncertainty(self, branch: Optional[str] = None) -> List[float]:
        bs = self._branches[branch or self._active]
        return [self.ensemble.uncertainty(k, None) for k in bs.history]

    def validate_against_oracle(self, steps: int = 20, seed: int = 999,
                                branch: Optional[str] = None) -> dict:
        """Roll the surrogate open-loop vs the exact simulator; report drift."""
        from .oracle import TruthOracle
        oracle = TruthOracle(seed=seed, n_cpus=self.n_cpus, dt_s=self.dt_s)
        oracle.reset(self.observe(branch))
        kir_v = self.observe(branch).model_copy(deep=True)
        errs = []
        for _ in range(steps):
            lat = kir_v.scheduler.avg_latency_ms
            from learned_kernel.policy.schemas import PolicyAction as PA, SchedulerAction as SA
            act = PA(policy_id="replay", scheduler=SA(target_latency_us=6000))
            kir_o = oracle.step(act)
            kir_v = self.ensemble.predict_kir(kir_v, act, dt_s=self.dt_s)
            errs.append(_kir_distance(kir_o, kir_v))
        return {"horizon": steps, "mean_drift": float(np.mean(errs)),
                "final_drift": float(errs[-1]), "drift_curve": [float(e) for e in errs]}
