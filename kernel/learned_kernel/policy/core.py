"""
policy/core.py — Learned Kernel Policy ABI.

The PolicyBase ABC explicitly declares observation and action contracts,
making every policy introspectable without running it.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .schemas import KernelIntermediateRepresentation, PolicyAction, SchedulerAction


class PolicyBase(ABC):
    """
    First-class Policy ABI.

    Every concrete policy must declare:
      • observation_schema — which KIR fields it reads (dot-notation)
      • action_schema      — which kernel parameters it may write
      • decide()           — KIR → bounded PolicyAction (or None)

    Contracts let runtime, validator, and provenance logger reason about a
    policy without executing it.
    """

    @property
    @abstractmethod
    def policy_id(self) -> str:
        """Unique stable identifier, e.g. 'heuristic-latency-v1'."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string."""

    @property
    @abstractmethod
    def observation_schema(self) -> List[str]:
        """KIR fields this policy reads, in dot-notation."""

    @property
    @abstractmethod
    def action_schema(self) -> List[str]:
        """Kernel parameters this policy may write."""

    @abstractmethod
    def decide(self, kstate: KernelIntermediateRepresentation) -> Optional[PolicyAction]:
        """Map an observation to a bounded PolicyAction; None = no change."""


# ─────────────────────────────────────────────────────────────── #
# Concrete baselines                                              #
# ─────────────────────────────────────────────────────────────── #

class LinuxDefaultPolicy(PolicyBase):
    """Baseline: mimic an untouched kernel (no interventions)."""

    @property
    def policy_id(self) -> str: return "linux-baseline"
    @property
    def version(self) -> str: return "1.0"
    @property
    def observation_schema(self) -> List[str]: return []
    @property
    def action_schema(self) -> List[str]: return []

    def decide(self, kstate: KernelIntermediateRepresentation) -> Optional[PolicyAction]:
        return None


class HeuristicLatencyPolicy(PolicyBase):
    """
    Threshold heuristic with HYSTERESIS and change-suppression.

    Previous revision re-emitted a full action every cycle, so the actuator
    rewrote sysctls 100×/s and the single 0.8 threshold caused mode flapping
    when utilisation hovered at the boundary. Now:

      • tight mode engages above ENTER_TIGHT (0.80)
      • it releases only below EXIT_TIGHT (0.70) — a deadband
      • identical consecutive decisions return None (no actuator churn)
    """

    ENTER_TIGHT = 0.80
    EXIT_TIGHT = 0.70
    _TIGHT_PARAMS = (2000, 500)
    _LOOSE_PARAMS = (12000, 3000)

    def __init__(self) -> None:
        self._tight = False
        self._last_emitted: Optional[Tuple[int, int]] = None

    @property
    def policy_id(self) -> str: return "heuristic-latency-v2"
    @property
    def version(self) -> str: return "2.0"

    @property
    def observation_schema(self) -> List[str]:
        return ["scheduler.cpus.*.utilization"]

    @property
    def action_schema(self) -> List[str]:
        return ["scheduler.target_latency_us", "scheduler.wakeup_granularity_us"]

    def mean_utilization(self, kstate: KernelIntermediateRepresentation) -> float:
        cpus = kstate.scheduler.cpus
        if not cpus:
            return 0.0
        return sum(c.utilization for c in cpus.values()) / len(cpus)

    def reset(self) -> None:
        self._tight = False
        self._last_emitted = None

    def decide(self, kstate: KernelIntermediateRepresentation) -> Optional[PolicyAction]:
        u = self.mean_utilization(kstate)

        if not self._tight and u > self.ENTER_TIGHT:
            self._tight = True
        elif self._tight and u < self.EXIT_TIGHT:
            self._tight = False

        params = self._TIGHT_PARAMS if self._tight else self._LOOSE_PARAMS
        if params == self._last_emitted:
            return None                      # no change → suppress churn
        self._last_emitted = params
        return PolicyAction(
            policy_id=self.policy_id,
            scheduler=SchedulerAction(
                target_latency_us=params[0],
                wakeup_granularity_us=params[1],
            ),
        )
