"""
runtime/adaptation.py — Workload-drift detection and online policy adaptation.

Drift = statistically significant reward degradation vs a local baseline,
judged by a Welch t-test over two disjoint rolling windows. A plain scalar
mean comparison has no variance model and fires on noise.

Fixes over the previous revision
--------------------------------
• deque(maxlen) instead of list.pop(0)
• check_drift(consume=True) clears the window when it fires so a single drift
  event cannot re-alarm every cycle until someone happens to adapt
• optional min_effect_size gate: requires an actual mean gap, not just a tiny
  variance, before declaring drift (cuts false positives further)
• reward function / trainer are injected, not constructed internally
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional

from ..policy.schemas import KernelIntermediateRepresentation
from ..policy.core import PolicyBase
from ..trainer.reward import RewardCalculator
from ..trainer.rl_trainer import OfflineTrainer


def welch_t_statistic(a: List[float], b: List[float]) -> float:
    """Welch's t-statistic; t < 0 means mean(a) < mean(b)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n_a, n_b = len(a), len(b)
    mu_a = sum(a) / n_a
    mu_b = sum(b) / n_b
    var_a = sum((x - mu_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mu_b) ** 2 for x in b) / (n_b - 1)
    denom = math.sqrt(var_a / n_a + var_b / n_b)
    if denom < 1e-9:
        return 0.0
    return (mu_a - mu_b) / denom


class WorkloadDriftManager:
    """
    Two rolling windows of length `window_size`:
      reference (older half) vs current (newer half).
    Drift fires when t < -t_threshold AND μ_ref − μ_cur ≥ min_effect_size.
    """

    def __init__(
        self,
        window_size: int = 10,
        t_threshold: float = 2.0,
        min_effect_size: float = 0.05,
        reward_fn: Optional[RewardCalculator] = None,
        trainer: Optional[OfflineTrainer] = None,
        verbose: bool = True,
    ):
        if window_size < 2:
            raise ValueError("window_size must be ≥ 2")
        self.window_size = window_size
        self.t_threshold = t_threshold
        self.min_effect_size = min_effect_size
        self._ring: Deque[float] = deque(maxlen=window_size * 2)
        self.reward_fn = reward_fn or RewardCalculator()
        self.trainer = trainer or OfflineTrainer()
        self.verbose = verbose
        self.last_statistic: Optional[float] = None

    # ------------------------------------------------------------------ #

    def add_step(
        self,
        prev: KernelIntermediateRepresentation,
        curr: KernelIntermediateRepresentation,
    ) -> float:
        r = self.reward_fn.calculate_step_reward(prev, curr)
        self._ring.append(r)
        return r

    def _windows(self):
        ref = list(self._ring)[: self.window_size]
        cur = list(self._ring)[self.window_size:]
        return ref, cur

    # ------------------------------------------------------------------ #

    def check_drift(self, consume: bool = False) -> bool:
        """True iff recent reward is significantly below baseline.

        consume=True clears the windows after firing so one drift event is
        reported exactly once.
        """
        required = self.window_size * 2
        fired = False
        if len(self._ring) >= required:
            reference, recent = self._windows()
            t = welch_t_statistic(recent, reference)
            self.last_statistic = t
            mu_ref = sum(reference) / len(reference)
            mu_rec = sum(recent) / len(recent)
            effect = mu_ref - mu_rec
            if t < -self.t_threshold and effect >= self.min_effect_size:
                fired = True
                if self.verbose:
                    print(f"[DriftManager] Drift detected  t={t:.3f} < -{self.t_threshold}"
                          f"  Δμ={effect:.4f}  (μ_recent={mu_rec:.4f}, μ_base={mu_ref:.4f})")
        if fired and consume:
            self._ring.clear()
        return fired

    # ------------------------------------------------------------------ #

    def trigger_adaptation(
        self,
        historical_trajectories: List[List[KernelIntermediateRepresentation]],
    ) -> PolicyBase:
        if self.verbose:
            print("[DriftManager] Triggering offline policy adaptation ...")
        new_policy = self.trainer.train(historical_trajectories)
        self._ring.clear()      # reset so the adaptation's effect is measurable
        return new_policy
