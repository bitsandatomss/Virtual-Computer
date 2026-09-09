"""
trainer/reward.py — Learned Kernel reward model.

    R(t) = β·T̄ - α·L̄ - γ·Ū - δ·Θ(T̄)

where all terms are dimensionless (normalised within expected ranges) so the
coefficients are directly interpretable as relative weights:

    T̄  normalised context-switch throughput      in [0, 1]
    L̄  normalised scheduler latency              in [0, 1]
    Ū  mean CPU utilisation                      in [0, 1]
    Θ  excess-switching ("thrash") term          in [0, 1]

Design note — why Θ exists
--------------------------
Throughput measured by raw switch rate is a *perverse incentive*: a policy can
manufacture reward by shrinking the latency target until the scheduler
preempts constantly, inflating T̄ while destroying cache locality and doing
less useful work. Θ(T̄) = clamp((T̄ - τ)/(1 - τ), 0, 1) activates only above a
sustainable switching threshold τ, so healthy throughput is still rewarded
but preemption storms are penalised.
"""
from ..policy.schemas import KernelIntermediateRepresentation

# --- normalisation constants ---------------------------------------------
# Throughput: a healthy server sees O(10 000) switches/second.
_NORM_THROUGHPUT: float = 10_000.0
# Latency: CFS default target is 6 ms; anything above ~20 ms is severely bad.
_NORM_LATENCY_MS: float = 20.0


class RewardCalculator:
    """Unified physical reward function for kernel policy evaluation."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta: float = 0.6,
        thrash_tau: float = 0.85,
        norm_throughput: float = _NORM_THROUGHPUT,
        norm_latency_ms: float = _NORM_LATENCY_MS,
    ):
        if not (0.0 <= thrash_tau < 1.0):
            raise ValueError("thrash_tau must lie in [0, 1)")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.thrash_tau = thrash_tau
        self._norm_tp = norm_throughput
        self._norm_lat = norm_latency_ms

    # ------------------------------------------------------------------ #

    def explain_step_reward(
        self,
        prev: KernelIntermediateRepresentation,
        curr: KernelIntermediateRepresentation,
    ) -> dict:
        """Return the individual normalised terms (for debugging/audit)."""
        dt = curr.timestamp - prev.timestamp
        if dt <= 0:
            dt = 1e-3

        # Kernel counters can reset; treat reset as unobservable (0 delta).
        sw_delta = max(0, curr.scheduler.total_context_switches
                       - prev.scheduler.total_context_switches)
        throughput_norm = min(sw_delta / dt, self._norm_tp) / self._norm_tp
        thrash = max(0.0, (throughput_norm - self.thrash_tau)
                     / (1.0 - self.thrash_tau))

        latency_norm = min(curr.scheduler.avg_latency_ms, self._norm_lat) / self._norm_lat

        avg_util = 0.0
        if curr.scheduler.cpus:
            avg_util = sum(c.utilization for c in curr.scheduler.cpus.values()) / len(
                curr.scheduler.cpus)

        return {
            "throughput": throughput_norm,
            "latency": latency_norm,
            "utilization": avg_util,
            "thrash": thrash,
            "reward": (
                self.beta * throughput_norm
                - self.alpha * latency_norm
                - self.gamma * avg_util
                - self.delta * thrash
            ),
        }

    def calculate_step_reward(
        self,
        prev: KernelIntermediateRepresentation,
        curr: KernelIntermediateRepresentation,
    ) -> float:
        return float(self.explain_step_reward(prev, curr)["reward"])
