"""conservation.py — trust requirement #2: conservation constraints.

context.txt (lines 589-591): "Does it preserve quantities that reality
guarantees must be preserved?"

A surrogate that predicts impossible kernel states — a decreasing switch
counter, time running backward, utilization outside [0,1], unbounded switch
rates — is untrustworthy no matter how good its mean accuracy is. These
invariants come from the exact dynamics (learned_kernel/simulator/env.py)
and the KIR ingestion contracts (policy/schemas.py). The oracle satisfies
them by construction; the surrogate must be MEASURED against them, in
particular out-of-distribution where accuracy metrics go blind.

Used by: benchmarks/run_levels.py (L4 mechanistic gate), VirtualLab
adversarial challenges, long-horizon rollout guards.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from learned_kernel.policy.schemas import KernelIntermediateRepresentation

# Physical constants mirrored from learned_kernel/simulator/env.py.
_MAX_SWITCH_RATE_S = 30_000.0
_MAX_LAT_MS = 40.0


def _mean_util(kir: KernelIntermediateRepresentation) -> float:
    cpus = list(kir.scheduler.cpus.values())
    return sum(c.utilization for c in cpus) / len(cpus) if cpus else 0.0


def check_transition(prev: KernelIntermediateRepresentation,
                     nxt: KernelIntermediateRepresentation,
                     dt_s: float = 0.5,
                     latent=None) -> List[Dict]:
    """Return one dict per violated invariant (empty = conserved)."""
    violations: List[Dict] = []

    d_sw = nxt.scheduler.total_context_switches - prev.scheduler.total_context_switches
    if d_sw < 0:
        violations.append({"invariant": "I1-monotonic-switches",
                           "detail": f"switch counter decreased by {-d_sw}"})

    dt = nxt.timestamp - prev.timestamp
    if not dt > 0:
        violations.append({"invariant": "I2-time-advance",
                           "detail": f"timestamp did not advance (dt={dt})"})
    elif abs(dt - dt_s) > dt_s:
        violations.append({"invariant": "I2-time-advance",
                           "detail": f"timestamp step {dt:.3f}s far from dt_s={dt_s}"})

    u = _mean_util(nxt)
    if not 0.0 <= u <= 1.0:
        violations.append({"invariant": "I3-util-bounds",
                           "detail": f"mean utilization {u} outside [0,1]"})
    if nxt.scheduler.avg_latency_ms <= 0:
        violations.append({"invariant": "I3-latency-positive",
                           "detail": f"latency {nxt.scheduler.avg_latency_ms} not positive"})
    if any(c.runnable_tasks < 0 for c in nxt.scheduler.cpus.values()):
        violations.append({"invariant": "I3-runnable-nonnegative",
                           "detail": "negative runnable_tasks predicted"})

    if dt > 0:
        rate = d_sw / dt
        if rate > _MAX_SWITCH_RATE_S:
            violations.append({"invariant": "I4-switch-rate-cap",
                               "detail": f"switch rate {rate:.0f}/s exceeds "
                                         f"physical cap {_MAX_SWITCH_RATE_S:.0f}/s"})

    if nxt.scheduler.avg_latency_ms > _MAX_LAT_MS:
        violations.append({"invariant": "I5-latency-cap",
                           "detail": f"latency {nxt.scheduler.avg_latency_ms}ms "
                                     f"exceeds physical cap {_MAX_LAT_MS}ms"})

    if latent is not None:
        import numpy as np
        linf = float(np.abs(latent).max())
        if linf > 1.0 + 1e-6:
            violations.append({"invariant": "I6-latent-bounded",
                               "detail": f"latent |.|_inf {linf:.3f} exceeds tanh bound 1"})

    return violations


def score_trajectory(kirs: List[KernelIntermediateRepresentation],
                     dt_s: float = 0.5) -> Dict:
    """Violation counts + rate over a full virtual or oracle trajectory."""
    total = 0
    by_invariant: Dict[str, int] = {}
    first_at: Optional[int] = None
    for i in range(1, len(kirs)):
        for v in check_transition(kirs[i - 1], kirs[i], dt_s):
            total += 1
            by_invariant[v["invariant"]] = by_invariant.get(v["invariant"], 0) + 1
            if first_at is None:
                first_at = i
    steps = max(len(kirs) - 1, 1)
    return {"transitions": steps, "violations": total,
            "violation_rate": total / steps,
            "by_invariant": by_invariant, "first_violation_at": first_at}


def conservation_report(virtual_trajs: List[List[KernelIntermediateRepresentation]],
                        oracle_trajs: List[List[KernelIntermediateRepresentation]],
                        dt_s: float = 0.5) -> Dict:
    """Side-by-side conservation: surrogate must match the oracle's ~zero rate."""
    def agg(trajs):
        scores = [score_trajectory(t, dt_s) for t in trajs]
        if not scores:
            return {"violation_rate": 0.0, "n_trajectories": 0}
        return {"violation_rate": sum(s["violation_rate"] for s in scores) / len(scores),
                "n_trajectories": len(scores),
                "trajectories_with_any_violation": sum(1 for s in scores if s["violations"]),
                "by_invariant": _merge([s["by_invariant"] for s in scores])}
    v, o = agg(virtual_trajs), agg(oracle_trajs)
    return {"virtual": v, "oracle": o,
            "parity": bool(v["violation_rate"] <= o["violation_rate"] + 1e-9)}


def _merge(dicts: List[Dict[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dicts:
        for k, n in d.items():
            out[k] = out.get(k, 0) + n
    return out
