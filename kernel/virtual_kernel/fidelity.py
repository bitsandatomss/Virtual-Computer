"""fidelity.py — "sufficiently faithful" is task-dependent (context.txt 624-641).

The file's key move: a surrogate need not reproduce the underlying mechanism;
it must reproduce RELEVANT BEHAVIOR, where relevance is defined by the
question asked. A surrogate with mediocre point accuracy can still be
sufficient for search (if it ranks candidates like the oracle), while a
surrogate with good average accuracy can still be unfit for robust control
(if its worst case diverges).

This module makes that explicit: each practitioner question gets a
FidelitySpec (metric + pass threshold + direction). levels.py and the
benchmarks check specs instead of arguing about a single accuracy number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal


@dataclass(frozen=True)
class FidelitySpec:
    task: str
    question: str
    metric: str
    threshold: float
    direction: Literal["min", "max"]  # min: value>=threshold passes; max: value<=threshold
    why: str = ""


STANDARD_SPECS: List[FidelitySpec] = [
    FidelitySpec(
        task="search-ranking",
        question="Does the surrogate rank candidate configurations like the oracle?",
        metric="spearman_rank", threshold=0.5, direction="min",
        why="Search only needs order preservation; bias that preserves rank is harmless."),
    FidelitySpec(
        task="search-topk",
        question="Is the oracle's true best inside the surrogate's top-k?",
        metric="topk_hit_rate", threshold=1.0, direction="min",
        why="Validation budget is spent on top-k; a miss wastes the oracle calls."),
    FidelitySpec(
        task="point-prediction",
        question="How wrong is one virtual step on unseen workloads?",
        metric="test_mse", threshold=0.05, direction="max",
        why="Single-step error bounds every multi-step claim built on top of it."),
    FidelitySpec(
        task="long-horizon",
        question="How many virtual steps before the surrogate diverges?",
        metric="divergence_step", threshold=4, direction="min",
        why="Planning horizons longer than the divergence step are fiction."),
    FidelitySpec(
        task="robust-worst-case",
        question="Under the most adversarial demand regime, how far from optimal?",
        metric="worst_case_gap", threshold=-0.5, direction="min",
        why="Fragile optima are the virtual-lab explosion scenario."),
    FidelitySpec(
        task="conservation-parity",
        question="Does the surrogate break physical invariants the oracle never breaks?",
        metric="conservation_parity", threshold=1.0, direction="min",
        why="Invariant violations are structural untrustworthiness, not noise."),
]


def check(spec: FidelitySpec, value: float) -> bool:
    if spec.direction == "min":
        return bool(value >= spec.threshold)
    return bool(value <= spec.threshold)


def fidelity_report(specs: List[FidelitySpec],
                    measurements: Dict[str, float]) -> Dict[str, Any]:
    rows = []
    for s in specs:
        v = measurements.get(s.metric)
        rows.append({"task": s.task, "question": s.question, "metric": s.metric,
                     "threshold": s.threshold, "direction": s.direction,
                     "value": v,
                     "passed": (check(s, v) if v is not None else None)})
    decided = [r for r in rows if r["passed"] is not None]
    return {"specs": rows,
            "n_passed": sum(1 for r in decided if r["passed"]),
            "n_decided": len(decided),
            "all_passed": bool(decided) and all(r["passed"] for r in decided)}
