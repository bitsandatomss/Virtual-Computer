"""Unified trust checks: determinism, conservation, paired comparison.

Each check returns a (passed, detail) pair and never raises: the
vertical slice reports them as gates, and a failed gate caps the
unified grade instead of crashing the run.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def check_deterministic(first: dict, second: dict) -> tuple[bool, dict]:
    d1 = {"cycles": first.get("cycles"), "instrs": first.get("instrs"),
          "objective": first.get("objective"), "regs": first.get("regs"),
          "mem_out": first.get("mem_out")}
    d2 = {"cycles": second.get("cycles"), "instrs": second.get("instrs"),
          "objective": second.get("objective"), "regs": second.get("regs"),
          "mem_out": second.get("mem_out")}
    passed = digest(d1) == digest(d2)
    return passed, {"digest": digest(d1), "match": passed}


def check_cpu_conservation(result: dict, prog_len: int) -> tuple[bool, dict]:
    """Straight-line programs: retired == length; counters sane."""
    problems = []
    if result.get("completion_status") != "HALTED":
        problems.append(f"status={result.get('completion_status')}")
    if result.get("instrs") != prog_len:
        problems.append(f"instrs={result.get('instrs')}!=len={prog_len}")
    for k in ("cycles", "energy", "objective"):
        v = result.get(k)
        if not isinstance(v, (int, float)) or not (0 <= v < 10**12):
            problems.append(f"{k}={v!r} out of range")
    if result.get("thermal_exposure", 0) < 0:
        problems.append("thermal_exposure<0")
    return (not problems), {"problems": problems}


def check_functional_equivalence(res_a: dict, res_b: dict, expected: int) -> tuple[bool, dict]:
    problems = []
    for tag, r in (("O0", res_a), ("O3", res_b)):
        if r.get("acc") != expected:
            problems.append(f"{tag} acc={r.get('acc')}!=expected={expected}")
        if r.get("mem_out") != expected:
            problems.append(f"{tag} mem_out={r.get('mem_out')}!=expected={expected}")
    if res_a.get("acc") != res_b.get("acc"):
        problems.append("O0/O3 results differ")
    return (not problems), {"problems": problems, "expected": expected,
                            "acc_O0": res_a.get("acc"), "acc_O3": res_b.get("acc")}


def check_kernel_invariants(latencies: Sequence[float], switches: Sequence[float]) -> tuple[bool, dict]:
    problems = []
    if any(not (0 < v <= 40.0) for v in latencies):
        problems.append("latency out of (0,40]ms")
    if any(b < a for a, b in zip(switches, switches[1:])):
        problems.append("switches non-monotonic (I1)")
    return (not problems), {"problems": problems, "steps": len(latencies)}


def paired_stats(deltas: Sequence[float], baseline: float, margin: float = 0.02) -> dict:
    """Seeded bootstrap median CI + gated verdict (layer metrology)."""
    from virtual_compiler.metrology import bootstrap_median_ci, gate_verdict

    deltas = list(deltas)
    lo, hi = bootstrap_median_ci(deltas)
    med = sorted(deltas)[len(deltas) // 2]
    return {"n": len(deltas), "median": med, "ci_low": lo, "ci_high": hi,
            "verdict": gate_verdict(med, lo, hi, margin=margin, baseline=baseline),
            "baseline": baseline, "margin": margin}
