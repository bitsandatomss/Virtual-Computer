"""Acquisition rules: which oracle view to buy next.

context.txt T16 ("which surface should I trace next to maximize expected
recoverable letters" → "which build maximizes expected improvement").
Random oracle spend is the baseline these rules must beat; the benchmark
harness (`benchmark.py`) can pit acquisition rules against each other.
"""
from __future__ import annotations

import math

ACQUISITIONS = ("uncertainty", "expected-improvement", "ucb", "info-gain")


def _incumbent(env) -> float | None:
    best: float | None = None
    for st in env.branches.values():
        if st.validated and st.build_ok and st.runtime_ms is not None:
            if best is None or st.runtime_ms < best:
                best = st.runtime_ms
    return best


def score_branch(env, name: str, rule: str = "uncertainty",
                 beta: float = 2.0) -> float:
    pred = env.surrogate.predict(env.branches[name])
    unc = pred.uncertainty
    if rule == "uncertainty":
        return unc
    if rule == "ucb":
        # lower-is-better UCB on runtime; missing estimates score poorly
        if pred.runtime_ms is None:
            return -unc
        return -(pred.runtime_ms - beta * unc * max(pred.runtime_ms, 1.0))
    if rule == "expected-improvement":
        inc = _incumbent(env)
        if inc is None or pred.runtime_ms is None:
            return unc
        improvement = max(0.0, inc - pred.runtime_ms)
        return improvement * (1.0 - unc) + 0.1 * unc
    if rule == "info-gain":
        # proxy: uncertainty weighted by how alone the branch is
        # (far from validated experience ⇒ more to learn)
        validated = sum(1 for s in env.branches.values() if s.validated)
        return unc * math.log(2.0 + len(env.branches) - validated)
    raise ValueError(f"unknown acquisition rule: {rule!r}")


def select(env, rule: str = "uncertainty",
           only_unvalidated: bool = True) -> str | None:
    """Return the branch name most worth validating, or None."""
    cands = [(n, s) for n, s in env.branches.items()
             if not (only_unvalidated and s.validated)]
    if not cands:
        return None
    scored = sorted(((score_branch(env, n, rule), n) for n, _ in cands),
                    key=lambda t: -t[0])
    return scored[0][1]
