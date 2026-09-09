"""The killer benchmark (context.txt T2/T17).

Every strategy runs under an *identical, finite oracle budget* on a suite
of *held-out programs*; the stock baseline gets the same budget. Reported
per program and worst-case across programs (T17's generalized objective:
the winner must win everywhere, not just on one workload):

- best oracle runtime found, oracle calls spent, virtual calls spent
- gain vs baseline, gain per oracle interaction
- refusal discipline: arms whose evidence is within-noise are reported
  as ``within-noise``, never as wins.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .evidence import EvidenceLog
from .search import STRATEGIES, gain_per_oracle, run_strategy

ARMS = ("stock", "random", "surrogate", "beam", "mcts", "swarm")


@dataclass
class ArmReport:
    arm: str
    best_branch: str
    best_runtime_ms: float | None
    baseline_ms: float | None
    oracle_calls: int
    virtual_calls: int
    gain: float | None
    gain_per_oracle: float | None
    verdict: str = "within-noise"
    flags: list[str] | None = None  # winning config, for verification


@dataclass
class ProgramReport:
    program: str
    arms: list[ArmReport] = field(default_factory=list)
    winner: str = "none"
    verification: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    schema: str = "virtual-compiler.benchmark.v1"
    objective: str = "runtime"
    budget_per_program: int = 6
    programs: list[ProgramReport] = field(default_factory=list)
    worst_case_gain: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "objective": self.objective,
            "budget_per_program": self.budget_per_program,
            "elapsed_s": self.elapsed_s,
            "worst_case_gain": self.worst_case_gain,
            "programs": [
                {"program": p.program, "winner": p.winner,
                 "verification": p.verification,
                 "arms": [vars(a) for a in p.arms]} for p in self.programs],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2),
                              encoding="utf-8")


def run_killer_benchmark(
    make_env: Callable[[str], Any],
    programs: dict[str, str],
    arms: tuple[str, ...] = ARMS,
    budget_per_program: int = 6,
    top_k: int = 2,
    objective: str = "runtime",
    evidence: EvidenceLog | None = None,
    seed: int = 0,
    seeds: int = 1,
    verify_winners: bool = False,
    compiler: str = "gcc",
) -> BenchmarkReport:
    """Run every arm on every program with equal oracle budgets.

    ``make_env(source_text)`` builds a fresh `VirtualCompiler` with the
    oracle budget already set; each arm receives a *fresh* env so budgets
    cannot leak between arms. ``seeds`` repeats every (program, arm) cell
    (labels ``prog@sN``) so arms differ by skill, not luck (CRITIQUE G6).
    With ``verify_winners``, the program winner is re-checked against
    stock via paired interleaved measurement + differential gate
    (CRITIQUE G3); verification builds are reported separately, never
    hidden inside arm budgets.
    """
    for arm in arms:
        if arm not in STRATEGIES:
            raise ValueError(f"unknown arm: {arm!r}")
    report = BenchmarkReport(objective=objective,
                             budget_per_program=budget_per_program)
    start = time.perf_counter()
    for prog_name, source in programs.items():
        for s in range(max(1, seeds)):
            label = prog_name if seeds == 1 else f"{prog_name}@s{seed + s}"
            prog = ProgramReport(program=label)
            for arm in arms:
                env = make_env(source)
                try:
                    env.validate("root")
                    baseline = env.branches["root"].runtime_ms
                except (RuntimeError, OSError):
                    baseline = None
                used_before = env.oracle.used
                try:
                    res = run_strategy(env, arm, top_k=top_k,
                                       objective=objective, seed=seed + s)
                except (RuntimeError, OSError):
                    res = None
                spent = env.oracle.used - used_before
                best = res.best_runtime_ms if res else baseline
                gain = (None if baseline is None or best is None
                        or baseline <= 0
                        else (baseline - best) / baseline)
                gpo = gain_per_oracle(baseline, best, spent)
                verdict = ("improves" if gain is not None and gain > 0.02
                           else "within-noise")
                prog.arms.append(ArmReport(
                    arm=arm, best_branch=res.best_branch if res else "root",
                    best_runtime_ms=best, baseline_ms=baseline,
                    oracle_calls=spent,
                    virtual_calls=res.virtual_calls if res else 0,
                    gain=gain, gain_per_oracle=gpo, verdict=verdict,
                    flags=(list(env.branches[res.best_branch].flags)
                           if res and res.best_branch in env.branches
                           else None)))
            gains = {a.arm: (a.gain if a.gain is not None else 0.0)
                     for a in prog.arms}
            prog.winner = max(gains, key=lambda k: gains[k])
            if gains[prog.winner] <= 0.02:
                prog.winner = "none"
            if verify_winners and prog.winner not in ("none", "stock"):
                prog.verification = _verify_winner(
                    source, prog, compiler, objective)
            report.programs.append(prog)
            if evidence is not None:
                evidence.emit("validate", "benchmark-program",
                              program=label, winner=prog.winner)
    for arm in arms:
        worst = min(
            (a.gain if a.gain is not None else 0.0)
            for p in report.programs for a in p.arms if a.arm == arm)
        report.worst_case_gain[arm] = worst
    report.elapsed_s = time.perf_counter() - start
    return report


def _verify_winner(source: str, prog: ProgramReport, compiler: str,
                   objective: str) -> dict:
    """Paired + differential re-check of winner vs stock (2 extra builds)."""
    from .metrology import build_and_compare

    arms = {a.arm: a for a in prog.arms}
    stock = arms.get("stock")
    winner = arms.get(prog.winner)
    if stock is None or winner is None:
        return {"status": "skipped"}
    if winner.flags is None or stock.flags is None:
        return {"status": "unrecorded-config"}
    if list(winner.flags) == list(stock.flags):
        return {"status": "same-config"}
    try:
        comp = build_and_compare(source, list(stock.flags),
                                 list(winner.flags), compiler=compiler)
    except (RuntimeError, OSError) as exc:
        return {"status": "oracle-unavailable", "error": str(exc)[:200]}
    if comp.paired is None:
        return {"status": "failed", "error": comp.error[:200]}
    return {
        "status": "verified",
        "equivalent": comp.equivalent,
        "paired_gain": (comp.paired.paired_median_delta_ms
                        / max(comp.paired.median_a_ms, 1e-9)),
        "ci_low": comp.paired.ci_low_ms,
        "ci_high": comp.paired.ci_high_ms,
        "verdict": comp.paired.verdict,
        "builds": comp.builds,
    }
