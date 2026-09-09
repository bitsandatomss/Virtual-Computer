"""The experimental engine: the endpoint of context.txt T15.

"N virtual trajectories → agents scrutinize → K real builds → results
update the model → search again." A campaign is the product's top-level
object; single screens are its inner step. Each round:

1. FAN-OUT — propose + expand N virtual trajectories (free).
2. SCRUTINY — swarm (analyst/adversary/critic) cuts N → shortlist.
3. PHYSICS — validate at most K (oracle budget).
4. UPDATE — validations already updated the surrogate (learn stage).
5. REPEAT — next round searches from the new incumbent.
"""
from __future__ import annotations

from typing import Any

from .agents import Adversary, Analyst, Critic, Proposer
from .environment import VirtualCompiler


def run_campaign(env: VirtualCompiler, rounds: int = 3,
                 fanout: int = 8, physics_k: int = 2,
                 rule: str = "expected-improvement") -> dict[str, Any]:
    """Run the closed loop; returns round-by-round engine telemetry."""
    from .acquisition import score_branch

    proposer, analyst, adversary, critic = (
        Proposer(), Analyst(), Adversary(), Critic())
    history: list[dict[str, Any]] = []
    total_virtual = total_oracle = 0
    for rnd in range(1, max(1, rounds) + 1):
        base = env.current
        cands = proposer.propose(env, n=fanout)
        expanded: list[str] = []
        for action, arg in cands:
            try:
                expanded.append(env.perturb(
                    action, arg, branch=base,
                    new_branch=f"camp-r{rnd}:{action}={arg}"))
                total_virtual += 1
            except ValueError:
                continue
        env.current = base
        # scrutiny: drop critic-flagged branches, rank the rest
        flagged = {v.branch for v in critic.review(env)}
        ranked = sorted(
            [n for n in expanded if n not in flagged],
            key=lambda n: -score_branch(env, n, rule))
        shortlist = ranked[:max(1, physics_k * 2)]
        ref = adversary.refute(env)
        validated: list[str] = []
        skipped_constraints: list[str] = []
        for name in shortlist:
            if len(validated) >= physics_k:
                break
            if env.oracle.used >= env.oracle_budget:
                break
            # simulator constrains the surrogate (N2): error-level
            # findings veto oracle spend before it happens.
            if any(f["level"] == "error" for f in env.check(name)):
                skipped_constraints.append(name)
                continue
            try:
                env.validate(name)
                validated.append(name)
                total_oracle += 1
            except (RuntimeError, OSError):
                continue
        # new incumbent = best validated, else previous
        scored = [(env.branches[n].runtime_ms or float("inf"), n)
                  for n in validated
                  if env.branches[n].build_ok]
        if scored:
            env.current = min(scored)[1]
        history.append({
            "round": rnd,
            "fanout_virtual": len(expanded),
            "flagged": len(flagged),
            "shortlist": shortlist,
            "validated": validated,
            "skipped_constraints": skipped_constraints,
            "adversarial_claim": ref.claim if ref else None,
            "analyst_claim": analyst.interpret(env).claim,
            "incumbent": env.current,
            "incumbent_runtime": env.branches[env.current].runtime_ms,
        })
        if env.oracle.used >= env.oracle_budget:
            break
    return {
        "rounds": history,
        "incumbent": env.current,
        "incumbent_runtime": env.branches[env.current].runtime_ms,
        "total_virtual": total_virtual,
        "total_oracle": total_oracle,
        "oracle_budget": env.oracle_budget,
    }
