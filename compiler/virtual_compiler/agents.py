"""Agent swarm operating inside the virtual compiler (context.txt T11).

Roles (cell swarm × chess swarm, transposed):

- Proposer   — hypotheses: which interventions deserve virtual expansion
- Optimizer  — chooses flag/policy perturbations (runs a search strategy)
- Analyst    — reads surrogate + oracle evidence into claims
- Adversary  — searches for refutations: branches that beat the incumbent
- Validator  — active learning: what deserves scarce oracle budget
- Critic     — surrogate-trust scoring: where the model may be wrong

All agents share one blackboard: the environment's branch DAG. Every
verdict cites a branch id. `run_swarm` keeps its v1 return contract
(`current`, `log`, `oracle_used`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .acquisition import select as acquire
from .environment import FLAG_ACTIONS, POLICY_ACTIONS, VirtualCompiler
from .search import virtual_screen
from .transforms import available


@dataclass
class AgentVerdict:
    agent: str
    claim: str
    branch: str
    detail: str = ""


@dataclass
class Blackboard:
    """Shared swarm memory: every claim, with its evidence branch."""
    verdicts: list[AgentVerdict] = field(default_factory=list)
    validations_spent: int = 0

    def post(self, verdict: AgentVerdict) -> None:
        self.verdicts.append(verdict)

    def consensus(self) -> str:
        if not self.verdicts:
            return "no-verdict"
        return self.verdicts[-1].branch


class Proposer:
    def propose(self, env: VirtualCompiler, n: int = 4) -> list[tuple[str, str]]:
        scored: list[tuple[float, tuple[str, str]]] = []
        base = env.current
        cands = ([("flags", k) for k in FLAG_ACTIONS]
                 + [("policy", k) for k in POLICY_ACTIONS])
        # Add source transforms that are applicable to the current source.
        if base in env.branches:
            src = env.branches[base].source_text
            for name in available(src):
                cands.append(("xform", name))
        for action, arg in cands:
            try:
                name = env.perturb(action, arg, branch=base,
                                   new_branch=f"tmp-propose:{arg}")
            except ValueError:
                continue
            m = env.measure("build_ok", name)
            scored.append((float(m["value"] or 0) - 0.5 * float(m["uncertainty"]),
                           (action, arg)))
            del env.branches[name]
        env.current = base
        scored.sort(key=lambda t: -t[0])
        return [c for _, c in scored[:n]]


class Optimizer:
    def run(self, env: VirtualCompiler, top_k: int = 3,
            objective: str = "runtime"):
        return virtual_screen(env, top_k=top_k, objective=objective)


class Analyst:
    def interpret(self, env: VirtualCompiler) -> AgentVerdict:
        cur = env.current
        obs = env.observe(cur)
        s = obs["surrogate"]
        claim = (f"branch {cur}: ok_prob={s['ok_prob']:.2f} "
                 f"unc={s['uncertainty']:.2f} validated={obs['validated']}")
        return AgentVerdict("analyst", claim, cur, str(obs))


class Adversary:
    """Seek the branch most likely to refute 'current is best'."""

    def refute(self, env: VirtualCompiler) -> AgentVerdict | None:
        cur = env.current
        best: AgentVerdict | None = None
        best_score = -1.0
        for name in env.branches:
            if name == cur:
                continue
            cmp = env.compare(name, cur)
            if cmp["winner"] == name and cmp["basis"] == "oracle-runtime":
                return AgentVerdict("adversary",
                                    f"{name} beats {cur} on oracle runtime",
                                    name, str(cmp))
            score = float(cmp.get("score_a", 0.0)) if cmp["winner"] == name else -1
            if cmp["winner"] == name and score > best_score:
                best_score = score
                best = AgentVerdict("adversary",
                                    f"{name} may beat {cur} (surrogate)",
                                    name, str(cmp))
        return best


class Validator:
    """Active learning: spend the next oracle view where it matters."""

    def __init__(self, rule: str = "uncertainty") -> None:
        self.rule = rule

    def next_to_validate(self, env: VirtualCompiler) -> str | None:
        choice = acquire(env, self.rule, only_unvalidated=True)
        if choice is not None:
            return choice
        cands = [(u, n) for n, u in env.most_uncertain()
                 if not env.branches[n].validated]
        if not cands:
            return None
        cands.sort(key=lambda t: -t[0])
        return cands[0][1]


class Critic:
    """Surrogate-trust scoring: flag high-stakes, low-evidence branches."""

    def review(self, env: VirtualCompiler,
               risk_threshold: float = 0.4) -> list[AgentVerdict]:
        out: list[AgentVerdict] = []
        for name, st in env.branches.items():
            if st.validated:
                continue
            unc = env.uncertainty(name)
            if unc >= risk_threshold:
                out.append(AgentVerdict(
                    "critic",
                    f"{name} is high-uncertainty ({unc:.2f}); "
                    "do not commit without validation",
                    name, f"uncertainty={unc:.3f}"))
        return out


def run_swarm(env: VirtualCompiler, oracle_rounds: int = 2,
              top_k: int = 3, rule: str = "uncertainty") -> dict:
    """Swarm loop: propose → optimize → analyze → refute → validate."""
    board = Blackboard()
    proposer, optimizer = Proposer(), Optimizer()
    analyst, adversary = Analyst(), Adversary()
    validator, critic = Validator(rule), Critic()
    for _ in range(max(1, oracle_rounds)):
        cands = proposer.propose(env)
        res = optimizer.run(env, top_k=top_k)
        board.post(AgentVerdict(
            "optimizer",
            f"screened {res.virtual_calls} virtual, "
            f"{res.oracle_calls} oracle; best={res.best_branch}",
            res.best_branch))
        board.post(analyst.interpret(env))
        for flag in critic.review(env):
            board.post(flag)
        ref = adversary.refute(env)
        if ref is not None:
            board.post(ref)
            nxt = validator.next_to_validate(env)
            if nxt is not None and env.oracle.used < env.oracle_budget:
                try:
                    env.validate(nxt)
                    board.validations_spent += 1
                    board.post(AgentVerdict("validator",
                                            f"validated {nxt}", nxt))
                except (RuntimeError, OSError):
                    pass
        else:
            break
    _ = cands
    return {"current": env.current, "log": board.verdicts,
            "oracle_used": env.oracle.used,
            "consensus": board.consensus()}
