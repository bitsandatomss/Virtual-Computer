"""Budgeted virtual screening: the four-system ladder as runnable code.

context.txt T2/T3: finite oracle budget B, unlimited surrogate calls, four
arms (exact search / learned-alone / MCTS+learned / stock baseline),
metric = strength per oracle interaction.

``virtual_screen`` (kept from v1) is beam-with-validation. The `Strategy`
classes make every rung of the ladder an experiment arm the killer
benchmark can run under identical budgets:

- ``stock``      — conventional baseline, no search (validates root only)
- ``random``     — exact search: uniform oracle spend, no surrogate
- ``surrogate``  — learned model alone: one-shot argmax, one validation
- ``beam``       — virtual expansion, surrogate ranking, top-k validation
- ``mcts``       — UCT-guided virtual rollouts, then top-k validation
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .acquisition import select as acquire
from .environment import FLAG_TUNE_POOL, VirtualCompiler
from .transforms import TRANSFORMS

DEFAULT_PRESETS: list[tuple[str, str]] = [
    ("flags", "O0"), ("flags", "O1"), ("flags", "O2"),
    ("flags", "O3"), ("flags", "Os"),
    ("flags", "O2+lto"), ("flags", "O3+native"),
    ("policy", "no-evrp"), ("policy", "no-ccp"),
]

# Full search space: presets (1-ply) + combinatorial tune actions (real
# depth — subsets compose, so trajectories are not redundant revisits)
# + source transforms (code-level interventions, WHAT_IT_IS §7).
TUNE_ACTIONS: list[tuple[str, str]] = (
    [("tune", f"+{f}") for f in FLAG_TUNE_POOL]
    + [("tune", f"-{f}") for f in FLAG_TUNE_POOL]
)
XFORM_ACTIONS: list[tuple[str, str]] = [
    ("xform", name) for name in TRANSFORMS
]
SEARCH_SPACE: list[tuple[str, str]] = DEFAULT_PRESETS + TUNE_ACTIONS + XFORM_ACTIONS


@dataclass
class SearchResult:
    best_branch: str
    best_runtime_ms: float | None
    oracle_calls: int
    virtual_calls: int
    evaluated: list[str] = field(default_factory=list)
    strategy: str = "beam"


def virtual_screen(env: VirtualCompiler,
                   candidates: list[tuple[str, str]] | None = None,
                   top_k: int = 3,
                   objective: str = "runtime") -> SearchResult:
    """Expand candidates virtually, validate only the top_k."""
    candidates = candidates if candidates is not None else SEARCH_SPACE
    base = env.current
    virtual_calls = 0
    scored: list[tuple[float, str]] = []
    for action, arg in candidates:
        try:
            name = env.perturb(action, arg, branch=base,
                               new_branch=f"screen:{action}={arg}")
        except ValueError:
            continue
        if objective == "size":
            m = env.measure("size", name)
            val = -(m["value"] or 0.0)
        else:
            m = env.measure("build_ok", name)
            val = float(m["value"] or 0.0) - 0.5 * float(m["uncertainty"])
        virtual_calls += 1
        scored.append((val, name))
    scored.sort(key=lambda t: -t[0])
    env.current = base
    evaluated: list[str] = []
    best, best_rt = base, env.branches[base].runtime_ms
    for _, name in scored[:max(1, top_k)]:
        try:
            rec = env.validate(name)
        except (RuntimeError, OSError):
            continue
        evaluated.append(name)
        if rec.build_ok and rec.runtime_ms is not None:
            if best_rt is None or rec.runtime_ms < best_rt:
                best, best_rt = name, rec.runtime_ms
    env.current = best
    return SearchResult(best_branch=best, best_runtime_ms=best_rt,
                        oracle_calls=len(evaluated),
                        virtual_calls=virtual_calls, evaluated=evaluated,
                        strategy="beam")


def gain_per_oracle(baseline_ms: float | None,
                    best_ms: float | None, oracle_calls: int) -> float | None:
    """The killer metric: improvement per real interaction."""
    if baseline_ms is None or best_ms is None or not oracle_calls:
        return None
    return (baseline_ms - best_ms) / baseline_ms / oracle_calls


class Strategy:
    name = "base"

    def run(self, env: VirtualCompiler, top_k: int = 3,
            objective: str = "runtime", seed: int = 0) -> SearchResult:
        raise NotImplementedError


class StockStrategy(Strategy):
    """Conventional baseline: validate the incoming configuration only."""
    name = "stock"

    def run(self, env, top_k=3, objective="runtime", seed=0):
        used_before = env.oracle.used
        try:
            env.validate(env.current)
            evaluated = [env.current]
        except (RuntimeError, OSError):
            evaluated = []
        st = env.branches[env.current]
        return SearchResult(env.current, st.runtime_ms,
                            env.oracle.used - used_before, 0,
                            evaluated, self.name)


class RandomStrategy(Strategy):
    """Exact search without learning: uniform random oracle spend."""
    name = "random"

    def run(self, env, top_k=3, objective="runtime", seed=0):
        rng = random.Random(seed)
        base = env.current
        cands = list(SEARCH_SPACE)
        rng.shuffle(cands)
        evaluated: list[str] = []
        best, best_rt = base, env.branches[base].runtime_ms
        for action, arg in cands[:max(1, top_k)]:
            try:
                name = env.perturb(action, arg, branch=base,
                                   new_branch=f"random:{action}={arg}")
                rec = env.validate(name)
            except (ValueError, RuntimeError, OSError):
                continue
            evaluated.append(name)
            if rec.build_ok and rec.runtime_ms is not None:
                if best_rt is None or rec.runtime_ms < best_rt:
                    best, best_rt = name, rec.runtime_ms
        env.current = best
        return SearchResult(best, best_rt, len(evaluated), 0,
                            evaluated, self.name)


class SurrogateOnlyStrategy(Strategy):
    """Learned model alone: one-shot argmax, single validation."""
    name = "surrogate"

    def run(self, env, top_k=3, objective="runtime", seed=0):
        res = virtual_screen(env, top_k=1, objective=objective)
        res.strategy = self.name
        return res


class BeamStrategy(Strategy):
    """Full virtual screen: expand all, rank, validate top-k."""
    name = "beam"

    def run(self, env, top_k=3, objective="runtime", seed=0):
        return virtual_screen(env, top_k=top_k, objective=objective)


class MCTSStrategy(Strategy):
    """UCT over virtual trajectories, then acquisition-driven validation.

    Selection/expansion/simulation happen against the surrogate (free);
    only the acquisition-ranked leaves spend oracle budget. This is the
    direct analogue of "MCTS + learned model" from context.txt T2 — and
    the ablation ``mcts`` vs ``beam`` answers T5's "does MCTS compensate
    for model error?".
    """
    name = "mcts"

    def __init__(self, simulations: int = 60,
                 exploration: float = 1.4) -> None:
        self.simulations = simulations
        self.exploration = exploration

    def run(self, env, top_k=3, objective="runtime", seed=0):
        rng = random.Random(seed)
        base = env.current
        preexisting = set(env.branches)
        # -- phase 1: UCT over virtual children of base -------------------
        children: list[str] = []
        for action, arg in SEARCH_SPACE:
            try:
                children.append(env.perturb(
                    action, arg, branch=base,
                    new_branch=f"mcts:{action}={arg}"))
            except ValueError:
                continue
        virtual_calls = len(children)
        visits = {c: 0 for c in children}
        values = {c: 0.0 for c in children}
        parent_visits = 0
        for _ in range(self.simulations):
            # UCT select
            def uct(c: str) -> float:
                if visits[c] == 0:
                    return float("inf")
                return (values[c] / visits[c] + self.exploration
                        * math.sqrt(math.log(parent_visits + 1) / visits[c]))

            pick = max(children, key=uct)
            # simulate: 1-2 random virtual steps from pick, score leaf
            leaf = pick
            for _ in range(rng.randint(0, 2)):
                a, g = rng.choice(SEARCH_SPACE)
                try:
                    leaf = env.perturb(a, g, branch=leaf)
                    virtual_calls += 1
                except ValueError:
                    break
            m = env.measure("build_ok", leaf)
            reward = float(m["value"] or 0.0) - 0.5 * float(m["uncertainty"])
            visits[pick] += 1
            values[pick] += reward
            parent_visits += 1
        ranked = sorted(children,
                        key=lambda c: values[c] / max(1, visits[c]),
                        reverse=True)
        # -- phase 2: acquisition-driven validation of top leaves ---------
        env.current = base
        evaluated: list[str] = []
        best, best_rt = base, env.branches[base].runtime_ms
        shortlist = ranked[:max(1, top_k)]
        # re-rank shortlist by acquisition (active learning meets search)
        shortlist = sorted(
            shortlist,
            key=lambda n: -(_acq(env, n) if _acq(env, n) is not None else -1))
        for name in shortlist:
            try:
                rec = env.validate(name)
            except (RuntimeError, OSError):
                continue
            evaluated.append(name)
            if rec.build_ok and rec.runtime_ms is not None:
                if best_rt is None or rec.runtime_ms < best_rt:
                    best, best_rt = name, rec.runtime_ms
        env.current = best
        # prune non-validated virtual leaves to keep the DAG legible
        for name in [n for n in env.branches if n not in preexisting]:
            if name not in evaluated and name != env.current:
                if not env.branches[name].validated:
                    del env.branches[name]
        return SearchResult(best, best_rt, len(evaluated), virtual_calls,
                            evaluated, self.name)


def _acq(env: VirtualCompiler, name: str) -> float | None:
    try:
        from .acquisition import score_branch
        return float(score_branch(env, name, "expected-improvement"))
    except Exception:
        return None


class SwarmStrategy(Strategy):
    """Budget-capped swarm as an ablation arm (CRITIQUE G5).

    Lets the benchmark answer "does the swarm beat beam at equal oracle
    budget?" instead of asserting it.
    """
    name = "swarm"

    def run(self, env, top_k=3, objective="runtime", seed=0):
        from .agents import run_swarm

        _ = objective
        before = set(env.branches)
        used_before = env.oracle.used
        out = run_swarm(env, oracle_rounds=2, top_k=top_k)
        created = len(set(env.branches) - before)
        st = env.branches[env.current]
        _ = seed
        return SearchResult(env.current, st.runtime_ms,
                            env.oracle.used - used_before, created,
                            [v.branch for v in out["log"]
                             if v.agent == "validator"], self.name)


STRATEGIES: dict[str, Strategy] = {
    "stock": StockStrategy(),
    "random": RandomStrategy(),
    "surrogate": SurrogateOnlyStrategy(),
    "beam": BeamStrategy(),
    "mcts": MCTSStrategy(),
    "swarm": SwarmStrategy(),
}


def run_strategy(env: VirtualCompiler, name: str, top_k: int = 3,
                 objective: str = "runtime", seed: int = 0) -> SearchResult:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy: {name!r}")
    _ = acquire  # acquisition seam used by mcts shortlist re-rank
    return STRATEGIES[name].run(env, top_k=top_k, objective=objective,
                                seed=seed)
