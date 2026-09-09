"""The virtual compilation environment (context.txt T8).

API mirrors the Virtual Cell sketch, transposed to computation:

    cc.observe()            latent + oracle status of current branch
    cc.perturb(...)         one intervention (flags / policy / edit)
    cc.branch(name)         fork a counterfactual world (free, unlimited)
    cc.rollout(steps)       sequential interventions z0 -> z1 -> z2 ...
    cc.measure(kind)        surrogate estimate (build_ok / runtime / size)
    cc.uncertainty()        where the surrogate is unreliable
    cc.compare(a, b)        score two branches without oracle cost
    cc.validate()           spend 1 oracle view (real measurement)
    cc.save()/load()        persist + replay the whole DAG

Interventions compose (T10: ``X0 -g1-> X1 -g2-> X2``). Branching keeps
competing hypotheses (``H1, H2``) alive instead of committing early.
Structure enforces T8's transition ``prediction → state transition →
interactive environment``: there is no public predict-bypass; the only
way to learn ground truth is ``validate()``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .evidence import EvidenceLog
from .oracle import Oracle, OracleBudgetExhausted, OracleResult
from .state import CompilationState
from .surrogate import Prediction, Surrogate

# Canonical intervention vocabulary (bounded, like the pass-gate grammar).
FLAG_ACTIONS: dict[str, tuple[str, ...]] = {
    "O0": ("-O0",),
    "O1": ("-O1",),
    "O2": ("-O2",),
    "O3": ("-O3",),
    "Os": ("-Os",),
    "O2+lto": ("-O2", "-flto"),
    "O3+native": ("-O3", "-march=native"),
    "O3+unroll": ("-O3", "-funroll-loops"),
}

POLICY_ACTIONS: dict[str, str] = {
    "no-evrp": "disable_pass=evrp",
    "no-ccp": "disable_pass=ccp",
    "no-cddce": "disable_pass=cddce",
}

# Combinatorial depth (CRITIQUE G4): presets are 1-ply *sets*; per-flag
# tune actions compose into subsets, so trajectories gain real depth and
# tree search has something to search. `-fno-X` is meaningful even when
# `-fX` is absent (it overrides -O2/-O3 defaults).
FLAG_TUNE_POOL: tuple[str, ...] = (
    "funroll-loops",
    "finline-functions",
    "fpeel-loops",
    "ftree-vectorize",
    "funswitch-loops",
    "fprefetch-loop-arrays",
    "fivopts",
    "finline-small-functions",
)


@dataclass
class ValidationRecord:
    branch: str
    build_ok: bool
    binary_size: int | None
    runtime_ms: float | None
    oracle_calls_used: int


class VirtualCompiler:
    """Branchable surrogate environment over one compilation task."""

    def __init__(self, source_text: str, test_cmd: str | None = None,
                 oracle_budget: int = 20,
                 surrogate: Surrogate | None = None,
                 compiler: str = "gcc",
                 oracle: Oracle | None = None,
                 evidence: EvidenceLog | None = None,
                 benchmark_runs: int = 3,
                 keep_dir: str | None = None) -> None:
        self.surrogate = surrogate or Surrogate()
        self.test_cmd = test_cmd
        self.compiler = compiler
        self.evidence = evidence or EvidenceLog()
        if oracle is not None:
            self.oracle = oracle
            self.oracle_budget = oracle.budget
        else:
            self.oracle = Oracle(compiler=compiler, test_cmd=test_cmd,
                                 benchmark_runs=benchmark_runs,
                                 budget=oracle_budget,
                                 evidence=self.evidence,
                                 keep_dir=keep_dir)
            self.oracle_budget = oracle_budget
        root = CompilationState(source_text=source_text, label="root")
        self.branches: dict[str, CompilationState] = {"root": root}
        self.current = "root"
        self.validations: list[ValidationRecord] = []
        self.evidence.emit("observe", "env-init",
                           branches=["root"],
                           oracle_budget=self.oracle_budget)

    @property
    def oracle_calls_used(self) -> int:
        return self.oracle.used

    # -- environment API ------------------------------------------------
    def observe(self, branch: str | None = None) -> dict:
        st = self.branches[branch or self.current]
        pred = self.surrogate.predict(st)
        self.evidence.emit("observe", "observe-branch",
                           branch=st.label, state_id=st.state_id,
                           validated=st.validated)
        return {
            "branch": st.label,
            "state_id": st.state_id,
            "flags": list(st.flags),
            "policy": st.policy_text,
            "validated": st.validated,
            "build_ok": st.build_ok,
            "runtime_ms": st.runtime_ms,
            "binary_size": st.binary_size,
            "surrogate": {
                "ok_prob": pred.ok_prob,
                "runtime_ms": pred.runtime_ms,
                "size_bytes": pred.size_bytes,
                "uncertainty": pred.uncertainty,
            },
            "oracle_used": self.oracle.used,
            "oracle_left": self.oracle.remaining,
        }

    def perturb(self, action: str, arg: str = "",
                branch: str | None = None,
                new_branch: str | None = None) -> str:
        """Apply one intervention; returns the (new) branch name.

        Actions: ``flags:<preset>``, ``policy:<preset|raw>``,
        ``tune:+<flag>`` / ``tune:-<flag>`` (add/remove one ``-f`` flag —
        composes: subsets, not presets), ``edit:<old>:::<new>``,
        ``append:<code>``. Pure in-memory transition ``F(z, a)`` — costs
        no oracle budget.
        """
        base_name = branch or self.current
        base = self.branches[base_name]
        flags, policy, source = base.flags, base.policy_text, base.source_text
        if action == "flags":
            if arg not in FLAG_ACTIONS:
                raise ValueError(f"unknown flags preset: {arg!r}")
            flags = FLAG_ACTIONS[arg]
        elif action == "policy":
            # Union semantics (CRITIQUE §6.6): repeated policy perturbs
            # compose pass-disable SUBSETS, so interactions are searchable.
            # `policy:none` clears back to stock.
            if arg in ("none", ""):
                policy = None
            else:
                new_rules = [
                    ln for ln in POLICY_ACTIONS.get(arg, arg).splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]
                if not new_rules:
                    raise ValueError(f"empty policy: {arg!r}")
                existing = [
                    ln for ln in (policy or "").splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]
                merged = existing + [r for r in new_rules
                                     if r not in existing]
                policy = "\n".join(merged)
        elif action == "tune":
            if len(arg) < 2 or arg[0] not in ("+", "-"):
                raise ValueError(
                    "tune arg must look like +funroll-loops/-funroll-loops")
            base_name_only = arg[1:]
            if base_name_only not in FLAG_TUNE_POOL:
                raise ValueError(f"unknown tune flag: {arg!r}")
            if arg[0] == "+":
                flag = "-" + base_name_only
            else:
                core = base_name_only[1:] if base_name_only.startswith("f") \
                    else base_name_only
                flag = "-fno-" + core
            negated = (("-fno-" + (base_name_only[1:]
                                   if base_name_only.startswith("f")
                                   else base_name_only))
                       if arg[0] == "+" else "-" + base_name_only)
            flags = tuple(f for f in flags if f != flag and f != negated)
            flags = (*flags, flag)
        elif action == "edit":
            old, _, new = arg.partition(":::")
            if not old or old not in source:
                raise ValueError("edit target not found in source")
            source = source.replace(old, new, 1)
        elif action == "append":
            source = source + ("\n" if not source.endswith("\n") else "") + arg + "\n"
        elif action == "xform":
            from .transforms import apply_transform, TRANSFORMS
            if arg not in TRANSFORMS:
                raise ValueError(f"unknown transform: {arg!r}")
            new_source = apply_transform(source, arg)
            if new_source is None:
                raise ValueError(f"transform {arg!r} inapplicable here")
            source = new_source
        else:
            raise ValueError(f"unknown action: {action!r}")
        name = self._unique_name(
            new_branch or f"{base_name}+{action}={arg[:24]}")
        self.branches[name] = CompilationState(
            source_text=source, flags=tuple(flags), policy_text=policy,
            label=name, parent=base_name, action=f"{action}:{arg[:60]}")
        self.current = name
        self.evidence.emit("intervene", "perturb",
                           from_branch=base_name, branch=name,
                           action=action, arg=arg[:120])
        self.evidence.emit("branch", "branch-created",
                           branch=name, parent=base_name)
        return name

    def branch(self, name: str, from_branch: str | None = None) -> str:
        src = self.branches[from_branch or self.current]
        name = self._unique_name(name)
        self.branches[name] = CompilationState(
            source_text=src.source_text, flags=src.flags,
            policy_text=src.policy_text, label=name,
            parent=src.label, action="branch")
        # same code+config == same truth: carry oracle facts
        self.branches[name].build_ok = src.build_ok
        self.branches[name].binary_size = src.binary_size
        self.branches[name].runtime_ms = src.runtime_ms
        self.branches[name].telemetry = dict(src.telemetry)
        self.branches[name].validated = src.validated
        self.current = name
        self.evidence.emit("branch", "branch-created",
                           branch=name, parent=src.label)
        return name

    def rollout(self, steps: Sequence[tuple[str, str]],
                branch: str | None = None) -> str:
        name = branch or self.current
        for action, arg in steps:
            name = self.perturb(action, arg, branch=name)
        return name

    def measure(self, kind: str = "runtime",
                branch: str | None = None) -> dict:
        st = self.branches[branch or self.current]
        pred = self.surrogate.predict(st)
        if kind == "build_ok":
            return {"value": pred.ok_prob, "uncertainty": pred.uncertainty}
        if kind == "runtime":
            return {"value": pred.runtime_ms, "uncertainty": pred.uncertainty}
        if kind == "size":
            return {"value": pred.size_bytes, "uncertainty": pred.uncertainty}
        raise ValueError(f"unknown measure kind: {kind!r}")

    def uncertainty(self, branch: str | None = None) -> float:
        return self.surrogate.predict(
            self.branches[branch or self.current]).uncertainty

    def most_uncertain(self) -> list[tuple[str, float]]:
        ranked = [(n, self.uncertainty(n)) for n in self.branches]
        ranked.sort(key=lambda t: -t[1])
        return ranked

    def compare(self, a: str, b: str) -> dict:
        pa: Prediction = self.surrogate.predict(self.branches[a])
        pb: Prediction = self.surrogate.predict(self.branches[b])
        score_a = pa.ok_prob - (pa.uncertainty * 0.5)
        score_b = pb.ok_prob - (pb.uncertainty * 0.5)
        va, vb = self.branches[a].validated, self.branches[b].validated
        if va and vb:
            ra, rb = self.branches[a].runtime_ms, self.branches[b].runtime_ms
            if ra is not None and rb is not None and ra > 0 and rb > 0:
                return {"winner": a if ra <= rb else b,
                        "basis": "oracle-runtime",
                        "runtime_a": ra, "runtime_b": rb}
        return {"winner": a if score_a >= score_b else b,
                "basis": "surrogate",
                "score_a": score_a, "score_b": score_b}

    # -- oracle ----------------------------------------------------------
    def check(self, branch: str | None = None) -> list[dict]:
        """Simulator-side constraint findings for one branch (N2/N3.2)."""
        from .constraints import check_state
        real_simulator = hasattr(self.oracle, "compiler")
        return check_state(
            self.branches[branch or self.current],
            self.oracle.compiler if real_simulator else "gcc",
            query_simulator=real_simulator)

    def validate(self, branch: str | None = None,
                 benchmark_runs: int = 3) -> ValidationRecord:
        """Spend 1 oracle view: real measurement of one branch."""
        name = branch or self.current
        st = self.branches[name]
        try:
            result: OracleResult = self.oracle.measure(st)
        except OracleBudgetExhausted as exc:
            raise RuntimeError(str(exc)) from exc
        _ = benchmark_runs  # runs are an oracle-construction parameter
        st.build_ok = result.build_ok
        st.binary_size = result.binary_size
        st.binary_sha256 = result.binary_sha256
        st.runtime_ms = result.runtime_ms
        st.telemetry = dict(result.telemetry)
        st.binary_path = result.kept_path
        st.validated = True
        self.surrogate.observe(st)  # learn stage: experience → model
        self.evidence.emit("learn", "surrogate-updated",
                           branch=name, experience=len(self.surrogate))
        rec = ValidationRecord(branch=name, build_ok=bool(st.build_ok),
                               binary_size=result.binary_size,
                               runtime_ms=result.runtime_ms,
                               oracle_calls_used=self.oracle.used)
        self.validations.append(rec)
        return rec

    # -- persistence / replay --------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "virtual-compiler.session.v1",
            "current": self.current,
            "oracle_used": self.oracle.used,
            "oracle_budget": self.oracle_budget,
            "branches": {n: s.to_dict() for n, s in self.branches.items()},
            "surrogate": self.surrogate.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")
        self.evidence.emit("represent", "session-saved",
                           path=str(path), branches=len(self.branches))

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "VirtualCompiler":
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        first = next(iter(rec["branches"].values()))
        env = cls(first["source_text"], oracle_budget=rec["oracle_budget"],
                  **kwargs)
        env.branches = {n: CompilationState.from_dict(s)
                        for n, s in rec["branches"].items()}
        env.current = rec["current"]
        env.oracle.used = rec.get("oracle_used", 0)
        for n, s in rec["branches"].items():
            if s.get("validated"):
                env.surrogate.observe(env.branches[n])
        return env

    # -- helpers ---------------------------------------------------------
    def _unique_name(self, want: str) -> str:
        if want not in self.branches:
            return want
        i = 2
        while f"{want}#{i}" in self.branches:
            i += 1
        return f"{want}#{i}"

    def lineage(self, branch: str | None = None) -> list[str]:
        out, cur = [], branch or self.current
        while cur is not None:
            out.append(cur)
            cur = self.branches[cur].parent
        return list(reversed(out))

    def diff_gate(self, branch: str | None = None,
                  parent: str | None = None,
                  trials: int = 8) -> dict:
        """Run the differential equivalence gate between two branches.

        Both must be validated with kept binaries. If the parent isn't
        specified, walks up the lineage to the nearest validated ancestor.
        Returns dict with 'equivalent', 'parent', 'candidate', 'reason'.
        """
        cand_name = branch or self.current
        cand = self.branches[cand_name]
        if parent is None:
            for anc in reversed(self.lineage(cand_name)):
                anc_st = self.branches[anc]
                if anc_st.validated and anc_st.binary_path and anc != cand_name:
                    parent = anc
                    break
        if parent is None:
            return {"equivalent": None, "parent": None,
                    "candidate": cand_name,
                    "reason": "no-validated-ancestor"}
        par = self.branches[parent]
        if not par.binary_path or not cand.binary_path:
            return {"equivalent": None, "parent": parent,
                    "candidate": cand_name,
                    "reason": "missing-binary"}
        try:
            from self_compiler.differential import run_differential
            ref = Path(par.binary_path)
            cand_p = Path(cand.binary_path)
            report = run_differential(str(ref), str(cand_p), trials=trials)
            equiv = bool(report.equivalent)
        except (OSError, ImportError) as exc:
            return {"equivalent": None, "parent": parent,
                    "candidate": cand_name,
                    "reason": f"differential-unavailable: {exc}"}
        cand.equivalent = equiv
        self.branches[cand_name].equivalent = equiv
        self.evidence.emit("validate", "diff-gate",
                           parent=parent, candidate=cand_name,
                           equivalent=equiv)
        return {"equivalent": equiv, "parent": parent,
                "candidate": cand_name, "reason": "ok"}
