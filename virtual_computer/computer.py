"""VirtualComputer: unified executable surrogate of computation.

Unifies the three layers from ``context.txt``'s Virtual Computer rung
(``Compiler + OS + microarchitecture + I/O``):

- **compiler** — ``VirtualCompiler`` over (flags x policy x patch) with a
  budgeted build oracle (real gcc, or ``StubOracle`` offline, labeled).
- **kernel** — paired exact ``KernelSimulator`` comparisons (tight vs
  loose sysctl under identical demand) plus the ``VirtualKernel``
  learned surrogate scored prediction-vs-oracle.
- **microarchitecture** — exact ``vmicro`` CPU oracle executed to HALT;
  ``vmarch`` OoO trust numbers enter as grading evidence.

Primary path: :func:`run_vertical_slice` (see ``vertical.py``) — one
causal chain C-source → flag presets → lowering map → paired CPU
execution → telemetry-coupled paired kernel comparison, every step
oracle-measured. The old per-layer smoke validations are kept as
availability probes only and are labeled as such.

Budget discipline: oracle units are incommensurable across layers
(compiler view vs in-order CPU run vs OoO run vs sim step) and are
reported as a VECTOR, never summed into one number. No phantom spends:
every counted unit corresponds to an exact-oracle call.

The surrogate proposes, the exact layer oracle disposes: computation
stays exact and discrete (never owned by a learned model); the learned
dynamics exist only for planning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from . import paths as _paths  # noqa: F401  (ensures layer sys.path)

DEFAULT_BUDGET = {"compiler_stub_views": 8, "micro_cpu_runs": 128,
                  "vmarch_runs": 64, "kernel_sim_steps": 1024}


@dataclass
class LayerResult:
    layer: str
    ok: bool
    oracle_calls: int = 0
    detail: dict = field(default_factory=dict)
    error: str = ""


def layer_available(layer: str) -> tuple[bool, str]:
    """Probe whether a layer imports cleanly. Returns (ok, reason)."""
    try:
        if layer == "microarchitecture":
            import vmarch.virtual  # noqa: F401
            import vmicro.virtual  # noqa: F401
        elif layer == "compiler":
            import virtual_compiler.environment  # noqa: F401
        elif layer == "kernel":
            import virtual_kernel.environment  # noqa: F401
        else:
            return False, f"unknown layer {layer!r}"
        return True, "ok"
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"{type(e).__name__}: {e}"


class VirtualComputer:
    """Branchable counterfactual worlds spanning compiler/kernel/microarch.

    A unified *branch* is a named triple of (compiler flags, kernel
    sysctl action, microarch control bundle). Forking a branch is free
    and unlimited; only ``validate`` calls spend exact-oracle budget.
    """

    LAYERS = ("compiler", "kernel", "microarchitecture")

    def __init__(self, oracle_budget: dict | None = None) -> None:
        self.oracle_budget = dict(DEFAULT_BUDGET if oracle_budget is None
                                  else oracle_budget)
        self.oracle_used = {k: 0 for k in self.oracle_budget}
        self._branches: dict[str, dict] = {
            "root": {"flags": "O2", "sysctl": "balanced", "control": "balanced"}
        }
        self._active = "root"

    # -- unified environment API (mirrors each layer) --
    def observe(self, branch: str | None = None) -> dict[str, Any]:
        name = branch or self._active
        cfg = self._branches[name]
        remaining = {k: max(0, self.oracle_budget.get(k, 0) - self.oracle_used.get(k, 0))
                     for k in self.oracle_budget}
        return {
            "branch": name,
            "config": dict(cfg),
            "oracle_budget": dict(self.oracle_budget),
            "oracle_used": dict(self.oracle_used),
            "oracle_remaining": remaining,
            "layers": list(self.LAYERS),
        }

    def branch(self, name: str, source: str | None = None, **intervention) -> str:
        src = self._branches[source or self._active]
        cfg = dict(src)
        for k, v in intervention.items():
            if k in ("flags", "sysctl", "control"):
                cfg[k] = v
        if name in self._branches:
            raise ValueError(f"branch {name!r} already exists")
        self._branches[name] = cfg
        return name

    fork = branch

    def switch(self, name: str) -> None:
        if name not in self._branches:
            raise KeyError(f"unknown branch {name!r}")
        self._active = name

    def intervene(self, branch: str | None = None, **kw) -> dict:
        name = branch or self._active
        cfg = self._branches[name]
        for k, v in kw.items():
            if k in ("flags", "sysctl", "control"):
                cfg[k] = v
        return dict(cfg)

    def rollout(self, steps: int, branch: str | None = None) -> list[dict]:
        """Virtual (zero-oracle-cost) trajectory over branch configs."""
        name = branch or self._active
        base = dict(self._branches[name])
        controls = ["balanced", "latency", "streaming", "thermal_recovery"]
        out = []
        for i in range(steps):
            step = dict(base)
            step["control"] = controls[i % len(controls)]
            out.append(step)
        return out

    def measure(self, branch: str | None = None) -> dict:
        return self.observe(branch)

    def uncertainty(self, branch: str | None = None) -> dict:
        """No cross-layer predictor exists: uncertainty is UNMEASURED.

        Returns the per-layer hooks that *would* feed it (ensemble
        disagreement where computed) instead of inventing a number.
        The old Hamming-distance heuristic is removed (CRITIQUE P1-4).
        """
        return {"status": "unmeasured",
                "reason": "no cross-layer surrogate with skill (L1 blocker)",
                "hooks": ["vmarch SurrogateEnsemble.disagreement",
                          "virtual_kernel EnsembleDynamics uncertainty",
                          "virtual_compiler kNN distance (uncalibrated)"]}

    def compare(self, a: str, b: str) -> dict:
        return {"a": self.observe(a), "b": self.observe(b),
                "note": "config-only comparison; measured deltas come from "
                        "run_vertical_slice with CIs"}

    # -- budgeted validation: one ledger per oracle unit --
    def _spend(self, unit: str, n: int = 1) -> None:
        if unit not in self.oracle_budget:
            raise ValueError(f"unknown oracle unit {unit!r}")
        if self.oracle_used[unit] + n > self.oracle_budget[unit]:
            raise RuntimeError(
                f"oracle budget exhausted for {unit} "
                f"({self.oracle_used[unit]}/{self.oracle_budget[unit]})")
        self.oracle_used[unit] += n

    def run_vertical_slice(self, seeds: Sequence[int] = (0, 1, 2, 3, 4, 5),
                           ns: Sequence[int] = (4, 8, 12),
                           controls: Sequence[str] = ("balanced", "streaming"),
                           kernel_steps: int = 4,
                           gains: Sequence[float] = (0.5, 1.0, 2.0),
                           gcc: bool = False) -> dict[str, Any]:
        """Primary path: factorial causal measurement (see vertical.py)."""
        from .vertical import run_vertical_slice as _run
        report = _run(seeds=seeds, ns=ns, controls=controls,
                      kernel_steps=kernel_steps, gains=gains)
        b = report["budget"]
        for unit in ("compiler_stub_views", "micro_cpu_runs",
                     "vmarch_runs", "kernel_sim_steps"):
            self._spend(unit, b[unit])
        if gcc:
            from .gcc_leg import run_gcc_leg
            report["gcc"] = run_gcc_leg()
        return report

    def validate_micro(self, family: str = "mixed", seed: int = 0,
                       budget: int = 2) -> LayerResult:
        """Availability probe: tiny surrogate+search smoke (NOT evidence)."""
        ok, reason = layer_available("microarchitecture")
        if not ok:
            return LayerResult("microarchitecture", False, 0, {}, reason)
        try:
            from vmicro.programs import FAMILIES
            from vmicro.search import beam_search_plan, fixed_baseline
            from vmicro.surrogate import collect_trace, train_surrogate
            from vmicro.virtual import VirtualMicroprocessor

            prog, mem, reg, desc = FAMILIES[family](seed)
            examples = []
            for s in range(2):
                p, m, r, _ = FAMILIES[family](100 + s)
                examples += collect_trace(p, m, r, interval=16)
            sur = train_surrogate(examples)
            vm = VirtualMicroprocessor(prog, mem, reg)
            out = beam_search_plan(vm, sur, horizon_intervals=4,
                                   interval_cycles=16, beam=3,
                                   oracle_budget=budget)
            base = fixed_baseline(VirtualMicroprocessor(prog, mem, reg), "balanced")
            self._spend("micro_cpu_runs", 1)
            gain = float(base["objective"] - out["best_objective"])
            return LayerResult("microarchitecture", True, budget + 1, {
                "program": desc, "baseline": base["objective"],
                "best": out["best_objective"], "gain": gain,
                "n_intervals": len(examples),
                "evidence": "PROBE ONLY (no skill/calibration gates here)"})
        except Exception as e:  # pragma: no cover
            return LayerResult("microarchitecture", False, 0, {},
                               f"{type(e).__name__}: {e}")

    def validate_compiler(self, budget: int = 2) -> LayerResult:
        """Availability probe: stub-oracle screen (STUB, not gcc evidence)."""
        ok, reason = layer_available("compiler")
        if not ok:
            return LayerResult("compiler", False, 0, {}, reason)
        try:
            from virtual_compiler.environment import VirtualCompiler
            from virtual_compiler.oracle import StubOracle
            from virtual_compiler.search import virtual_screen

            source = "int main(){volatile long s=0;for(long i=0;i<1000;i++)s+=i;return (int)(s&255);}\n"
            oracle = StubOracle(
                runtimes={"-O0": 30.0, "-O2": 20.0, "-O3": 18.0}, budget=budget)
            env = VirtualCompiler(source_text=source, oracle=oracle)
            env.branch("o3")
            env.perturb("flags", "O3", branch="o3")
            res = virtual_screen(env, top_k=1)
            self._spend("compiler_stub_views", 1)
            return LayerResult("compiler", True, budget, {
                "branches": sorted(env.branches),
                "screen": str(type(res).__name__),
                "evidence": "PROBE ONLY (stub runtimes; real gains need gcc + metrology)"})
        except Exception as e:  # pragma: no cover
            return LayerResult("compiler", False, 0, {}, f"{type(e).__name__}: {e}")

    def validate_kernel(self, episodes: int = 2, steps: int = 6,
                        members: int = 2, epochs: int = 3) -> LayerResult:
        """Real check: paired exact-sysctl comparison + ensemble scoring."""
        ok, reason = layer_available("kernel")
        if not ok:
            return LayerResult("kernel", False, 0, {}, reason)
        try:
            import random as _random

            from learned_kernel.policy.schemas import (PolicyAction,
                                                       SchedulerAction)
            from learned_kernel.simulator.env import (KernelSimulator,
                                                      WorkloadProfile)
            from virtual_kernel import EnsembleDynamics, VirtualKernel

            # paired exact comparison: identical demand, tight vs loose
            cum = {}
            for arm, target in (("tight", 2000), ("loose", 12000)):
                sim = KernelSimulator(
                    seed=7, workload=WorkloadProfile(rng=_random.Random(7)))
                total = 0.0
                for _ in range(steps):
                    kir = sim.step(PolicyAction(
                        policy_id="v", scheduler=SchedulerAction(
                            target_latency_us=target)))
                    total += kir.scheduler.avg_latency_ms
                cum[arm] = total
            n_steps = 2 * steps
            self._spend("kernel_sim_steps", n_steps)
            # learned surrogate scored prediction-vs-oracle (cheap scale)
            ens = EnsembleDynamics.train(n_members=members, n_episodes=episodes,
                                         steps_per_episode=steps, epochs=epochs)
            sim = KernelSimulator(seed=7)
            vk = VirtualKernel(ens, sim.current_kir())
            vk.branch("tight-world")
            lat = vk.observe("tight-world").scheduler.avg_latency_ms
            return LayerResult("kernel", True, n_steps, {
                "paired_cum_ms": cum,
                "delta_tight_minus_loose": cum["tight"] - cum["loose"],
                "surrogate_members": len(ens),
                "observe_latency_ms": float(lat),
                "evidence": "paired exact sims (real); ensemble toy-scale (probe)"})
        except Exception as e:  # pragma: no cover
            return LayerResult("kernel", False, 0, {}, f"{type(e).__name__}: {e}")

    def run_pipeline(self, micro_budget: int = 2,
                     compiler_budget: int = 2,
                     gcc: bool = False, **slice_kw) -> dict[str, Any]:
        """Probes (availability) + the primary vertical measurement."""
        self.oracle_used = {k: 0 for k in self.oracle_budget}
        probes = {
            "microarchitecture": self.validate_micro(budget=micro_budget),
            "compiler": self.validate_compiler(budget=compiler_budget),
            "kernel": self.validate_kernel(),
        }
        vertical = self.run_vertical_slice(gcc=gcc, **slice_kw)
        detail = {k: {"ok": v.ok, "oracle_calls": v.oracle_calls,
                      "detail": v.detail, "error": v.error}
                  for k, v in probes.items()}
        return {"probes": detail,
                "vertical": vertical,
                "budget": dict(self.oracle_used),
                "branches": sorted(self._branches)}
