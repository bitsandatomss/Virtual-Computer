"""Branchable virtual environment over the exact CPU.

API mirrors the context.txt Virtual Cell sketch, adapted to computation:
  cpu.observe() / cpu.measure() / cpu.branch() / cpu.rollout() /
  cpu.compare() / cpu.uncertainty()

Plus learn -> branch -> search -> act -> validate helpers used by search.py.
"""
from __future__ import annotations

from typing import Any

from vmicro.isa import Instruction
from vmicro.machine import ACTION_LIBRARY, CPU, ControlBundle


class VirtualMicroprocessor:
    """An executable, forkable view of a CPU execution.

    The exact CPU is the oracle. This wrapper adds:
    - persistent execution state with snapshot/restore
    - branching into counterfactual worlds (fork)
    - bounded rollouts without disturbing the parent
    - measurement, comparison, and surrogate-uncertainty hooks
    """

    def __init__(self, program: list[Instruction], mem_init=None, reg_init=None, control: str | ControlBundle = "balanced"):
        bundle = ACTION_LIBRARY[control] if isinstance(control, str) else control
        self.cpu = CPU(program, mem_init=mem_init, reg_init=reg_init, control=bundle)
        self.oracle_calls = 0
        self.surrogate_calls = 0

    @classmethod
    def _from_cpu(cls, cpu: CPU) -> "VirtualMicroprocessor":
        obj = cls.__new__(cls)
        obj.cpu = cpu
        obj.oracle_calls = 0
        obj.surrogate_calls = 0
        return obj

    # -- core environment API (context.txt: observe/branch/rollout/measure/compare) --
    def observe(self) -> dict[str, Any]:
        return self.cpu.observe()

    def measure(self, what: str = "all") -> dict[str, Any]:
        res = self.cpu.result()
        if what == "all":
            return res
        return {what: res[what]}

    def perturb(self, control: str | ControlBundle) -> list[str]:
        """Intervene on the computational state: switch the control bundle.

        Analogue of cell.perturb(type, target): changes how computation flows
        without rewriting program semantics (hard correctness preserved).
        """
        bundle = ACTION_LIBRARY[control] if isinstance(control, str) else control
        return self.cpu.set_control(bundle, controller_energy=0.01)

    def branch(self) -> "VirtualMicroprocessor":
        """Fork an independent counterfactual world (deep copy)."""
        child = VirtualMicroprocessor._from_cpu(self.cpu.fork())
        return child

    def snapshot(self) -> dict:
        return self.cpu.snapshot()

    def restore(self, snap: dict) -> None:
        self.cpu.restore(snap)

    def step(self, cycles: int = 1) -> list[dict]:
        out = []
        for _ in range(cycles):
            if self.cpu.halted:
                break
            out.append(self.cpu.step_cycle())
            self.oracle_calls += 1
        return out

    def rollout(self, cycles: int = 64, control: str | ControlBundle | None = None) -> dict:
        """Hypothetical trajectory from a fork; parent state untouched."""
        child = self.branch()
        if control is not None:
            child.perturb(control)
        child.step(cycles)
        result = child.cpu.result()
        result["final_pc"] = child.cpu.fetch_pc
        result["halted"] = child.cpu.halted
        self.oracle_calls += child.oracle_calls
        return result

    def compare(self, other: "VirtualMicroprocessor") -> dict[str, Any]:
        a, b = self.cpu.result(), other.cpu.result()
        return {
            "cycles_delta": b["cycles"] - a["cycles"],
            "energy_delta": b["energy"] - a["energy"],
            "objective_delta": b["objective"] - a["objective"],
            "a": a, "b": b,
        }

    def uncertainty(self, surrogate=None) -> float:
        """Ask the surrogate how uncertain the current region is.

        With no surrogate, falls back to a calibrated heuristic: high miss /
        mispredict rates and low headroom => unreliable region.
        """
        if surrogate is not None:
            self.surrogate_calls += 1
            return float(surrogate.uncertainty(self.cpu.feature_vector()))
        o = self.observe()
        return float(min(1.0, o["l1_miss_rate"] * 0.6 + o["branch_mispredict_rate"] * 0.8 + max(0.0, -o["headroom"]) * 0.1 + (0.2 if o["bubble"] else 0.0)))

    def run(self, plan: list[str | ControlBundle] | None = None, interval: int = 8) -> dict:
        bundles = None
        if plan:
            bundles = [ACTION_LIBRARY[p] if isinstance(p, str) else p for p in plan]
        return self.cpu.run(bundles, interval=interval)
