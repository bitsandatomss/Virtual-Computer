"""Branchable virtual environment: world tree over the exact oracle.

Implements context.txt lines 70-98 and 893-921 for computation: persistent
execution state, counterfactual branching with lineage, bounded rollouts that
never disturb the parent, measurement, comparison, uncertainty hooks, and
rival-hypothesis preservation (MCTS over hypotheses).

A World is a fork of the oracle at a point in (program, config, control)
space. The tree records parent/children, the intervention that created each
edge, and terminal measurements — the audit trail from virtual claim to
oracle validation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from vmarch import isa
from vmarch.config import ControlBundle, MicroarchConfig
from vmarch.oracle import OracleSim


class World:
    _next_id = 0

    def __init__(self, sim: OracleSim, parent: int | None, edge: str,
                 label: str = "") -> None:
        self.id = World._next_id
        World._next_id += 1
        self.sim = sim
        self.parent = parent
        self.edge = edge
        self.label = label
        self.children: list[int] = []
        self.terminal: dict | None = None

    def digest(self) -> str:
        o = self.sim.observe()
        payload = json.dumps({"pc": o["fetch_pc"], "cyc": o["cycle"],
                              "ret": o["retired"], "cfg": o["config_digest"],
                              "act": o["action"]}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class VirtualMicroarchitecture:
    """Executable virtual microarchitecture (the product's central object)."""

    def __init__(self, program: list[isa.Instruction], cfg: MicroarchConfig | None = None,
                 mem_init=None, reg_init=None, control: str | ControlBundle = "balanced") -> None:
        World._next_id = 0
        root = World(OracleSim(program, cfg, mem_init, reg_init, control),
                     parent=None, edge="root", label="root")
        self.program = program
        self.worlds: dict[int, World] = {0: root}
        self.active = 0
        self.oracle_intervals = 0  # budget-relevant oracle consumption

    # -- basic verbs --
    def observe(self, wid: int | None = None) -> dict[str, Any]:
        w = self.worlds[self.active if wid is None else wid]
        o = w.sim.observe()
        o["world"] = w.id
        return o

    def measure(self, wid: int | None = None) -> dict[str, Any]:
        w = self.worlds[self.active if wid is None else wid]
        m = w.sim.result()
        m["world"] = w.id
        return m

    def perturb(self, control: str | ControlBundle, wid: int | None = None) -> list[str]:
        w = self.worlds[self.active if wid is None else wid]
        return w.sim.perturb(control)

    def branch(self, label: str = "", wid: int | None = None,
               control: str | ControlBundle | None = None) -> int:
        """Fork a counterfactual world; optionally intervene on the edge."""
        src = self.worlds[self.active if wid is None else wid]
        child_sim = src.sim.fork()
        edge = "fork"
        if control is not None:
            child_sim.perturb(control)
            edge = f"do({control if isinstance(control, str) else control.name})"
        child = World(child_sim, parent=src.id, edge=edge, label=label)
        src.children.append(child.id)
        self.worlds[child.id] = child
        return child.id

    def step(self, cycles: int, wid: int | None = None) -> None:
        w = self.worlds[self.active if wid is None else wid]
        w.sim.step(cycles)

    def rollout(self, cycles: int, control: str | ControlBundle | None = None,
                wid: int | None = None, label: str = "rollout") -> dict[str, Any]:
        """Bounded hypothetical trajectory on a disposable fork (parent untouched)."""
        cid = self.branch(label=label, wid=wid, control=control)
        child = self.worlds[cid]
        child.sim.step(cycles)
        res = child.sim.result()
        res["world"] = cid
        res["halted"] = child.sim.core.halted
        return res

    def run_world(self, wid: int, plan: list | None = None, interval: int = 8) -> dict:
        w = self.worlds[wid]
        res = w.sim.run(plan, interval=interval)
        self.oracle_intervals += 1
        w.terminal = res
        res["world"] = wid
        return res

    def compare(self, a: int, b: int) -> dict[str, Any]:
        ma, mb = self.measure(a), self.measure(b)
        return {"cycles_delta": mb["cycles"] - ma["cycles"],
                "energy_delta": mb["energy"] - ma["energy"],
                "objective_delta": mb["objective"] - ma["objective"],
                "a": ma, "b": mb}

    def uncertainty(self, surrogate=None, wid: int | None = None) -> float:
        w = self.worlds[self.active if wid is None else wid]
        if surrogate is not None:
            return float(surrogate.uncertainty(w.sim.features()))
        o = w.sim.observe()
        return float(min(1.0, o["l1d_miss"] * 0.6 + o["br_miss"] * 0.8 +
                         max(0.0, -o["headroom"]) * 0.1))

    # -- tree / lineage / trajectories --
    def lineage(self, wid: int) -> list[dict]:
        chain = []
        w = self.worlds[wid]
        while w is not None:
            chain.append({"world": w.id, "edge": w.edge, "label": w.label})
            w = self.worlds.get(w.parent) if w.parent is not None else None
        return list(reversed(chain))

    def collect_dataset(self, wid: int, intervals: int, bundles: list[str],
                        interval: int = 8) -> list[tuple[list[float], str, float, list[float]]]:
        """Oracle trajectory chopped into (x, a, cost, x') transition tuples."""
        w = self.worlds[wid]
        out = []
        bi = 0
        while bi < intervals and not w.sim.core.halted:
            b = bundles[bi % len(bundles)]
            w.sim.perturb(b)
            x = w.sim.features()
            c0 = (w.sim.core.cycle, w.sim.core.energy_dyn, w.sim.core.thermal_exposure)
            w.sim.step(interval)
            c1 = (w.sim.core.cycle, w.sim.core.energy_dyn, w.sim.core.thermal_exposure)
            cost = (c1[0] - c0[0]) + 0.08 * (c1[1] - c0[1]) + 0.6 * (c1[2] - c0[2])
            out.append((x, b, cost, w.sim.features()))
            bi += 1
        self.oracle_intervals += 1
        return out

    def switch(self, wid: int) -> None:
        if wid not in self.worlds:
            raise ValueError(f"unknown world {wid}")
        self.active = wid
