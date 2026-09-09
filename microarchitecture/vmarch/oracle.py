"""Exact-oracle facade: determinism, versioning, feature/result contracts.

The oracle is the context.txt "exact rules" (cf. chess rules as perfect
verification): every surrogate claim is checkable against it. Guarantees:
- deterministic: same (program, config, control plan, mem/regs) -> same result
- precise: stores/branches retire in order; wrong-path stores never visible
- snapshotable: fork() gives bit-identical counterfactual worlds
- versioned: config digest + VMARCH_VERSION recorded on every result
"""
from __future__ import annotations

from typing import Any

from vmarch import isa
from vmarch.config import ACTION_LIBRARY, ControlBundle, MicroarchConfig
from vmarch.core import FEATURE_NAMES, Core
from vmarch.version import FEATURE_VERSION, VMARCH_VERSION


class OracleSim:
    """Thin owned-core wrapper with budget accounting and contracts."""

    def __init__(self, program: list[isa.Instruction], cfg: MicroarchConfig | None = None,
                 mem_init=None, reg_init=None, control: str | ControlBundle = "balanced") -> None:
        bundle = ACTION_LIBRARY[control] if isinstance(control, str) else control
        self.core = Core(program, cfg=cfg, mem_init=mem_init, reg_init=reg_init, control=bundle)
        self.oracle_calls = 0  # counts committed oracle intervals (budget unit)

    @classmethod
    def _from_core(cls, core: Core) -> "OracleSim":
        o = cls.__new__(cls)
        o.core = core
        o.oracle_calls = 0
        return o

    # -- environment verbs --
    def observe(self) -> dict[str, Any]:
        o = self.core.observe()
        o["vmarch_version"] = VMARCH_VERSION
        o["feature_version"] = FEATURE_VERSION
        o["config_digest"] = self.core.cfg.digest()
        return o

    def features(self) -> list[float]:
        return self.core.feature_vector()

    def perturb(self, control: str | ControlBundle) -> list[str]:
        b = ACTION_LIBRARY[control] if isinstance(control, str) else control
        return self.core.set_control(b)

    def step(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            if self.core.halted:
                break
            self.core.step_cycle()

    def run(self, plan: list[str | ControlBundle] | None = None, interval: int = 8) -> dict:
        bundles = None
        if plan:
            bundles = [ACTION_LIBRARY[p] if isinstance(p, str) else p for p in plan]
        res = self.core.run(bundles, interval=interval)
        res["vmarch_version"] = VMARCH_VERSION
        return res

    def result(self) -> dict:
        res = self.core.result()
        res["vmarch_version"] = VMARCH_VERSION
        return res

    # -- worlds --
    def snapshot(self) -> dict:
        return self.core.snapshot()

    def restore(self, snap: dict) -> None:
        self.core.restore(snap)

    def fork(self) -> "OracleSim":
        return OracleSim._from_core(self.core.fork())

    @staticmethod
    def feature_names() -> list[str]:
        assert len(FEATURE_NAMES) == 24
        return list(FEATURE_NAMES)
