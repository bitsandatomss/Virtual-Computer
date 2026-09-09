"""Machine configuration + runtime control contracts (versioned, hashed).

Two levels, mirroring real practice and the DEA thesis:
- MicroarchConfig: DESIGN-time (tape-out) choices — widths, sizes, predictor,
  cache geometry. Changed by the Designer agent / DSE, never by the online policy.
- ControlBundle: RUN-time (control-plane) knobs — DVFS, prefetch, speculation,
  accelerator, fetch/schedule policy. Changed every interval by the controller.
  Hard correctness + thermal safety always preserved outside learned control.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum

from vmarch.version import VMARCH_VERSION


class PowerMode(str, Enum):
    ECO = "eco"
    BALANCED = "balanced"
    TURBO = "turbo"


class SpeculationMode(str, Enum):
    OFF = "off"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class AcceleratorMode(str, Enum):
    OFF = "off"
    SELECTIVE = "selective"
    AGGRESSIVE = "aggressive"


class FetchPolicy(str, Enum):
    SEQUENTIAL = "sequential"   # strict in-order fetch
    PRIORITY_LOOP = "priority_loop"  # tiny loop-buffer boost (same correctness)


class SchedulePolicy(str, Enum):
    AGE_ORDERED = "age_ordered"  # oldest-ready-first (default, highest perf)
    FIFO_CLASS = "fifo_class"    # per-class FIFO (weaker, for ablations)


@dataclass(frozen=True)
class ControlBundle:
    name: str
    power_mode: PowerMode
    prefetch_degree: int
    speculation_mode: SpeculationMode
    accelerator_mode: AcceleratorMode
    fetch_policy: FetchPolicy = FetchPolicy.SEQUENTIAL
    schedule_policy: SchedulePolicy = SchedulePolicy.AGE_ORDERED

    def __post_init__(self) -> None:
        if self.prefetch_degree not in (0, 1, 2, 4):
            raise ValueError("prefetch_degree must be one of 0,1,2,4")

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in d.items()}


ACTION_LIBRARY: dict[str, ControlBundle] = {
    "balanced": ControlBundle("balanced", PowerMode.BALANCED, 1, SpeculationMode.CONSERVATIVE, AcceleratorMode.SELECTIVE),
    "latency": ControlBundle("latency", PowerMode.TURBO, 2, SpeculationMode.AGGRESSIVE, AcceleratorMode.AGGRESSIVE),
    "streaming": ControlBundle("streaming", PowerMode.TURBO, 4, SpeculationMode.CONSERVATIVE, AcceleratorMode.AGGRESSIVE),
    "irregular": ControlBundle("irregular", PowerMode.BALANCED, 0, SpeculationMode.OFF, AcceleratorMode.OFF),
    "control": ControlBundle("control", PowerMode.BALANCED, 0, SpeculationMode.AGGRESSIVE, AcceleratorMode.OFF),
    "thermal_recovery": ControlBundle("thermal_recovery", PowerMode.ECO, 0, SpeculationMode.OFF, AcceleratorMode.OFF),
}
BUNDLE_NAMES = sorted(ACTION_LIBRARY)


@dataclass(frozen=True)
class MicroarchConfig:
    """Design-time microarchitecture. All fields validated; hash identifies it."""
    name: str = "baseline"
    fetch_width: int = 4
    decode_width: int = 2
    issue_width: int = 2
    commit_width: int = 2
    rob_size: int = 48
    rs_size: int = 16
    lsq_size: int = 12
    phys_regs: int = 64
    l1i_lines: int = 16
    l1d_lines: int = 32
    l1_assoc: int = 4
    l2_lines: int = 128
    l2_assoc: int = 8
    line_words: int = 4
    mshr_entries: int = 8
    predictor: str = "tournament"   # none|bimodal|gshare|tournament|tage_lite
    btb_entries: int = 64
    mem_words: int = 4096
    ambient: float = 30.0
    thermal_limit: float = 42.0
    thermal_target: float = 38.0
    thermal_heating: float = 0.11
    thermal_cooling: float = 0.018
    max_cycles: int = 60_000
    trace: bool = False  # per-line memory event log (T3); zero overhead when off
    vmarch_version: str = VMARCH_VERSION

    def __post_init__(self) -> None:
        for f in ("fetch_width", "decode_width", "issue_width", "commit_width",
                  "rob_size", "rs_size", "lsq_size", "phys_regs"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be positive")
        if self.predictor not in ("none", "bimodal", "gshare", "tournament", "tage_lite"):
            raise ValueError(f"unknown predictor {self.predictor}")
        if self.thermal_target > self.thermal_limit:
            raise ValueError("thermal target cannot exceed limit")

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def design_points(cls) -> dict[str, "MicroarchConfig"]:
        """Named tape-out candidates for DSE / cross-config generalization."""
        return {
            "small": cls("small", fetch_width=2, decode_width=1, issue_width=1,
                         commit_width=1, rob_size=16, rs_size=6, lsq_size=6,
                         phys_regs=40, l1d_lines=16, l2_lines=64, mshr_entries=4,
                         predictor="bimodal", btb_entries=32),
            "baseline": cls(),
            "wide": cls("wide", fetch_width=6, decode_width=4, issue_width=4,
                        commit_width=4, rob_size=96, rs_size=32, lsq_size=24,
                        phys_regs=96, l1d_lines=64, l2_lines=256, mshr_entries=16,
                        predictor="tage_lite", btb_entries=128),
            "mem_heavy": cls("mem_heavy", l1d_lines=64, l2_lines=256, mshr_entries=16,
                             lsq_size=24, predictor="gshare"),
            "branch_heavy": cls("branch_heavy", predictor="tage_lite",
                                btb_entries=128, rob_size=96, rs_size=32),
        }


POWER_CHARACTERISTICS = {
    PowerMode.ECO: {"width_scale": 0.5, "latency_scale": 1.25, "energy_scale": 0.76, "leakage": 0.06},
    PowerMode.BALANCED: {"width_scale": 1.0, "latency_scale": 1.0, "energy_scale": 1.0, "leakage": 0.10},
    PowerMode.TURBO: {"width_scale": 1.5, "latency_scale": 0.78, "energy_scale": 1.34, "leakage": 0.18},
}

# Energy per microarchitectural event (normalized proxies, not joules).
ENERGY = {
    "fetch": 0.20, "decode": 0.10, "rename": 0.15, "issue": 0.20, "commit": 0.10,
    "alu": 1.0, "mul": 4.0, "div": 7.0, "vec": 5.0, "vec_accel": 2.8, "accel_setup": 1.0,
    "branch": 1.5, "mispredict": 2.0, "predictor_access": 0.05, "btb_access": 0.03,
    "l1_hit": 0.4, "l2_hit": 1.5, "dram": 6.0, "prefetch": 0.5, "lsq_fwd": 0.2,
}


class SafetyGovernor:
    """Hard constraints outside learned/heuristic control (checked every cycle)."""

    def apply(self, action: ControlBundle, headroom: float) -> tuple[ControlBundle, list[str]]:
        if headroom <= 0.0:
            return ACTION_LIBRARY["thermal_recovery"], ["thermal_limit"]
        if headroom < 3.0 and action.power_mode is PowerMode.TURBO:
            return replace(action, name=f"{action.name}:guarded",
                           power_mode=PowerMode.BALANCED), ["turbo_headroom"]
        return action, []
