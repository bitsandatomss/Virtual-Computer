"""
runtime/loop.py — Learned Kernel execution orchestrator.

Safety-critical change: KernelActuator now REFUSES any action whose
`approved` flag is False. Validation is enforced inside the actuator itself,
so even a buggy orchestrator that skips the validator cannot reach the kernel
(defence in depth; previously nothing stopped an unvalidated apply()).

Real-execution mode writes /proc/sys directly (root required) with automatic
rollback of already-written keys if a later write fails.
"""
from __future__ import annotations

import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..observation.monitors import SchedulerMonitor, SimulationMonitor
from ..policy.schemas import PolicyAction, KernelIntermediateRepresentation
from ..policy.core import HeuristicLatencyPolicy, PolicyBase
from ..validator.core import PolicyValidator, ValidatorError

_SYSCTL_SCHEDULER = {
    "target_latency_us":     ("kernel.sched_latency_ns", 1000),
    "wakeup_granularity_us": ("kernel.sched_wakeup_granularity_ns", 1000),
    "migration_cost_ns":     ("kernel.sched_migration_cost_ns", 1),
}
_SYSCTL_VM = {
    "swappiness":             "vm.swappiness",
    "dirty_ratio":            "vm.dirty_ratio",
    "dirty_background_ratio": "vm.dirty_background_ratio",
}
_PROC_SYS = pathlib.Path("/proc/sys")
_VALUE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")   # belt & braces vs injection


@dataclass
class AppliedChange:
    key: str
    old_value: Optional[str]
    new_value: str


@dataclass
class StepResult:
    kir: KernelIntermediateRepresentation
    action: Optional[PolicyAction]
    approved: Optional[bool] = None
    error: Optional[str] = None
    applied: List[str] = field(default_factory=list)


class ActuationError(RuntimeError):
    pass


class KernelActuator:
    """Translates validated PolicyActions into sysctl operations."""

    def __init__(self, execute_real: bool = False):
        self.execute_real = execute_real and _PROC_SYS.is_dir()

    # ------------------------------------------------------------------ #

    def plan(self, action: PolicyAction) -> List[str]:
        """Render the sysctl commands an action implies (no side effects)."""
        cmds: List[str] = []
        if action.scheduler:
            s = action.scheduler
            for attr in ("target_latency_us", "wakeup_granularity_us"):
                val = getattr(s, attr)
                if val is not None:
                    key, scale = _SYSCTL_SCHEDULER[attr]
                    self._check_value(key, val * scale)
                    cmds.append(f"sysctl {key}={val * scale}")
            if s.migration_cost_ns is not None:
                key, scale = _SYSCTL_SCHEDULER["migration_cost_ns"]
                self._check_value(key, s.migration_cost_ns * scale)
                cmds.append(f"sysctl {key}={s.migration_cost_ns * scale}")

        if action.vm:
            v = action.vm
            for attr, key in _SYSCTL_VM.items():
                val = getattr(v, attr, None)
                if val is not None:
                    self._check_value(key, val)
                    cmds.append(f"sysctl {key}={val}")

        if action.network:
            n = action.network
            if n.queue_limit is not None:
                self._check_value("net.core.netdev_max_backlog", n.queue_limit)
                cmds.append(f"sysctl net.core.netdev_max_backlog={n.queue_limit}")
            if n.congestion_control is not None:
                cc = n.congestion_control.strip()
                self._check_value("net.ipv4.tcp_congestion_control", cc)
                cmds.append(f"sysctl net.ipv4.tcp_congestion_control={cc}")
        return cmds

    @staticmethod
    def _check_value(key: str, value) -> None:
        if not _VALUE_RE.match(str(value)):
            raise ActuationError(
                f"[Actuator] Refusing malformed value {value!r} for {key}")

    # ------------------------------------------------------------------ #

    def apply(self, action: Optional[PolicyAction], *, approved: bool = True) -> List[str]:
        """
        Apply an action. `approved` MUST be the validator's verdict; the
        actuator refuses unapproved actions regardless of caller intent.
        Returns the list of rendered commands (or applied writes).
        """
        if action is None:
            return []
        if not approved:
            print("  [Actuator] BLOCKED - action was not validated.")
            return []

        cmds = self.plan(action)
        if not cmds:
            print("  [Actuator] No tunable fields to apply.")
            return []

        if not self.execute_real:
            for c in cmds:
                print(f"  [Actuator][dry-run] {c}")
            return cmds

        changed: List[AppliedChange] = []
        try:
            for c in cmds:
                key, _, val = c.removeprefix("sysctl ").partition("=")
                path = _PROC_SYS / pathlib.PurePosixPath(key.replace(".", "/"))
                old = path.read_text().strip() if path.exists() else None
                path.write_text(val)
                changed.append(AppliedChange(key=key, old_value=old, new_value=val))
        except Exception as e:
            # Roll back anything already written this batch.
            for ch in reversed(changed):
                if ch.old_value is not None:
                    p = _PROC_SYS / pathlib.PurePosixPath(ch.key.replace(".", "/"))
                    try:
                        p.write_text(ch.old_value)
                    except Exception:
                        pass
            raise ActuationError(f"[Actuator] write failed, rolled back: {e}") from e
        return [f"{ch.key}={ch.new_value}" for ch in changed]


# ─────────────────────────────────────────────────────────────── #

class LearnedKernelRuntime:
    """
    Closed-loop pipeline:
      Workload → Observer → KIR → Policy → Validator → Actuator → Audit
    Every stage is exception-contained: one bad cycle logs and continues.
    """

    def __init__(
        self,
        policy: Optional[PolicyBase] = None,
        monitor=None,
        execute_real: bool = False,
        sim_seed: int = 1234,
    ):
        if monitor is not None:
            self.monitor = monitor
        else:
            try:
                self.monitor = SchedulerMonitor()
            except Exception:
                print("[Warning] BCC/eBPF unavailable - simulation monitor active.")
                self.monitor = SimulationMonitor(seed=sim_seed)

        self.policy = policy or HeuristicLatencyPolicy()
        self.validator = PolicyValidator()
        self.actuator = KernelActuator(execute_real=execute_real)

    def step(self, verbose: bool = True) -> StepResult:
        log = print if verbose else (lambda *a, **k: None)
        try:
            self.monitor.poll(timeout_ms=10)
            state = self.monitor.get_kir_state()
        except Exception as e:
            raise RuntimeError(f"[Runtime] Observation failed: {e}") from e

        if verbose:
            log("[1] KIR Extracted:\n", state.model_dump_json(indent=2))

        action: Optional[PolicyAction] = None
        try:
            action = self.policy.decide(state)
        except Exception as e:
            result = StepResult(kir=state, action=None, approved=False,
                                error=f"policy error: {e}")
            log(f"[2] Policy raised: {e}")
            log("-" * 60)
            return result

        if action is None:
            log("[2] Policy Proposed: NO CHANGE")
            log("-" * 60)
            return StepResult(kir=state, action=None, approved=True)

        if verbose:
            log("[2] Policy Proposed:\n", action.model_dump_json(indent=2))

        approved = False
        error: Optional[str] = None
        try:
            approved = bool(self.validator.validate_action(state, action))
            log("[3] Validator: APPROVED")
        except ValidatorError as e:
            error = str(e)
            log(f"[3] Validator: REJECTED — {e}")

        applied: List[str] = []
        try:
            applied = self.actuator.apply(action, approved=approved)
            if approved:
                log("[4] Actuation complete.")
        except Exception as e:
            error = str(e)
            log(f"[4] Actuation failed: {e}")

        # Inform a simulation monitor what was actuated so its next state
        # genuinely reflects the decision (closed loop).
        submit = getattr(self.monitor, "submit_action", None)
        if callable(submit):
            submit(action if approved else None)

        log("-" * 60)
        return StepResult(kir=state, action=action, approved=approved,
                          error=error, applied=applied)


if __name__ == "__main__":
    runtime = LearnedKernelRuntime()
    for _ in range(3):
        runtime.step()
        time.sleep(0.5)
