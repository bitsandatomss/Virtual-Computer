"""
validator/core.py — The deterministic safety authority.

Layered-defence contract: action *schemas* stay permissive so violations are
visible here; this module owns every numeric/semantic bound. Nothing reaches
the actuator without passing through it (the actuator re-checks the approval
flag as a second line of defence).
"""
from __future__ import annotations

from ..policy.schemas import PolicyAction, KernelIntermediateRepresentation

# Algorithms safe to write to net.ipv4.tcp_congestion_control. Anything not
# listed is rejected — this also blocks newline/semicolon sysctl-injection
# strings from ever reaching a command surface.
ALLOWED_CONGESTION_CONTROL = frozenset({"cubic", "bbr", "reno", "westwood"})


class ValidatorError(Exception):
    pass


class PolicyValidator:
    """Deterministic verifier: no destabilising parameter reaches the kernel."""

    def __init__(self):
        self.MIN_TARGET_LATENCY_US = 1000
        self.MAX_TARGET_LATENCY_US = 24000
        self.MIN_WAKEUP_GRANULARITY_US = 100
        self.MAX_MIGRATION_COST_NS = 500000
        self.MAX_QUEUE_LIMIT = 5000
        self.MIN_SWAPPINESS = 0
        self.MAX_SWAPPINESS = 100

    def validate_action(self, kstate: KernelIntermediateRepresentation, action: PolicyAction) -> bool:
        # 1. SUBSYSTEM-LOCAL CHECKS ────────────────────────────────────────
        if action.scheduler:
            s = action.scheduler
            if s.target_latency_us is not None:
                lat = s.target_latency_us
                if lat < self.MIN_TARGET_LATENCY_US or lat > self.MAX_TARGET_LATENCY_US:
                    raise ValidatorError(
                        f"[Safety Violation] target_latency_us {lat} is outside safe operating bands "
                        f"[{self.MIN_TARGET_LATENCY_US}, {self.MAX_TARGET_LATENCY_US}].")

            if s.wakeup_granularity_us is not None:
                g = s.wakeup_granularity_us
                if g < self.MIN_WAKEUP_GRANULARITY_US:
                    raise ValidatorError(
                        f"[Safety Violation] wakeup_granularity_us {g} below floor "
                        f"{self.MIN_WAKEUP_GRANULARITY_US} would cause scheduler thrash.")
                if s.target_latency_us is not None and g > s.target_latency_us:
                    raise ValidatorError(
                        "[Safety Violation] wakeup_granularity_us cannot exceed target_latency_us.")

            if s.migration_cost_ns is not None:
                cost = s.migration_cost_ns
                if cost < 0 or cost > self.MAX_MIGRATION_COST_NS:
                    raise ValidatorError(
                        f"[Safety Violation] migration_cost_ns {cost} could lead to thrashing.")

        if action.vm:
            v = action.vm
            if v.swappiness is not None and not (
                    self.MIN_SWAPPINESS <= v.swappiness <= self.MAX_SWAPPINESS):
                raise ValidatorError(
                    f"[Safety Violation] VM Swappiness {v.swappiness} out of bounds "
                    f"[{self.MIN_SWAPPINESS}, {self.MAX_SWAPPINESS}].")
            if v.dirty_ratio is not None and not (0 <= v.dirty_ratio <= 100):
                raise ValidatorError(f"[Safety Violation] VM dirty_ratio {v.dirty_ratio} out of bounds.")
            if v.dirty_background_ratio is not None and not (0 <= v.dirty_background_ratio <= 100):
                raise ValidatorError(
                    f"[Safety Violation] VM dirty_background_ratio {v.dirty_background_ratio} out of bounds.")
            if (v.dirty_ratio is not None and v.dirty_background_ratio is not None
                    and v.dirty_background_ratio > v.dirty_ratio):
                raise ValidatorError(
                    "[Safety Violation] dirty_background_ratio cannot exceed dirty_ratio.")

        if action.network:
            n = action.network
            if n.congestion_control is not None:
                cc = n.congestion_control.strip().lower()
                if cc not in ALLOWED_CONGESTION_CONTROL:
                    raise ValidatorError(
                        f"[Safety Violation] congestion control '{n.congestion_control}' "
                        f"not in allowlist {sorted(ALLOWED_CONGESTION_CONTROL)}.")
            if n.queue_limit is not None and n.queue_limit <= 0:
                raise ValidatorError("[Safety Violation] Network backlog must be strictly positive.")
            if n.queue_limit is not None and n.queue_limit > self.MAX_QUEUE_LIMIT:
                raise ValidatorError(
                    f"[Safety Violation] Network backlog exceeds the configured safe ceiling "
                    f"({self.MAX_QUEUE_LIMIT}).")

        # 2. JOINT MULTI-SUBSYSTEM CHECKS ──────────────────────────────────
        if action.network and action.network.queue_limit is not None:
            if action.network.queue_limit >= self.MAX_QUEUE_LIMIT:
                # A maximum-size queue is safe only with fresh, non-critical
                # memory telemetry. Missing telemetry ≠ healthy memory.
                if kstate.memory is None or kstate.memory.pressure > 0.8:
                    raise ValidatorError(
                        "[Joint Safety Violation] Cannot expand TCP/Net queues to the maximum "
                        "without non-critical VM memory telemetry.")

        return True
