"""
observation/monitors.py — KIR producers.

Two implementations:

* SchedulerMonitor — attaches to sched:sched_switch via BCC and performs REAL
  per-CPU accounting in user space: busy-time deltas give utilisation, a
  recently-active-task set estimates runqueue depth. The previous revision
  returned hardcoded mocks (util=0.5) regardless of system load.

* SimulationMonitor — emits coherent synthetic KIR streams driven by the same
  dynamics as KernelSimulator, replacing the runtime's old fabricated constant
  state (util=0.88 forever). This is what non-Linux dev/demo runs use.

Semantic fix: `total_context_switches` is documented as "switches observed
since attach" (an event count), while reward consumes *deltas* of it — so any
monotone counter works; rates are derived from timestamp gaps.
"""
from __future__ import annotations

import pathlib
import time
from collections import OrderedDict
from typing import Dict, Optional

try:
    from bcc import BPF            # type: ignore
except ImportError:
    BPF = None

from ..policy.schemas import CPUMetrics, SchedulerState, KernelIntermediateRepresentation
from ..simulator.env import KernelSimulator


# ─────────────────────────────────────────────────────────────── #
# Real monitor (Linux + BCC only)                                  #
# ─────────────────────────────────────────────────────────────── #

class SchedulerMonitor:
    """eBPF-backed scheduler observation with genuine per-CPU accounting."""

    _TASK_TTL_S = 2.0          # a task counts toward runnable for this long after sight

    def __init__(self, bpf_source_path: Optional[str] = None):
        if BPF is None:
            raise RuntimeError("BCC library not installed or non-Linux environment.")

        if bpf_source_path is None:
            cdir = pathlib.Path(__file__).parent.resolve()
            bpf_source_path = str(cdir / "ebpf_scheduler.c")

        with open(bpf_source_path, "r", encoding="utf-8") as f:
            bpf_text = f.read()

        self.bpf = BPF(text=bpf_text)
        self.cpu_metrics_map = self.bpf.get_table("cpu_metrics")
        self.bpf["events"].open_perf_buffer(self._handle_event)

        self.total_switches = 0
        self._last_ts_ns: Dict[int, int] = {}
        self._busy_ns: Dict[int, int] = {}
        self._window_start_ns: Optional[int] = None
        # cpu -> OrderedDict[pid -> last_seen_monotonic]
        self._recent_tasks: Dict[int, "OrderedDict[int, float]"] = {}

    def _handle_event(self, cpu, data, size):
        event = self.bpf["events"].event(data)
        ts_s = time.monotonic()
        cpu_id = int(getattr(event, "cpu", 0) or 0)
        prev_pid = int(getattr(event, "prev_pid", 0))

        self.total_switches += 1

        # Busy-time accounting: prev_pid != 0 ⇒ CPU was running a task.
        now_ns = time.monotonic_ns()
        last = self._last_ts_ns.get(cpu_id)
        if last is not None and prev_pid != 0:
            self._busy_ns[cpu_id] = self._busy_ns.get(cpu_id, 0) + (now_ns - last)
        self._last_ts_ns[cpu_id] = now_ns
        if self._window_start_ns is None:
            self._window_start_ns = now_ns

        seen = self._recent_tasks.setdefault(cpu_id, OrderedDict())
        if prev_pid != 0:
            seen.pop(prev_pid, None)
            seen[prev_pid] = ts_s

    def poll(self, timeout_ms: int = 100):
        self.bpf.perf_buffer_poll(timeout=timeout_ms)

    def get_kir_state(self) -> KernelIntermediateRepresentation:
        now_ns = time.monotonic_ns()
        start = self._window_start_ns or now_ns
        window = max((now_ns - start) / 1e9, 1e-6)

        cpus: Dict[int, CPUMetrics] = {}
        keys = {int(k.value) for k in self.cpu_metrics_map.keys()} | set(self._busy_ns)
        for cpu_id in sorted(keys):
            busy = self._busy_ns.get(cpu_id, 0)
            util = min(1.0, (busy / 1e9) / window)
            now_s = time.monotonic()
            seen = self._recent_tasks.get(cpu_id, OrderedDict())
            runnable = sum(1 for t in seen.values() if now_s - t <= self._TASK_TTL_S)
            cpus[cpu_id] = CPUMetrics(utilization=round(util, 4),
                                      runnable_tasks=runnable,
                                      irq_time=0.0)

        if not cpus:
            cpus[0] = CPUMetrics(utilization=0.0, runnable_tasks=0, irq_time=0.0)

        state = SchedulerState(
            cpus=cpus,
            total_context_switches=self.total_switches,
            avg_latency_ms=self._estimate_latency_ms(cpus),
        )
        # Reset accounting window for next interval.
        self._busy_ns = {}
        self._window_start_ns = now_ns
        return KernelIntermediateRepresentation(timestamp=time.time(), scheduler=state)

    @staticmethod
    def _estimate_latency_ms(cpus: Dict[int, CPUMetrics]) -> float:
        """First-order estimate: queue depth × per-task dispatch quantum."""
        mean_q = sum(c.runnable_tasks for c in cpus.values()) / len(cpus)
        return round(min(20.0, mean_q * 1.5), 3)


# ─────────────────────────────────────────────────────────────── #
# Simulation monitor                                               #
# ─────────────────────────────────────────────────────────────── #

class SimulationMonitor:
    """KIR source for dev/demo environments; shares KernelSimulator dynamics."""

    def __init__(self, seed: int = 1234, n_cpus: int = 2):
        self._env = KernelSimulator(seed=seed, n_cpus=n_cpus)
        self._pending_action: Optional[object] = None

    def submit_action(self, action) -> None:
        """Runtime informs the simulated kernel what was actuated."""
        self._pending_action = action

    def poll(self, timeout_ms: int = 100) -> None:
        pass

    def get_kir_state(self) -> KernelIntermediateRepresentation:
        return self._env.step(self._pending_action)
