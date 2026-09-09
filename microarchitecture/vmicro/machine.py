"""Exact oracle CPU: ISA execution + causal microarchitecture.

Not a learned model. This is the ground truth the surrogate learns and the
search validates against (context.txt: "perfect verification... every virtual
claim can immediately be checked against the exact game" - here, the exact CPU).

Microarchitecture (all causal, deterministic, seeded):
- in-order fetch/issue, scoreboard RAW hazards, configurable issue width
- functional-unit latencies (ALU 1c, MUL 3c, DIV 6c, VEC 4c / accel 2c + setup)
- L1 cache: 16 lines x 4 words, LRU, hit 2c / miss 20c, sequential prefetcher
- 2-bit branch predictor keyed by PC, speculation modes OFF/CONSERVATIVE/AGGRESSIVE
- power modes ECO/BALANCED/TURBO (issue width, latency/energy scale, leakage)
- 4 thermal zones, calibrated heating so turbo trades cycles for temperature
- hard SafetyGovernor outside learned control (same thesis as DEA)

Snapshot/restore via explicit state copy -> enables branchable virtual worlds.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from vmicro.isa import Instruction, Op

MEM_WORDS = 4096
L1_LINES = 16
LINE_WORDS = 4
L1_HIT_LATENCY = 2
DRAM_MISS_LATENCY = 20
MISPREDICT_PENALTY = 4


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


@dataclass(frozen=True)
class ControlBundle:
    name: str
    power_mode: PowerMode
    prefetch_degree: int  # 0, 1, 2, 4
    speculation_mode: SpeculationMode
    accelerator_mode: AcceleratorMode

    def __post_init__(self) -> None:
        if self.prefetch_degree not in (0, 1, 2, 4):
            raise ValueError("prefetch_degree must be one of 0,1,2,4")


ACTION_LIBRARY: dict[str, ControlBundle] = {
    "balanced": ControlBundle("balanced", PowerMode.BALANCED, 1, SpeculationMode.CONSERVATIVE, AcceleratorMode.SELECTIVE),
    "latency": ControlBundle("latency", PowerMode.TURBO, 2, SpeculationMode.AGGRESSIVE, AcceleratorMode.AGGRESSIVE),
    "streaming": ControlBundle("streaming", PowerMode.TURBO, 4, SpeculationMode.CONSERVATIVE, AcceleratorMode.AGGRESSIVE),
    "irregular": ControlBundle("irregular", PowerMode.BALANCED, 0, SpeculationMode.OFF, AcceleratorMode.OFF),
    "control": ControlBundle("control", PowerMode.BALANCED, 0, SpeculationMode.AGGRESSIVE, AcceleratorMode.OFF),
    "thermal_recovery": ControlBundle("thermal_recovery", PowerMode.ECO, 0, SpeculationMode.OFF, AcceleratorMode.OFF),
}

POWER_CHARACTERISTICS = {
    PowerMode.ECO: {"issue_width": 1, "latency_scale": 1.25, "energy_scale": 0.76, "leakage": 0.06},
    PowerMode.BALANCED: {"issue_width": 2, "latency_scale": 1.0, "energy_scale": 1.0, "leakage": 0.10},
    PowerMode.TURBO: {"issue_width": 3, "latency_scale": 0.78, "energy_scale": 1.34, "leakage": 0.18},
}

FEATURE_NAMES = [
    "bias", "pc_progress", "pending_pressure", "mem_fraction_window",
    "branch_fraction_window", "vec_fraction_window", "l1_miss_rate",
    "branch_mispredict_rate", "thermal_headroom", "hot",
    "active_balanced", "active_latency", "active_streaming",
    "active_irregular", "active_control", "active_recovery",
]
_ACTIVE_SLOT = {
    "balanced": "active_balanced", "latency": "active_latency",
    "streaming": "active_streaming", "irregular": "active_irregular",
    "control": "active_control", "thermal_recovery": "active_recovery",
}


@dataclass
class _InFlight:
    pc: int
    instr: Instruction
    cycles_remaining: int
    dest_value: Any = None
    dest_addr: int | None = None
    branch_taken_actual: bool = False
    branch_predicted_taken: bool = False
    branch_mispredicted: bool = False
    mem_latency: int = 0
    energy: float = 0.0


@dataclass
class CPUStats:
    cycles: int = 0
    instrs_retired: int = 0
    energy: float = 0.0
    controller_energy: float = 0.0
    l1_hits: int = 0
    l1_misses: int = 0
    prefetches_issued: int = 0
    useful_prefetches: int = 0
    branch_predictions: int = 0
    branch_mispredictions: int = 0
    accelerator_ops: int = 0
    issue_stalls: int = 0
    safety_overrides: int = 0
    action_switches: int = 0
    thermal_exposure: float = 0.0
    cycles_above_target: int = 0


class SafetyGovernor:
    def apply(self, action: ControlBundle, headroom: float) -> tuple[ControlBundle, list[str]]:
        if headroom <= 0.0:
            return ACTION_LIBRARY["thermal_recovery"], ["thermal_limit"]
        if headroom < 3.0 and action.power_mode is PowerMode.TURBO:
            return replace(action, name=f"{action.name}:guarded", power_mode=PowerMode.BALANCED), ["turbo_headroom"]
        return action, []


class CPU:
    """Deterministic exact CPU oracle."""

    def __init__(
        self,
        program: list[Instruction],
        mem_init: dict[int, int] | None = None,
        reg_init: dict[int, int] | None = None,
        control: ControlBundle | None = None,
        ambient: float = 30.0,
        thermal_limit: float = 42.0,
        thermal_target: float = 38.0,
        thermal_heating: float = 0.11,
        thermal_cooling: float = 0.018,
        max_cycles: int = 50_000,
    ) -> None:
        if not program:
            raise ValueError("program must be non-empty")
        self.program = list(program)
        self.regs = [0] * 32
        if reg_init:
            for k, v in reg_init.items():
                self.regs[k] = int(v)
        self.mem = [0] * MEM_WORDS
        if mem_init:
            for k, v in mem_init.items():
                self.mem[k % MEM_WORDS] = int(v)
        self.pc = 0
        self.halted = False
        self.retired_pcs: list[int] = []
        self.in_flight: list[_InFlight] = []
        self.pending_writes: set[int] = set()
        self.fetch_pc = 0
        self.control = control or ACTION_LIBRARY["balanced"]
        self.requested = self.control
        self.ambient = ambient
        self.thermal_limit = thermal_limit
        self.thermal_target = thermal_target
        self.thermal_heating = thermal_heating
        self.thermal_cooling = thermal_cooling
        self.max_cycles = max_cycles
        self.temperatures = [ambient] * 4
        self.peak_temperature = ambient
        self.stats = CPUStats()
        self.governor = SafetyGovernor()
        self.cache: OrderedDict[int, None] = OrderedDict()
        self.prefetched: set[int] = set()
        self.branch_table: dict[int, int] = {}  # pc -> 2-bit counter 0..3
        self.action_histogram: dict[str, int] = {}
        self._bubble = 0  # mispredict flush bubbles remaining
        self._accel_last_cycle: int | None = None
        self._window_ops: list[Instruction] = []

    # ---- observation / features ----
    def headroom(self) -> float:
        return self.thermal_limit - max(self.temperatures)

    def observe(self) -> dict[str, Any]:
        n = len(self.program)
        return {
            "pc": self.pc, "fetch_pc": self.fetch_pc,
            "regs": list(self.regs), "halted": self.halted,
            "cycles": self.stats.cycles, "retired": self.stats.instrs_retired,
            "pending": len(self.in_flight), "bubble": self._bubble,
            "l1_miss_rate": self.l1_miss_rate, "branch_mispredict_rate": self.branch_mispredict_rate,
            "temperatures": list(self.temperatures), "headroom": self.headroom(),
            "action": self.control.name,
        }

    @property
    def l1_miss_rate(self) -> float:
        t = self.stats.l1_hits + self.stats.l1_misses
        return self.stats.l1_misses / t if t else 0.0

    @property
    def branch_mispredict_rate(self) -> float:
        t = self.stats.branch_predictions
        return self.stats.branch_mispredictions / t if t else 0.0

    def feature_vector(self) -> list[float]:
        n = max(len(self.program), 1)
        win = self._window_ops[-16:] if self._window_ops else []
        denom = max(len(win), 1)
        mem_f = sum(i.is_mem for i in win) / denom if win else 0.0
        br_f = sum(i.is_branch for i in win) / denom if win else 0.0
        vec_f = sum(i.is_vector for i in win) / denom if win else 0.0
        head = max(-1.0, min(self.headroom() / 12.0, 1.0))
        slot = _ACTIVE_SLOT.get(self.control.name.split(":")[0], "")
        feats = [
            1.0, self.fetch_pc / n, min(len(self.in_flight) / 4.0, 1.0),
            mem_f, br_f, vec_f, self.l1_miss_rate, self.branch_mispredict_rate,
            head, float(self.headroom() < 3.0),
        ]
        feats.extend(1.0 if s == slot else 0.0 for s in _ACTIVE_SLOT.values())
        assert len(feats) == len(FEATURE_NAMES)
        return feats

    # ---- control ----
    def set_control(self, action: ControlBundle, controller_energy: float = 0.0) -> list[str]:
        applied, reasons = self.governor.apply(action, self.headroom())
        if reasons:
            self.stats.safety_overrides += 1
        if applied.name != self.control.name:
            self.stats.action_switches += 1
        self.requested = action
        self.control = applied
        self.stats.controller_energy += controller_energy
        self.action_histogram[applied.name] = self.action_histogram.get(applied.name, 0) + 1
        return reasons

    # ---- snapshot / fork (branchable worlds) ----
    def snapshot(self) -> dict:
        return copy.deepcopy({
            "regs": self.regs, "mem": self.mem, "pc": self.pc, "fetch_pc": self.fetch_pc,
            "halted": self.halted, "retired": list(self.retired_pcs),
            "in_flight": self.in_flight, "pending": self.pending_writes,
            "control": self.control, "requested": self.requested,
            "temps": self.temperatures, "peak": self.peak_temperature,
            "stats": self.stats, "cache": self.cache, "prefetched": self.prefetched,
            "btable": self.branch_table, "hist": self.action_histogram,
            "bubble": self._bubble, "accel_last": self._accel_last_cycle,
            "window": self._window_ops,
        })

    def restore(self, snap: dict) -> None:
        s = copy.deepcopy(snap)
        self.regs = s["regs"]; self.mem = s["mem"]; self.pc = s["pc"]
        self.fetch_pc = s["fetch_pc"]; self.halted = s["halted"]
        self.retired_pcs = s["retired"]; self.in_flight = s["in_flight"]
        self.pending_writes = s["pending"]; self.control = s["control"]
        self.requested = s["requested"]; self.temperatures = s["temps"]
        self.peak_temperature = s["peak"]; self.stats = s["stats"]
        self.cache = s["cache"]; self.prefetched = s["prefetched"]
        self.branch_table = s["btable"]; self.action_histogram = s["hist"]
        self._bubble = s["bubble"]; self._accel_last_cycle = s["accel_last"]
        self._window_ops = s["window"]

    def fork(self) -> "CPU":
        c = CPU.__new__(CPU)
        c.program = self.program  # immutable sharing is fine
        snap = self.snapshot()
        # bind attributes without re-running __init__
        c.regs = snap["regs"]; c.mem = snap["mem"]; c.pc = snap["pc"]
        c.fetch_pc = snap["fetch_pc"]; c.halted = snap["halted"]
        c.retired_pcs = snap["retired"]; c.in_flight = snap["in_flight"]
        c.pending_writes = snap["pending"]; c.control = snap["control"]
        c.requested = snap["requested"]; c.temperatures = snap["temps"]
        c.peak_temperature = snap["peak"]; c.stats = snap["stats"]
        c.cache = snap["cache"]; c.prefetched = snap["prefetched"]
        c.branch_table = snap["btable"]; c.action_histogram = snap["hist"]
        c._bubble = snap["bubble"]; c._accel_last_cycle = snap["accel_last"]
        c._window_ops = snap["window"]
        c.ambient = self.ambient; c.thermal_limit = self.thermal_limit
        c.thermal_target = self.thermal_target; c.thermal_heating = self.thermal_heating
        c.thermal_cooling = self.thermal_cooling; c.max_cycles = self.max_cycles
        c.governor = self.governor
        return c

    # ---- internals ----
    def _line(self, addr: int) -> int:
        return (addr % MEM_WORDS) // LINE_WORDS

    def _cache_access(self, addr: int, is_load: bool) -> tuple[int, float]:
        line = self._line(addr)
        hit = line in self.cache
        if hit:
            self.cache.move_to_end(line)
            self.stats.l1_hits += 1
            if line in self.prefetched:
                self.stats.useful_prefetches += 1
                self.prefetched.discard(line)
            latency, energy = L1_HIT_LATENCY, 0.4
        else:
            self.stats.l1_misses += 1
            # insert with LRU eviction
            self.cache[line] = None
            if len(self.cache) > L1_LINES:
                self.cache.popitem(last=False)
            latency, energy = DRAM_MISS_LATENCY, 6.0
        # sequential prefetch on loads only
        if is_load and self.control.prefetch_degree:
            for d in range(1, self.control.prefetch_degree + 1):
                pl = line + d
                if pl not in self.cache:
                    self.stats.prefetches_issued += 1
                    self.prefetched.add(pl)
                    self.cache[pl] = None
                    if len(self.cache) > L1_LINES:
                        self.cache.popitem(last=False)
                    energy += 0.5  # bandwidth/energy cost of prefetch
        return latency, energy

    def _predict(self, pc: int) -> bool:
        return self.branch_table.get(pc, 1) >= 2

    def _update_predictor(self, pc: int, taken: bool) -> None:
        c = self.branch_table.get(pc, 1)
        c = min(3, c + 1) if taken else max(0, c - 1)
        self.branch_table[pc] = c

    def _resolve_branch(self, instr: Instruction, regs_snapshot: list[int]) -> bool:
        if instr.op is Op.BEQ:
            return regs_snapshot[instr.rs1] == regs_snapshot[instr.rs2]
        if instr.op is Op.BNE:
            return regs_snapshot[instr.rs1] != regs_snapshot[instr.rs2]
        return True  # JUMP always taken

    def _alu_latency_energy(self, instr: Instruction) -> tuple[int, float, bool]:
        ch = POWER_CHARACTERISTICS[self.control.power_mode]
        ls, es = float(ch["latency_scale"]), float(ch["energy_scale"])
        op = instr.op
        use_accel = False
        if op in (Op.ADD, Op.SUB, Op.ADDI, Op.MOV, Op.NOP):
            base, e = 1, 1.0
        elif op is Op.MUL:
            base, e = 3, 4.0
        elif op is Op.DIV:
            base, e = 6, 7.0
        elif op in (Op.VADD, Op.VMUL):
            eligible = (self.control.accelerator_mode is AcceleratorMode.AGGRESSIVE) or (
                self.control.accelerator_mode is AcceleratorMode.SELECTIVE)
            if eligible:
                use_accel = True
                base, e = 2, 2.8
            else:
                base, e = 4, 5.0
        elif op in (Op.BEQ, Op.BNE, Op.JUMP):
            base, e = 1, 1.5
        elif op is Op.HALT:
            base, e = 1, 0.2
        else:
            base, e = 1, 1.0
        import math
        lat = max(1, math.ceil(base * ls))
        extra_setup = 0
        if use_accel and (self._accel_last_cycle is None or self.stats.cycles - self._accel_last_cycle > 16):
            extra_setup, e = 2, e + 1.0
        return lat + extra_setup, e * es, use_accel

    def _try_issue_one(self) -> bool:
        """Issue single oldest instruction if no RAW hazard. Returns True if issued
        (or halted/completed PC advance), False on stall. Handles branches."""
        if self.halted or self._bubble > 0:
            return False
        if not (0 <= self.fetch_pc < len(self.program)):
            self.halted = True
            return False
        instr = self.program[self.fetch_pc]
        # RAW hazard: any source pending?
        if any(r in self.pending_writes for r in instr.reads):
            return False
        # Memory ordering: program-order memory. A younger mem op must wait
        # while any older mem op is still in flight, otherwise a LOAD can
        # bypass an older STORE to the same address and read stale data.
        # (Conservative LSQ: serializes memory, guarantees correctness.)
        if instr.is_mem and any(op.instr.is_mem for op in self.in_flight):
            return False
        # structural: cap in-flight
        if len(self.in_flight) >= 8:
            return False
        regs_now = list(self.regs)
        if instr.op is Op.HALT:
            # HALT is a barrier: only issue once the pipeline has drained,
            # otherwise it can retire before older in-flight ops (e.g. LOAD)
            # and halt the machine with work still unretired.
            if self.in_flight:
                return False
            self.in_flight.append(_InFlight(self.fetch_pc, instr, 1))
            for r in instr.writes:
                self.pending_writes.add(r)
            self.fetch_pc += 1
            return True
        if instr.is_branch:
            mode = self.control.speculation_mode
            taken = self._resolve_branch(instr, regs_now)
            predicted = False
            mispredicted = False
            if mode is SpeculationMode.OFF:
                # stall: resolve immediately, no speculation benefit
                lat, energy = 2, 1.5 * float(POWER_CHARACTERISTICS[self.control.power_mode]["energy_scale"])
                self.in_flight.append(_InFlight(self.fetch_pc, instr, lat, energy=energy,
                                               branch_taken_actual=taken, branch_predicted_taken=taken))
                self.stats.branch_predictions += 0  # abstained
            else:
                predicted = self._predict(self.fetch_pc)
                self.stats.branch_predictions += 1
                mispredicted = (predicted != taken)
                if mispredicted:
                    self.stats.branch_mispredictions += 1
                lat, energy = 1, 1.5 * float(POWER_CHARACTERISTICS[self.control.power_mode]["energy_scale"])
                if mispredicted:
                    lat += MISPREDICT_PENALTY
                    energy += 2.0
                    # conservative pays full flush bubble; aggressive hides 2 cycles
                    self._bubble += MISPREDICT_PENALTY if mode is SpeculationMode.CONSERVATIVE else max(0, MISPREDICT_PENALTY - 2)
                self.in_flight.append(_InFlight(self.fetch_pc, instr, lat, energy=energy,
                                               branch_taken_actual=taken, branch_predicted_taken=predicted,
                                               branch_mispredicted=mispredicted))
                self._update_predictor(self.fetch_pc, taken)
            # PC update at issue (speculation resolves immediately in this model,
            # mispredict cost expressed as latency + bubbles)
            self.fetch_pc = self.fetch_pc + instr.imm if taken else self.fetch_pc + 1
            self._window_ops.append(instr)
            return True
        if instr.op is Op.LOAD:
            addr = (regs_now[instr.rs1] + instr.imm) % MEM_WORDS
            lat, energy = self._cache_access(addr, True)
            import math
            ls = float(POWER_CHARACTERISTICS[self.control.power_mode]["latency_scale"])
            lat = max(1, math.ceil(lat * (0.6 + 0.4 * ls)))
            energy += 0.2
            val = self.mem[addr]
            self.in_flight.append(_InFlight(self.fetch_pc, instr, lat, dest_value=val, mem_latency=lat, energy=energy))
            for r in instr.writes:
                self.pending_writes.add(r)
            self.fetch_pc += 1
            self._window_ops.append(instr)
            return True
        if instr.op is Op.STORE:
            addr = (regs_now[instr.rs1] + instr.imm) % MEM_WORDS
            lat, energy = self._cache_access(addr, False)
            import math
            ls = float(POWER_CHARACTERISTICS[self.control.power_mode]["latency_scale"])
            lat = max(1, math.ceil(lat * (0.6 + 0.4 * ls)))
            energy += 0.3
            val = regs_now[instr.rs2]
            self.in_flight.append(_InFlight(self.fetch_pc, instr, lat, dest_value=val, dest_addr=addr, mem_latency=lat, energy=energy))
            self.fetch_pc += 1
            self._window_ops.append(instr)
            return True
        lat, energy, use_accel = self._alu_latency_energy(instr)
        # compute result now (deterministic), write back at completion
        a, b = regs_now[instr.rs1], regs_now[instr.rs2]
        if instr.op is Op.ADD or instr.op is Op.VADD:
            val = a + b
        elif instr.op is Op.SUB:
            val = a - b
        elif instr.op is Op.MUL or instr.op is Op.VMUL:
            val = a * b
        elif instr.op is Op.DIV:
            val = (a // b) if b != 0 else 0
        elif instr.op is Op.ADDI:
            val = a + instr.imm
        elif instr.op is Op.MOV:
            val = a
        else:
            val = 0
        if use_accel:
            self.stats.accelerator_ops += 1
            self._accel_last_cycle = self.stats.cycles
        self.in_flight.append(_InFlight(self.fetch_pc, instr, lat, dest_value=val, energy=energy))
        for r in instr.writes:
            self.pending_writes.add(r)
        self.fetch_pc += 1
        self._window_ops.append(instr)
        return True

    def step_cycle(self) -> dict:
        if self.halted:
            raise RuntimeError("cannot step a halted CPU")
        # Hard safety enforced every cycle (not only at control intervals),
        # mirroring DEA's _enforce_safety_between_intervals: correctness and
        # thermal safety stay outside learned/heuristic control.
        applied, reasons = self.governor.apply(self.control, self.headroom())
        if reasons and applied != self.control:
            self.stats.safety_overrides += 1
            self.stats.action_switches += 1
            self.control = applied
        ch = POWER_CHARACTERISTICS[self.control.power_mode]
        width = int(ch["issue_width"])
        issued = 0
        if self._bubble > 0:
            self._bubble -= 1
        else:
            for _ in range(width):
                if self._try_issue_one():
                    issued += 1
                else:
                    break
        if issued == 0 and (self.fetch_pc < len(self.program) and not self.halted):
            # only count stall if work remains and nothing issued but pipeline not drained by halt
            if self.in_flight or self._bubble:
                pass
            else:
                self.stats.issue_stalls += 1
        # advance in-flight, retire
        dispatch_energy = 0.0
        zone_power = [float(ch["leakage"]) / 4.0] * 4
        for op in self.in_flight:
            zone_power[0 if not op.instr.is_mem and not op.instr.is_vector else (1 if op.instr.is_mem else 2)] += op.energy / max(op.cycles_remaining, 1)
        for op in self.in_flight:
            dispatch_energy += op.energy / max(op.cycles_remaining, 1) * 0.0  # energy charged at retire below
        still: list[_InFlight] = []
        retired_now = 0
        retired_energy = 0.0
        for op in self.in_flight:
            op.cycles_remaining -= 1
            if op.cycles_remaining <= 0:
                ins = op.instr
                if ins.op is Op.HALT:
                    self.halted = True
                elif ins.op is Op.LOAD:
                    self.regs[ins.rd] = int(op.dest_value)
                    self.pending_writes.discard(ins.rd)
                elif ins.op is Op.STORE:
                    assert op.dest_addr is not None
                    self.mem[op.dest_addr] = int(op.dest_value)
                elif ins.writes:
                    self.regs[ins.writes[0]] = int(op.dest_value)
                    self.pending_writes.discard(ins.writes[0])
                self.regs[0] = 0
                self.retired_pcs.append(op.pc)
                self.stats.instrs_retired += 1
                retired_now += 1
                retired_energy += op.energy
            else:
                still.append(op)
        self.in_flight = still
        leakage = float(ch["leakage"])
        self.stats.energy += retired_energy + leakage
        # thermal
        current_power = sum(zone_power)
        for i, p in enumerate(zone_power):
            t = self.temperatures[i]
            self.temperatures[i] = t + self.thermal_heating * p - self.thermal_cooling * (t - self.ambient)
        hottest = max(self.temperatures)
        self.peak_temperature = max(self.peak_temperature, hottest)
        excess = max(0.0, hottest - self.thermal_target)
        if excess > 0:
            self.stats.thermal_exposure += excess ** 2
            self.stats.cycles_above_target += 1
        self.stats.cycles += 1
        return {"issued": issued, "retired": retired_now, "halted": self.halted,
                "power": current_power, "temps": list(self.temperatures)}

    def run(self, control_plan: list[ControlBundle] | None = None, interval: int = 8) -> dict:
        """Run to HALT/end/timeout. control_plan optionally sets bundle every interval."""
        step = 0
        idx = 0
        while not self.halted and self.stats.cycles < self.max_cycles:
            if control_plan and self.stats.cycles % interval == 0 and idx < len(control_plan):
                self.set_control(control_plan[idx], controller_energy=0.01)
                idx += 1
            if self.fetch_pc >= len(self.program) and not self.in_flight:
                self.halted = True
                break
            if self.fetch_pc >= len(self.program):
                # drain pipeline
                self.step_cycle()
                continue
            try:
                self.step_cycle()
            except RuntimeError:
                break
            step += 1
        return self.result("HALTED" if self.halted else "TIMEOUT")

    def result(self, status: str | None = None) -> dict:
        s = self.stats
        total = round(s.energy, 6)
        ctrl = round(s.controller_energy, 6)
        metrics = {
            "completion_status": status or ("HALTED" if self.halted else "TIMEOUT"),
            "cycles": s.cycles, "instrs": s.instrs_retired,
            "ipc": s.instrs_retired / s.cycles if s.cycles else 0.0,
            "energy": total, "controller_energy": ctrl,
            "l1_hits": s.l1_hits, "l1_misses": s.l1_misses,
            "l1_miss_rate": self.l1_miss_rate,
            "prefetches": s.prefetches_issued, "useful_prefetches": s.useful_prefetches,
            "branch_mispredict_rate": self.branch_mispredict_rate,
            "branch_mispredictions": s.branch_mispredictions,
            "accelerator_ops": s.accelerator_ops,
            "peak_temp": round(self.peak_temperature, 4),
            "thermal_exposure": round(s.thermal_exposure, 6),
            "safety_overrides": s.safety_overrides,
            "action_switches": s.action_switches,
            "action_histogram": dict(self.action_histogram),
        }
        metrics["objective"] = (
            metrics["cycles"] + 0.08 * (metrics["energy"] - metrics["controller_energy"])
            + 0.6 * metrics["thermal_exposure"] + 0.02 * metrics["controller_energy"]
        )
        return metrics
