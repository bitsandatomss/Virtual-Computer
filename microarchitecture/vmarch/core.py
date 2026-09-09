"""Out-of-order execution engine: the exact microarchitecture oracle.

Models fetch (I-cache + predictor + BTB) → decode → rename (RAT, phys file,
checkpoints) → dispatch (ROB + RS + LSQ) → out-of-order issue → execute
(INT/MUL/DIV/VEC/LSU/BRU) → in-order commit, with store-queue forwarding,
memory-disambiguation stalls, FENCE ordering, precise squash, DVFS, and a
4-zone thermal model.

Abstractions (documented, not hidden):
- Branch *targets* are exact (ISA has no indirect branches); only direction
  is predicted. BTB is maintained for timing/energy/stats fidelity.
- Predictor history advances at commit (non-speculative); fetch uses committed
  history. No speculative-history repair problem by construction.
- Stores retire through an idealized store buffer: cache write occurs at
  commit with no extra retirement stall.
- Wrong-path loads may pollute caches (realistic); squashed stores never
  reach the hierarchy (commit-only writes).

Everything is deterministic (no RNG) and deepcopy-able for world forking.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from vmarch import isa
from vmarch.branch import BTB, Predictor, make_predictor
from vmarch.cache import MemoryHierarchy
from vmarch.config import (ENERGY, POWER_CHARACTERISTICS, AcceleratorMode,
                           ControlBundle, MicroarchConfig, PowerMode,
                           SafetyGovernor, SpeculationMode, ACTION_LIBRARY)

MISPREDICT_PENALTY = 4

FEATURE_NAMES = [
    "bias", "rob_occupancy", "rs_occupancy", "lsq_occupancy",
    "fetch_starved", "dependence_stalled", "l1d_miss_rate", "l1i_miss_rate",
    "branch_mispredict_rate", "mshr_pressure", "dram_row_hit_rate",
    "sequentiality", "thermal_headroom", "hot",
    "mem_fraction", "store_fraction", "branch_fraction", "vec_fraction",
    "active_balanced", "active_latency", "active_streaming",
    "active_irregular", "active_control", "active_recovery",
]
_ACTIVE_SLOT = {
    "balanced": "active_balanced", "latency": "active_latency",
    "streaming": "active_streaming", "irregular": "active_irregular",
    "control": "active_control", "thermal_recovery": "active_recovery",
}


def _s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


class ROBEntry:
    __slots__ = ("pc", "ins", "dst", "dst_phys", "prev_phys", "srcs", "done",
                 "value", "addr", "taken_actual", "taken_pred", "mispredicted",
                 "is_mem", "is_store", "is_load", "is_branch", "is_fence",
                 "is_halt", "is_prefetch", "checkpoint", "exec_cycles",
                 "src_vals")

    def __init__(self, pc: int, ins: isa.Instruction) -> None:
        self.pc = pc
        self.ins = ins
        self.dst = ins.rd if ins.writes else -1
        self.dst_phys = -1
        self.prev_phys = -1
        self.srcs: tuple[int, ...] = ()
        self.done = False
        self.value: int | None = None
        self.addr: int | None = None
        self.taken_actual: bool | None = None
        self.taken_pred: bool | None = None
        self.mispredicted = False
        op = ins.op
        self.is_mem = op in (isa.Op.LOAD, isa.Op.STORE)
        self.is_store = op is isa.Op.STORE
        self.is_load = op is isa.Op.LOAD
        self.is_branch = op in (isa.Op.BEQ, isa.Op.BNE, isa.Op.JUMP)
        self.is_fence = op is isa.Op.FENCE
        self.is_halt = op in (isa.Op.HALT, isa.Op.ECALL)
        self.is_prefetch = op is isa.Op.PREFETCH
        self.checkpoint = None
        self.exec_cycles = 0
        self.src_vals: tuple[int, ...] | None = None  # operands captured at
        # dispatch (sources guaranteed ready); the phys file may be recycled
        # before completion, the ROB entry may not.


class RSEntry:
    __slots__ = ("rob_pos", "unit", "age")

    def __init__(self, rob_pos: int, unit: str, age: int) -> None:
        self.rob_pos = rob_pos
        self.unit = unit
        self.age = age


class Core:
    """Single OoO core. Owns arch state, pipeline, predictor, memory, power."""

    def __init__(self, program: list[isa.Instruction], cfg: MicroarchConfig | None = None,
                 mem_init: dict[int, int] | None = None,
                 reg_init: dict[int, int] | None = None,
                 control: ControlBundle | None = None) -> None:
        if not program:
            raise ValueError("program must be non-empty")
        self.program = program
        self.cfg = cfg or MicroarchConfig()
        self.regs = [0] * 32
        if reg_init:
            for k, v in reg_init.items():
                self.regs[k] = _s32(int(v))
        self.regs[0] = 0
        self.mem = [0] * self.cfg.mem_words
        if mem_init:
            for k, v in mem_init.items():
                self.mem[k % self.cfg.mem_words] = _s32(int(v))
        # rename structures
        self.nphys = self.cfg.phys_regs
        self.phys_val = [0] * self.nphys
        self.phys_ready = [True] * self.nphys
        self.rat = list(range(32))
        # Free physical registers: all indices not covered by the initial RAT.
        # (range stop/step must descend: pop() hands out 32, 33, ... in order.)
        self.free_list = list(range(self.nphys - 1, 31, -1))
        # pipeline structures
        self.rob: list[ROBEntry] = []
        self.rob_head = 0
        self.rs: list[RSEntry] = []
        self.lsq: list[int] = []  # rob positions, program order
        self.executing: list[list] = []  # [rob_pos, remaining]
        self._drop_redirect: set[int] = set()  # rob slots retired by redirect guard
        self._pending_free: list[int] = []  # P2: prev-phys awaiting reclaim
        # (freed only once no undispatched RS entry still sources them)
        self.fetch_queue: list[dict] = []
        self.fetch_pc = 0
        self.fetch_bubble = 0
        self.fetch_halted = False
        self.halted = False
        self.exit_code = 0
        # speculation control
        self.control = control or ACTION_LIBRARY["balanced"]
        self.requested = self.control
        self.governor = SafetyGovernor()
        # microarch units
        self.predictor: Predictor = make_predictor(self.cfg.predictor)
        self.btb = BTB(self.cfg.btb_entries)
        self.memory = MemoryHierarchy(self.cfg)
        self.memory.trace = [] if cfg.trace else None
        # power/thermal
        self.temperatures = [self.cfg.ambient] * 4
        self.peak_temperature = self.cfg.ambient
        self.energy_dyn = 0.0
        self.controller_energy = 0.0
        # stats
        self.cycle = 0
        self.retired = 0
        self.retired_pcs: list[int] = []
        self.st = {k: 0 for k in (
            "rob_full", "rs_full", "lsq_full", "fetch_starve", "dep_stall",
            "mshr_stall", "squashes", "flush_bubbles", "forwards", "fence_stall",
            "icache_stall", "commit_idle", "l1i_miss", "stores", "loads")}
        self.safety_overrides = 0
        self.action_switches = 0
        self.action_histogram: dict[str, int] = {}
        self.thermal_exposure = 0.0
        self.cycles_above_target = 0
        self._age = 0
        self._accel_last = -10**9
        self._window: list[isa.Instruction] = []
        self._mem_lines: list[int] = []
        self._no_progress = 0
        # per-cycle activity energy buckets
        self._act = [0.0, 0.0, 0.0, 0.0]

    # ============ control / observation ============
    def headroom(self) -> float:
        return self.cfg.thermal_limit - max(self.temperatures)

    def set_control(self, action: ControlBundle, controller_energy: float = 0.01) -> list[str]:
        applied, reasons = self.governor.apply(action, self.headroom())
        if reasons:
            self.safety_overrides += 1
        if applied.name != self.control.name:
            self.action_switches += 1
        self.requested = action
        self.control = applied
        self.controller_energy += controller_energy
        self.action_histogram[applied.name] = self.action_histogram.get(applied.name, 0) + 1
        return reasons

    def _widths(self) -> tuple[int, int, int, int]:
        ch = POWER_CHARACTERISTICS[self.control.power_mode]
        s = float(ch["width_scale"])
        sc = lambda b: max(1, int(round(b * s)))
        return (sc(self.cfg.fetch_width), sc(self.cfg.decode_width),
                sc(self.cfg.issue_width), sc(self.cfg.commit_width))

    def _escale(self) -> float:
        return float(POWER_CHARACTERISTICS[self.control.power_mode]["energy_scale"])

    def _lscale(self) -> float:
        return float(POWER_CHARACTERISTICS[self.control.power_mode]["latency_scale"])

    def observe(self) -> dict[str, Any]:
        return {
            "fetch_pc": self.fetch_pc, "halted": self.halted,
            "cycle": self.cycle, "retired": self.retired,
            "rob_occ": (len(self.rob) - self.rob_head) / max(self.cfg.rob_size, 1),
            "rs_occ": len(self.rs) / max(self.cfg.rs_size, 1),
            "lsq_occ": len(self.lsq) / max(self.cfg.lsq_size, 1),
            "l1d_miss": self.memory.l1d_miss_rate,
            "l1i_miss": self.memory.l1i_miss_rate,
            "br_miss": self.predictor.rate,
            "mshr_pressure": len(self.memory.mshr) / max(self.cfg.mshr_entries, 1),
            "row_hit": self.memory.dram.row_hit_rate,
            "temps": list(self.temperatures), "headroom": self.headroom(),
            "action": self.control.name,
        }

    def feature_vector(self) -> list[float]:
        o = self.observe()
        win = self._window[-16:] if self._window else []
        dn = max(len(win), 1)
        mem_f = sum(1 for i in win if i.is_mem) / dn if win else 0.0
        st_f = sum(1 for i in win if i.op is isa.Op.STORE) / dn if win else 0.0
        br_f = sum(1 for i in win if i.is_branch) / dn if win else 0.0
        vec_f = sum(1 for i in win if i.is_vector) / dn if win else 0.0
        head = max(-1.0, min(o["headroom"] / 12.0, 1.0))
        slot = _ACTIVE_SLOT.get(self.control.name.split(":")[0], "")
        feats = [1.0, o["rob_occ"], o["rs_occ"], o["lsq_occ"],
                 float(self.st["fetch_starve"] > self.st["commit_idle"]),
                 float(bool(self.rs) and not any(self._rs_ready(e) for e in self.rs)),
                 o["l1d_miss"], o["l1i_miss"], o["br_miss"], o["mshr_pressure"],
                 o["row_hit"], self._sequentiality(),
                 head, float(o["headroom"] < 3.0),
                 mem_f, st_f, br_f, vec_f]
        feats.extend(1.0 if s == slot else 0.0 for s in _ACTIVE_SLOT.values())
        assert len(feats) == len(FEATURE_NAMES), (len(feats), len(FEATURE_NAMES))
        return feats

    def snapshot(self) -> dict:
        d = copy.deepcopy(self.__dict__)
        d.pop("program", None)
        d.pop("governor", None)
        return d

    def restore(self, snap: dict) -> None:
        prog, gov = self.program, self.governor
        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(snap))
        self.program, self.governor = prog, gov

    def fork(self) -> "Core":
        c = Core.__new__(Core)
        c.program = self.program
        c.governor = self.governor
        snap = self.snapshot()
        c.__dict__.update(copy.deepcopy(snap))
        c.program, c.governor = self.program, self.governor
        return c

    # ============ helpers ============
    def _rs_ready(self, e: RSEntry) -> bool:
        rob = self.rob[e.rob_pos]
        return all(self.phys_ready[p] for p in rob.srcs)

    def _sequentiality(self) -> float:
        lines = self._mem_lines[-16:]
        if len(lines) < 2:
            return 0.0
        return sum(1 for a, b in zip(lines, lines[1:]) if b == a + 1) / (len(lines) - 1)

    def _unit_for(self, ins: isa.Instruction) -> str:
        op = ins.op
        if op in (isa.Op.ADD, isa.Op.SUB, isa.Op.ADDI, isa.Op.MOV, isa.Op.NOP):
            return "int"
        if op is isa.Op.MUL:
            return "mul"
        if op is isa.Op.DIV:
            return "div"
        if op in (isa.Op.VADD, isa.Op.VMUL):
            return "vec"
        if op in (isa.Op.LOAD, isa.Op.STORE, isa.Op.PREFETCH):
            return "lsu"
        if op in (isa.Op.BEQ, isa.Op.BNE, isa.Op.JUMP):
            return "bru"
        return "int"  # FENCE/HALT/ECALL resolve at commit

    def _branch_in_flight(self) -> bool:
        return any(e.is_branch and not e.done for e in self.rob[self.rob_head:])

    def _older_fence(self, rob_pos: int) -> bool:
        return any(e.is_fence for e in self.rob[self.rob_head:rob_pos])

    # ============ pipeline stages (run in reverse order) ============
    def step_cycle(self) -> dict:
        if self.halted:
            raise RuntimeError("cannot step a halted core")
        applied, reasons = self.governor.apply(self.control, self.headroom())
        if reasons and applied != self.control:
            self.safety_overrides += 1
            self.action_switches += 1
            self.control = applied
        fw, dw, iw, cw = self._widths()
        self._act = [0.0, 0.0, 0.0, 0.0]
        self.memory.cycle = self.cycle  # stamp for the event trace
        issued_now = self._commit(cw)
        retired_now = issued_now
        self._advance_exec()
        n_issued = self._issue(iw)
        self._rename(dw)
        self._fetch(fw)
        # power/thermal
        leakage = float(POWER_CHARACTERISTICS[self.control.power_mode]["leakage"])
        zone = [leakage / 4.0 + a for a in self._act]
        zone[3] += 0.0
        for i, p in enumerate(zone):
            t = self.temperatures[i]
            self.temperatures[i] = t + self.cfg.thermal_heating * p - \
                self.cfg.thermal_cooling * (t - self.cfg.ambient)
        hottest = max(self.temperatures)
        self.peak_temperature = max(self.peak_temperature, hottest)
        excess = max(0.0, hottest - self.cfg.thermal_target)
        if excess > 0:
            self.thermal_exposure += excess ** 2
            self.cycles_above_target += 1
        self.energy_dyn += sum(self._act) + leakage
        self.memory.tick()
        self.cycle += 1
        if (self.fetch_halted and self.rob_head >= len(self.rob)
                and not self.executing and not self.fetch_queue
                and not self.halted):
            self.halted = True  # fall-through past program end: clean halt
        progress = retired_now + n_issued
        self._no_progress = 0 if progress else self._no_progress + 1
        if self._no_progress > 4 * self.cfg.rob_size + 64 and not self.halted:
            raise RuntimeError("deadlock: no issue/commit progress")
        return {"issued": n_issued, "committed": retired_now, "halted": self.halted}

    # ---- commit ----
    def _commit(self, cw: int) -> int:
        n = 0
        while n < cw and self.rob_head < len(self.rob):
            e = self.rob[self.rob_head]
            if not e.done:
                if n == 0:
                    self.st["commit_idle"] += 1
                break
            ins = e.ins
            if e.mispredicted:
                self._squash(self.rob_head)
                self.rob_head += 1
                n += 1
                continue
            if e.is_store:
                line = e.addr // self.cfg.line_words
                res = self.memory.store(line, self._escale())
                self._act[1] += res.energy
                # Ground truth update: the store becomes globally visible at
                # commit (in-order). Younger loads dispatched earlier are
                # ordered via the LSQ (stall/forward), so execute-time reads
                # of self.mem[] stay precise.
                self.mem[e.addr] = _s32(int(e.value))
                self.st["stores"] += 1
            elif e.is_branch:
                pass  # predictor already updated at resolve
            elif e.is_halt:
                if ins.op is isa.Op.ECALL:
                    self.exit_code = self.regs[ins.rs1]
                self.halted = True
                self.rob_head += 1
                self.retired += 1
                self.retired_pcs.append(e.pc)
                n += 1
                break
            if e.dst >= 0:
                self.regs[e.dst] = self.phys_val[e.dst_phys]
                self.regs[0] = 0
                if e.prev_phys >= 0:
                    self._pending_free.append(e.prev_phys)  # P2: reclaim on sweep
            if e.is_mem and self.lsq and self.lsq[0] == self.rob_head:
                self.lsq.pop(0)  # committed: no longer an ordering hazard
            self.retired += 1
            self.retired_pcs.append(e.pc)
            self._window.append(ins)
            self._act[0] += ENERGY["commit"]
            if e.dst_phys >= 0:
                pass  # stays mapped until overwritten
            self.rob_head += 1
            n += 1
        # compact occasionally
        if self.rob_head > 256:
            self.rob = self.rob[self.rob_head:]
            shift = self.rob_head
            for r in self.rs:
                r.rob_pos -= shift
            self.lsq = [p - shift for p in self.lsq]
            for ex in self.executing:
                ex[0] -= shift
            self.rob_head = 0
        self._reclaim()
        return n

    # ---- execute advance ----
    def _advance_exec(self) -> None:
        still = []
        for ex in self.executing:
            if ex[0] in self._drop_redirect:
                continue  # redirect-dropped younger: never completes
            ex[1] -= 1
            if ex[1] <= 0:
                self._complete(ex[0])
            else:
                still.append(ex)
        self.executing = still
        self._drop_redirect.clear()

    def _complete(self, rob_pos: int) -> None:
        e = self.rob[rob_pos]
        ins = e.ins
        if e.is_load:
            self.phys_val[e.dst_phys] = e.value
            self.phys_ready[e.dst_phys] = True
            e.done = True
        elif e.is_store:
            e.done = True  # cache write happens at commit
        elif e.is_branch:
            self._resolve_branch(e)
        elif e.is_prefetch:
            e.done = True
        else:
            if e.dst_phys >= 0 and e.value is not None:
                self.phys_val[e.dst_phys] = e.value
                self.phys_ready[e.dst_phys] = True
            e.done = True

    def _compute_alu(self, e: ROBEntry) -> tuple[int | None, int, float, bool]:
        ins = e.ins
        ls, es = self._lscale(), self._escale()
        use_accel = False
        op = ins.op
        a = self.phys_val[e.srcs[0]] if len(e.srcs) > 0 else 0
        b = self.phys_val[e.srcs[1]] if len(e.srcs) > 1 else 0
        if op is isa.Op.ADD:
            v, base, en = _s32(a + b), 1, ENERGY["alu"]
        elif op is isa.Op.SUB:
            v, base, en = _s32(a - b), 1, ENERGY["alu"]
        elif op is isa.Op.ADDI:
            v, base, en = _s32(a + ins.imm), 1, ENERGY["alu"]
        elif op is isa.Op.MOV:
            v, base, en = a, 1, ENERGY["alu"]
        elif op is isa.Op.NOP:
            v, base, en = None, 1, ENERGY["alu"] * 0.2
        elif op is isa.Op.MUL:
            v, base, en = _s32(a * b), 3, ENERGY["mul"]
        elif op is isa.Op.DIV:
            v, base, en = _s32(a // b) if b != 0 else 0, 6, ENERGY["div"]
        elif op in (isa.Op.VADD, isa.Op.VMUL):
            eligible = self.control.accelerator_mode in (
                AcceleratorMode.AGGRESSIVE, AcceleratorMode.SELECTIVE)
            if eligible:
                use_accel = True
                base, en = 2, ENERGY["vec_accel"]
            else:
                base, en = 4, ENERGY["vec"]
            v = _s32(a + b) if op is isa.Op.VADD else _s32(a * b)
        else:
            v, base, en = None, 1, 0.0
        lat = max(1, math.ceil(base * ls))
        extra = 0
        if use_accel:
            if self.cycle - self._accel_last > 16:
                extra, en = 2, en + ENERGY["accel_setup"]
            self._accel_last = self.cycle
            self.st["loads"] = self.st.get("loads", 0)
        return v, lat + extra, en * es, use_accel

    # ---- issue ----
    def _issue(self, iw: int) -> int:
        if self.control.schedule_policy.value == "age_ordered":
            order = sorted(self.rs, key=lambda r: r.age)
        else:
            order = list(self.rs)
        units = {"int": 2, "mul": 1, "div": 1, "vec": 1, "lsu": 1, "bru": 1}
        if self.control.power_mode is PowerMode.ECO:  # ECO narrows backend
            units = {k: 1 for k in units}
        used = {k: 0 for k in units}
        dispatched = 0
        for r in order:
            if dispatched >= iw:
                break
            if used[r.unit] >= units[r.unit]:
                continue
            e = self.rob[r.rob_pos]
            if not self._rs_ready(r):
                continue
            if r.unit == "lsu":
                verdict = self._lsu_ready(e, r.rob_pos)
                if verdict == "fwd":  # store-to-load forwarded: retire RS entry
                    r.rob_pos = -1
                    continue
                if not verdict:
                    continue
            if self._dispatch(e, r):
                used[r.unit] += 1
                dispatched += 1
        # remove dispatched (marked rob_pos -1)
        self.rs = [r for r in self.rs if r.rob_pos >= 0]
        if self.rs and dispatched == 0 and not self.executing:
            # nothing ready and nothing in flight: dependence stall unless drained
            if any(not e.done for e in self.rob[self.rob_head:]):
                self.st["dep_stall"] += 1
        return dispatched

    def _lsu_ready(self, e: ROBEntry, rob_pos: int) -> bool | str:
        """Tri-state: True (dispatch now), False (stall), 'fwd' (forwarded, done)."""
        ins = e.ins
        if ins.op is isa.Op.PREFETCH:
            return True
        if ins.op is isa.Op.STORE:
            if self._older_fence(rob_pos):
                self.st["fence_stall"] += 1
                return False
            return True  # addr calc always possible; ordering enforced by commit
        # LOAD: memory disambiguation against older stores + fences
        for pos in self.lsq:
            if pos >= rob_pos:
                break
            o = self.rob[pos]
            if o.is_fence:
                self.st["fence_stall"] += 1
                return False
            if o.is_store:
                if o.addr is None:
                    return False  # older store address unknown: stall
                a = (self.phys_val[e.srcs[0]] + ins.imm) % self.cfg.mem_words if e.srcs else ins.imm % self.cfg.mem_words
                if o.addr == a:
                    if len(o.srcs) > 1 and not self.phys_ready[o.srcs[1]]:
                        return False  # store data not ready yet
                    e.value = self.phys_val[o.srcs[1]] if len(o.srcs) > 1 else 0
                    # Materialize into the destination physical register now:
                    # commit reads phys_val (not e.value), and this entry will
                    # never pass through _complete.
                    if e.dst_phys >= 0:
                        self.phys_val[e.dst_phys] = e.value
                        self.phys_ready[e.dst_phys] = True
                    e.done = True
                    self.st["forwards"] += 1
                    self._act[1] += ENERGY["lsq_fwd"]
                    return "fwd"
        # MSHR capacity pre-check for likely misses
        a = (self.phys_val[e.srcs[0]] + ins.imm) % self.cfg.mem_words if e.srcs else 0
        line = a // self.cfg.line_words
        if (line not in self.memory.mshr and len(self.memory.mshr) >= self.memory.mshr_cap
                and not self.memory.l1d.probe(line)):
            self.st["mshr_stall"] += 1
            return False
        return True

    def _dispatch(self, e: ROBEntry, r: RSEntry) -> bool:
        ins = e.ins
        ls = self._lscale()
        if e.done:  # forwarded loads
            r.rob_pos = -1
            return False
        if r.unit in ("int", "mul", "div", "vec"):
            v, lat, en, accel = self._compute_alu(e)
            e.value = v
            e.exec_cycles = lat
            self.executing.append([r.rob_pos, lat])
            self._act[2 if accel else 0] += en
            r.rob_pos = -1
            return True
        if r.unit == "lsu":
            if ins.op is isa.Op.PREFETCH:
                a = (self.phys_val[e.srcs[0]] + ins.imm) % self.cfg.mem_words if e.srcs else 0
                self.memory._prefetch(a // self.cfg.line_words, max(1, self.control.prefetch_degree))
                e.done = True
                r.rob_pos = -1
                return False
            a = (self.phys_val[e.srcs[0]] + ins.imm) % self.cfg.mem_words if e.srcs else 0
            e.addr = a
            # Capture store DATA at dispatch (sources guaranteed ready): the
            # phys file may be recycled before commit, the ROB entry may not.
            if ins.op is isa.Op.STORE:
                e.value = self.phys_val[e.srcs[1]] if len(e.srcs) > 1 else 0
            self._mem_lines.append(a // self.cfg.line_words)
            if len(self._mem_lines) > 64:
                self._mem_lines = self._mem_lines[-64:]
            if ins.op is isa.Op.STORE:
                e.exec_cycles = max(1, math.ceil(1 * ls))
                self.executing.append([r.rob_pos, e.exec_cycles])
                self._act[1] += ENERGY["l1_hit"] * 0.5
                r.rob_pos = -1
                return True
            # LOAD executes cache access now (speculative fill allowed)
            res = self.memory.load(a // self.cfg.line_words, self.control.prefetch_degree, self._escale())
            if res.energy == 0.0 and not res.l1_hit:
                self.st["mshr_stall"] += 1
                return False
            e.value = self.mem[a]
            e.exec_cycles = max(1, math.ceil(res.latency * (0.6 + 0.4 * ls)))
            self.executing.append([r.rob_pos, e.exec_cycles])
            self._act[1] += res.energy
            r.rob_pos = -1
            return True
        if r.unit == "bru":
            lat = max(1, math.ceil(1 * ls))
            e.exec_cycles = lat
            # Capture operands now (P2): resolve runs at completion, a cycle
            # later, by which the src phys may be recycled (deep pile-ups).
            e.src_vals = tuple(self.phys_val[s] for s in e.srcs)
            self.executing.append([r.rob_pos, lat])
            self._act[0] += ENERGY["branch"]
            r.rob_pos = -1
            return True
        return False

    # ---- branch resolve ----
    def _resolve_branch(self, e: ROBEntry) -> None:
        ins = e.ins
        if ins.op is isa.Op.JUMP:
            e.taken_actual = True
        else:
            sv = e.src_vals if e.src_vals is not None else tuple(
                self.phys_val[s] for s in e.srcs)
            a = sv[0] if len(sv) > 0 else 0
            b = sv[1] if len(sv) > 1 else 0
            e.taken_actual = (a == b) if ins.op is isa.Op.BEQ else (a != b)
        e.mispredicted = (e.taken_pred is not None) and (e.taken_pred != e.taken_actual)
        # NOTE: speculation-OFF branches carry taken_pred=None and can never
        # mispredict: nothing was fetched on a predicted path, so there is
        # nothing to flush and no bubble to pay. Squash is strictly a
        # speculation-recovery mechanism.
        actual_target = e.pc + ins.imm if e.taken_actual else e.pc + 1
        if e.taken_actual:
            self.btb.install(e.pc, actual_target)
        if e.taken_actual and e.taken_pred is None:
            # Non-speculated taken branch (spec-OFF, or fetched under an
            # override): fetch went down fall-through and must redirect.
            # Younger ROB entries are off-path: drop them. This is NOT a
            # mispredict (nothing was speculated), so no squash counters or
            # bubbles — just a redirect.
            # NOTE: identity search, NOT list.index (ROBEntry __eq__ collides
            # across loop iterations with identical pc+ins).
            # NOTE 2: this runs inside _advance_exec's loop over the OLD
            # executing list, so executing slots are retired via the
            # _drop_redirect set (checked by the loop) rather than by
            # reassigning self.executing here (which the loop would clobber).
            pos = next(i for i in range(len(self.rob)) if self.rob[i] is e)
            doomed = set(range(pos + 1, len(self.rob)))
            self._drop_redirect |= doomed
            self.rob = self.rob[:pos + 1]
            keep = set(range(self.rob_head, pos + 1))
            self.rs = [r for r in self.rs if r.rob_pos in keep]
            self.lsq = [p for p in self.lsq if p in keep]
            self.fetch_queue.clear()
            self.fetch_pc = actual_target
            self.fetch_halted = False
            self._rebuild_free_list()
        self._predictor_commit(e.pc, e.taken_actual, e.taken_pred)
        e.done = True
        e.value = actual_target  # reuse for squash redirect

    def _predictor_commit(self, pc: int, taken: bool, predicted: bool | None) -> None:
        p = self.predictor
        if p.name == "none" or predicted is None:
            p.update(pc, taken)
            return
        if p.name == "bimodal":
            p.commit(pc, taken, predicted)
        elif p.name in ("gshare", "tournament"):
            p.commit(pc, taken, predicted, p.history)
        elif p.name == "tage_lite":
            p.commit(pc, taken, predicted, p.base[pc & p.mask] >= 2)

    def _rebuild_free_list(self) -> None:
        # P1 fix: live = rat + in-flight dsts + in-flight SRCS. Sources are
        # invisible to the old (rat+dsts) rule, so a rebuild could recycle a
        # dead-but-ready phys still referenced by an undispatched RS entry;
        # its ready bit would flip False with no producer left -> deadlock.
        # P2: pending-reclaim phys are live until the sweep releases them
        # (else they would be double-freed here and double-allocated later).
        # P3: repair ready bits from scratch. A dropped pre-completion writer
        # leaves ready[D]=False with no producer (squash/redirect orphan);
        # conversely a dropped post-completion writer leaves a speculative
        # value with ready=True. Derive readiness: youngest kept owner wins;
        # ownerless live phys hold committed values -> ready.
        live = set(self.rat) | set(self._pending_free)
        owner_done: dict[int, bool] = {}
        for i in range(self.rob_head, len(self.rob)):
            o = self.rob[i]
            if o.dst_phys >= 0:
                live.add(o.dst_phys)
                owner_done[o.dst_phys] = o.done
            live.update(o.srcs)
        self.free_list = [p for p in range(self.nphys) if p not in live]
        for d in live:
            self.phys_ready[d] = owner_done.get(d, True)

    def _reclaim(self) -> None:
        # P2: release pending prev-phys whose last undispatched consumer has
        # dispatched (dispatched entries captured values; only RS matters).
        if not self._pending_free:
            return
        needed: set[int] = set()
        for r in self.rs:
            if 0 <= r.rob_pos < len(self.rob):
                needed.update(self.rob[r.rob_pos].srcs)
        still = [p for p in self._pending_free if p in needed]
        for p in self._pending_free:
            if p not in needed and p not in self.free_list:
                self.free_list.append(p)
        self._pending_free = still

    def _squash(self, branch_pos: int) -> None:
        e = self.rob[branch_pos]
        self.st["squashes"] += 1
        bubble = MISPREDICT_PENALTY if self.control.speculation_mode.value == "conservative" \
            else max(0, MISPREDICT_PENALTY - 2)
        self.fetch_bubble = max(self.fetch_bubble, bubble)
        self.st["flush_bubbles"] += bubble
        self._act[0] += ENERGY["mispredict"]
        # restore rename state
        self.rat = list(e.checkpoint)
        # truncate structures to <= branch_pos
        keep = {i for i in range(self.rob_head, branch_pos + 1)}
        self.rob = self.rob[:branch_pos + 1]
        self.rs = [r for r in self.rs if r.rob_pos in keep]
        self.lsq = [p for p in self.lsq if p in keep]
        self.executing = [ex for ex in self.executing if ex[0] in keep]
        self.fetch_queue.clear()
        self.fetch_pc = e.value
        self.fetch_halted = False
        # rebuild free list exactly: all phys not live
        self._rebuild_free_list()

    # ---- rename/decode ----
    def _rename(self, dw: int) -> None:
        n = 0
        while n < dw and self.fetch_queue and len(self.rob) - self.rob_head < self.cfg.rob_size:
            fq = self.fetch_queue[0]
            ins: isa.Instruction = fq["ins"]
            if len(self.rs) >= self.cfg.rs_size:
                self.st["rs_full"] += 1
                break
            if ins.is_mem and len(self.lsq) >= self.cfg.lsq_size:
                self.st["lsq_full"] += 1
                break
            if ins.writes and not self.free_list:
                self.st["rob_full"] += 1
                break
            self.fetch_queue.pop(0)
            pos = len(self.rob)
            e = ROBEntry(fq["pc"], ins)
            # rename sources
            e.srcs = tuple(self.rat[r] for r in ins.reads)
            # allocate dest
            if e.dst >= 0:
                e.prev_phys = self.rat[e.dst]
                e.dst_phys = self.free_list.pop()
                self.phys_ready[e.dst_phys] = False
                self.rat[e.dst] = e.dst_phys
            if e.is_branch:
                e.checkpoint = list(self.rat)
                e.taken_pred = fq["pred_taken"]
            if e.is_halt or e.is_fence:
                e.done = e.is_halt or False
                if e.is_halt:
                    e.done = True
            self.rob.append(e)
            if e.is_mem:
                self.lsq.append(pos)
            unit = self._unit_for(ins)
            if not (e.is_halt or e.is_fence):
                self.rs.append(RSEntry(pos, unit, self._age))
                self._age += 1
            else:
                if e.is_fence:
                    e.done = True  # ordering enforced via _older_fence presence checks
            self._act[0] += ENERGY["decode"] + ENERGY["rename"]
            n += 1
        if self.fetch_queue and len(self.rob) - self.rob_head >= self.cfg.rob_size:
            self.st["rob_full"] += 1

    # ---- fetch ----
    def _fetch(self, fw: int) -> None:
        if self.fetch_bubble > 0:
            self.fetch_bubble -= 1
            return
        if self.fetch_halted:
            return
        n = 0
        while n < fw and len(self.fetch_queue) < 2 * fw + 4:
            if not 0 <= self.fetch_pc < len(self.program):
                if self.fetch_pc >= len(self.program):
                    self.fetch_halted = True
                break
            line = self.fetch_pc // self.cfg.line_words
            lat, hit = self.memory.fetch_line(line)
            self._act[0] += ENERGY["fetch"] + ENERGY["predictor_access"] + ENERGY["btb_access"]
            if not hit:
                self.st["l1i_miss"] += 1
                self.fetch_bubble = max(self.fetch_bubble, lat - 1)
                self.st["icache_stall"] += lat
                self._act[1] += ENERGY["dram"] * 0.5
                break
            ins = self.program[self.fetch_pc]
            if ins.is_branch and self.control.speculation_mode is SpeculationMode.OFF \
                    and self._branch_in_flight():
                self.st["fetch_starve"] += 1
                break
            pred_taken = None
            if ins.is_branch and self.control.speculation_mode is not SpeculationMode.OFF:
                if ins.op is isa.Op.JUMP:
                    pred_taken = True
                else:
                    pred_taken = bool(self.predictor.predict(self.fetch_pc))
            self.fetch_queue.append({"pc": self.fetch_pc, "ins": ins, "pred_taken": pred_taken})
            if ins.is_halt:
                self.fetch_halted = True
                self.fetch_pc += 1
                n += 1
                break
            self.fetch_pc = self.fetch_pc + ins.imm if (ins.is_branch and pred_taken) else self.fetch_pc + 1
            n += 1
            if ins.is_branch and self.control.speculation_mode is SpeculationMode.OFF:
                break  # non-speculative fetch stops at every branch (P0 fix:
                       # else fall-through floods the window and a taken
                       # resolve has nothing to redirect)
        if n == 0 and not self.fetch_halted and self.fetch_bubble == 0:
            self.st["fetch_starve"] += 1

    # ---- run / result ----
    def run(self, control_plan: list[ControlBundle] | None = None, interval: int = 8) -> dict:
        idx = 0
        while not self.halted and self.cycle < self.cfg.max_cycles:
            if control_plan and self.cycle % interval == 0 and idx < len(control_plan):
                self.set_control(control_plan[idx])
                idx += 1
            try:
                self.step_cycle()
            except RuntimeError as err:
                if "deadlock" in str(err):
                    return self.result("DEADLOCK")
                break
        return self.result("HALTED" if self.halted else "TIMEOUT")

    def result(self, status: str | None = None) -> dict[str, Any]:
        s = status or ("HALTED" if self.halted else "TIMEOUT")
        total = round(self.energy_dyn, 6)
        ctrl = round(self.controller_energy, 6)
        m = {
            "completion_status": s, "cycles": self.cycle, "instrs": self.retired,
            "ipc": self.retired / self.cycle if self.cycle else 0.0,
            "energy": total, "controller_energy": ctrl,
            "l1d_miss_rate": self.memory.l1d_miss_rate,
            "l1i_miss_rate": self.memory.l1i_miss_rate,
            "branch_mispredict_rate": self.predictor.rate,
            "mpki": (self.predictor.mispredictions / max(self.retired, 1)) * 1000,
            "prefetch_accuracy": self.memory.prefetch_accuracy,
            "prefetches": self.memory.prefetches_issued,
            "forwards": self.st["forwards"], "squashes": self.st["squashes"],
            "row_hit_rate": self.memory.dram.row_hit_rate,
            "peak_temp": round(self.peak_temperature, 4),
            "thermal_exposure": round(self.thermal_exposure, 6),
            "safety_overrides": self.safety_overrides,
            "action_switches": self.action_switches,
            "action_histogram": dict(self.action_histogram),
            "stalls": dict(self.st),
            "exit_code": self.exit_code,
            "config_digest": self.cfg.digest(),
        }
        m["objective"] = (m["cycles"] + 0.08 * (m["energy"] - m["controller_energy"])
                          + 0.6 * m["thermal_exposure"] + 0.02 * m["controller_energy"])
        return m
