"""Lowering map v2: C source -> flag presets -> ISA programs.

Status: a MODEL, not a compiler (see docs/CRITIQUE.md P0-1/Gap-1). It
hand-maps two real compiler flag presets (from the layer's own
``FLAG_ACTIONS`` vocabulary) to ISA programs implementing the same
computation with different instruction mixes — the analogue of what
-O0 vs -O3 do to this loop (spill-heavy scalar vs blocked vector).

v2 changes (docs/DEEPENING.md round 2): parameterized problem size N
(genuine timing variance across N; per-seed address offsets were
measured to have NO effect — fully-associative LRU keyed by access
order — and removed). Both programs execute on the exact CPU oracles
(vmicro in-order AND vmarch OoO, whose ISA is a superset with identical
semantics on shared ops) to HALT, with functional equivalence checked
bit-for-bit per (seed, N).

Fixed workload family: integer dot product of length N.
  C: int dot(int *a, int *b) { int s=0; for (i<N) s += a[i]*b[i]; }
Vectors at MEM[100..] (a) and MEM[200..] (b), values 1..9 seeded.
Accumulator in R3, result also stored at MEM[OUT_ADDR].
All immediates < 1024 (vmarch 11-bit bound).
"""
from __future__ import annotations

import hashlib
import random

LOWERING_VERSION = "lowering-v2"
N_DEFAULT = 8
A_BASE = 100
B_BASE = 200
SPILL_ADDR = 900
OUT_ADDR = 910

#: The two presets, using the compiler layer's own flag vocabulary.
PRESETS = ("O0", "O3")


def c_source(n: int) -> str:
    return (
        "int dot(int *a, int *b) {\n"
        "  int s = 0;\n"
        f"  for (int i = 0; i < {n}; i++) s += a[i] * b[i];\n"
        "  return s;\n"
        "}\n"
    )


C_SOURCE = c_source(N_DEFAULT)


def vector_memory(seed: int, n: int) -> tuple[dict[int, int], int]:
    """Deterministic operand vectors + expected dot product."""
    rng = random.Random(seed)
    mem: dict[int, int] = {}
    expected = 0
    for i in range(n):
        a = rng.randint(1, 9)
        b = rng.randint(1, 9)
        mem[A_BASE + i] = a
        mem[B_BASE + i] = b
        expected += a * b
    return mem, expected


def lower_O0(n: int) -> list[str]:
    """Unoptimized: scalar element-at-a-time with acc spill/reload."""
    lines = ["ADDI R3, R0, 0"]
    for i in range(n):
        lines += [
            f"LOAD R5, R0, {A_BASE + i}",
            f"LOAD R6, R0, {B_BASE + i}",
            "MUL R7, R5, R6",
            "ADD R3, R3, R7",
            f"STORE R0, R3, {SPILL_ADDR}",
            f"LOAD R3, R0, {SPILL_ADDR}",
        ]
    lines += [f"STORE R0, R3, {OUT_ADDR}", "HALT"]
    return lines


def lower_O3(n: int) -> list[str]:
    """Optimized: 2-wide blocks, vector mul/add, single spill-free acc."""
    assert n % 2 == 0, "lowering-v2 O3 blocks 2-wide; use even N"
    lines = ["ADDI R3, R0, 0"]
    for i in range(0, n, 2):
        lines += [
            f"LOAD R5, R0, {A_BASE + i}",
            f"LOAD R6, R0, {B_BASE + i}",
            "VMUL R7, R5, R6",
            f"LOAD R8, R0, {A_BASE + i + 1}",
            f"LOAD R9, R0, {B_BASE + i + 1}",
            "VMUL R10, R8, R9",
            "VADD R11, R7, R10",
            "ADD R3, R3, R11",
        ]
    lines += [f"STORE R0, R3, {OUT_ADDR}", "HALT"]
    return lines


def lower(preset: str, n: int = N_DEFAULT) -> list[str]:
    if preset == "O0":
        return lower_O0(n)
    if preset == "O3":
        return lower_O3(n)
    raise ValueError(f"{LOWERING_VERSION} covers presets {PRESETS}, got {preset!r}")


def op_histogram(assembled) -> dict[str, int]:
    """Count ops by class from an assembled program."""
    hist: dict[str, int] = {}
    for ins in assembled:
        key = "mem" if ins.is_mem else ("vector" if ins.is_vector else ("alu" if ins.writes or ins.reads else "ctrl"))
        hist[key] = hist.get(key, 0) + 1
        hist["total"] = hist.get("total", 0) + 1
    return hist


def translate_to_vmarch(vmicro_prog) -> list:
    """Map vmicro instructions onto the vmarch superset ISA (same semantics)."""
    from vmarch import isa as misa

    return [misa.Instruction(misa.Op[i.op.name], i.rd, i.rs1, i.rs2, i.imm)
            for i in vmicro_prog]


def lowering_digest(ns: tuple[int, ...] = (4, 8, 12)) -> str:
    h = hashlib.sha256()
    h.update(LOWERING_VERSION.encode())
    for n in ns:
        h.update(b"\x00" + "\n".join(lower_O0(n)).encode())
        h.update(b"\x00" + "\n".join(lower_O3(n)).encode())
        h.update(b"\x00" + c_source(n).encode())
    return h.hexdigest()[:12]
