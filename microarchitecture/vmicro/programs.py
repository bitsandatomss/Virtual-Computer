"""Program generators: ISA analogues of dynamic-execution workload families.

Each returns (program, mem_init, reg_init, description).
Deterministic given seed.
"""
from __future__ import annotations

import random

from vmicro.assembler import assemble


def matrix_like(seed: int = 0, n: int = 48) -> tuple:
    rng = random.Random(seed)
    lines = []
    # stream through memory rows doing vector ops: R1 base, R2 accumulator
    lines.append("ADDI R1, R0, 0")
    lines.append("ADDI R2, R0, 0")
    for i in range(n):
        addr = (i * 4) % 512
        lines.append(f"LOAD R3, R1, {addr}")
        lines.append(f"LOAD R4, R1, {(addr + 8) % 512}")
        lines.append("VADD R5, R3, R4")
        lines.append("VMUL R6, R5, R3")
        lines.append("ADD R2, R2, R6")
        if rng.random() < 0.15:
            lines.append(f"ADDI R1, R1, {rng.randint(1, 4)}")
    lines.append("STORE R1, R2, 900")
    lines.append("HALT")
    return assemble(lines), {a: (a * 7 + 3) % 256 for a in range(1024)}, {}, f"matrix_like n={n} seed={seed}"


def pointer_chase(seed: int = 0, n: int = 24) -> tuple:
    rng = random.Random(seed)
    # serialized dependent loads: R1 holds address, each load feeds next address
    addrs = rng.sample(range(0, 512), n)
    mem = {}
    for i, a in enumerate(addrs):
        mem[a] = addrs[(i + 1) % n]  # linked list
    lines = [f"ADDI R1, R0, {addrs[0]}"]
    for _ in range(n):
        lines += ["LOAD R2, R1, 0", "ADD R1, R2, R0", "ADDI R3, R3, 1"]
    lines += ["STORE R1, R3, 950", "HALT"]
    return assemble(lines), mem, {}, f"pointer_chase n={n} seed={seed}"


def branch_heavy(seed: int = 0, n: int = 40) -> tuple:
    rng = random.Random(seed)
    lines = ["ADDI R1, R0, 0", "ADDI R2, R0, 100"]
    for _ in range(n):
        v = rng.randint(0, 5)
        lines.append(f"ADDI R3, R0, {v}")
        lines.append(f"ADDI R4, R0, {rng.randint(0, 5)}")
        # branch taken ~ unpredictably; offsets skip 1-2 instrs
        lines.append(f"BEQ R3, R4, 2")
        lines.append("ADDI R1, R1, 1")
        lines.append("ADDI R1, R1, 2")
        lines.append(f"BNE R3, R0, 2")
        lines.append("ADD R2, R2, R1")
        lines.append("SUB R2, R2, R1")
    lines += ["STORE R0, R1, 960", "HALT"]
    return assemble(lines), {}, {}, f"branch_heavy n={n} seed={seed}"


def mixed(seed: int = 0) -> tuple:
    p1, m1, r1, _ = matrix_like(seed, 16)
    p2, m2, r2, _ = branch_heavy(seed + 1, 12)
    p3, m3, r3, _ = pointer_chase(seed + 2, 10)
    prog = p1[:-1] + p2[:-1] + p3  # strip intermediate HALTs
    mem = dict(m1); mem.update(m2); mem.update(m3)
    return prog, mem, {}, f"mixed seed={seed}"


FAMILIES = {"matrix": matrix_like, "pointer": pointer_chase, "branch": branch_heavy, "mixed": mixed}
