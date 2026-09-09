"""Workload families: VM-ISA programs with phase structure and held-out discipline.

Families mirror the DEA stress taxonomy at ISA level:
- stream   : sequential vector loads + VEC chains (prefetch-sensitive)
- pointer  : serialized dependent loads (ILP-bound; honest negative result)
- branch   : chaotic branches with predictor aliasing (speculation-sensitive)
- mixed    : gemm -> barrier -> irregular transitions
- phase    : macro-phase program stream->pointer->branch with HIDDEN labels,
             the zero-shot-context analogue: the policy never sees phase ids.

All generators are seeded; every instance differs (no memorization).
Held-out splits use disjoint seed ranges; overlap is refused by bench.py.
"""
from __future__ import annotations

import random

from vmarch import isa
from vmarch.isa import assemble


def stream_like(seed: int = 0, n: int = 40) -> tuple:
    rng = random.Random(seed)
    lines = ["ADDI R1, R0, 0", "ADDI R2, R0, 0"]
    base = rng.randrange(0, 64)
    for i in range(n):
        a0 = (base + i * 4) % 1008
        a1 = (base + i * 4 + 16) % 1008
        lines += [f"LOAD R3, R1, {a0}", f"LOAD R4, R1, {a1}",
                  "VADD R5, R3, R4", "VMUL R6, R5, R3", "ADD R2, R2, R6"]
        if rng.random() < 0.1:
            lines.append(f"ADDI R1, R1, {rng.randint(0, 2)}")
    return assemble(lines + ["STORE R0, R2, 1000", "HALT"]), \
        {a: (a * 7 + 3) % 1000 for a in range(1024)}, {}, f"stream n={n} seed={seed}"


def pointer_like(seed: int = 0, n: int = 20) -> tuple:
    rng = random.Random(seed)
    addrs = rng.sample(range(0, 1000), n + 1)
    mem = {a: addrs[(i + 1) % (n + 1)] for i, a in enumerate(addrs)}
    lines = [f"ADDI R1, R0, {addrs[0]}", "ADDI R3, R0, 0"]
    for _ in range(n):
        lines += ["LOAD R2, R1, 0", "ADD R1, R2, R0", "ADDI R3, R3, 1"]
    return assemble(lines + ["STORE R1, R3, 1001", "HALT"]), mem, {}, f"pointer n={n} seed={seed}"


def branch_like(seed: int = 0, n: int = 36) -> tuple:
    rng = random.Random(seed)
    lines = ["ADDI R1, R0, 0", "ADDI R2, R0, 500"]
    for _ in range(n):
        v, w = rng.randint(0, 5), rng.randint(0, 5)
        lines += [f"ADDI R3, R0, {v}", f"ADDI R4, R0, {w}",
                  "BEQ R3, R4, 2", "ADDI R1, R1, 1", "ADDI R1, R1, 2",
                  "BNE R3, R0, 2", "ADD R2, R2, R1", "SUB R2, R2, R1"]
    return assemble(lines + ["STORE R0, R1, 1002", "HALT"]), {}, {}, f"branch n={n} seed={seed}"


def mixed_like(seed: int = 0) -> tuple:
    p1, m1, _, _ = stream_like(seed, 12)
    p2, m2, _, _ = branch_like(seed + 1, 10)
    p3, m3, _, _ = pointer_like(seed + 2, 8)
    prog = p1[:-1] + p2[:-1] + p3
    mem = dict(m1)
    mem.update(m3)
    return prog, mem, {}, f"mixed seed={seed}"


def phase_changing(seed: int = 0) -> tuple:
    """Three serialized macro-phases, labels hidden from the policy."""
    rng = random.Random(seed)
    order = ["stream", "pointer", "branch"]
    rng.shuffle(order)
    gens = {"stream": lambda: stream_like(seed * 31 + 1, 14),
            "pointer": lambda: pointer_like(seed * 31 + 2, 8),
            "branch": lambda: branch_like(seed * 31 + 3, 12)}
    prog_all: list = []
    mem: dict = {}
    for ph in order:
        p, m, _, _ = gens[ph]()
        prog_all += p[:-1]
        mem.update(m)
    prog_all.append(isa.Instruction(isa.Op.HALT))
    return prog_all, mem, {}, f"phase order={order} seed={seed}"


# ---- T1: looping workloads (hot I-cache, reuse, trainable branches) ----
def loop_stream(seed: int = 0, iters: int = 48) -> tuple:
    """Counted loop over a striding vector body. Working set (~2 lines/iter)
    exceeds L1D but fits L2 at default geometry: honest L1-miss/L2-hit regime
    with prefetch leverage. Backward branch trains the predictor."""
    rng = random.Random(seed)
    base = rng.randrange(0, 64)
    lines = ["ADDI R1, R0, %d" % base, "ADDI R2, R0, 0",
             "ADDI R7, R0, %d" % iters,
             "LOAD R3, R1, 0", "LOAD R4, R1, 16",
             "VADD R5, R3, R4", "VMUL R6, R5, R3", "ADD R2, R2, R6",
             "ADDI R1, R1, 4", "ADDI R7, R7, -1",
             "BNE R7, R0, %d" % (3 - 10),
             "STORE R0, R2, 1000", "HALT"]
    mem = {a: (a * 7 + 3) % 1000 for a in range(1024)}
    return assemble(lines), mem, {}, f"loop_stream iters={iters} seed={seed}"


def loop_pointer(seed: int = 0, hops: int = 32) -> tuple:
    """Counted linked-list traversal: ILP-bound even when hot."""
    rng = random.Random(seed)
    addrs = rng.sample(range(0, 1000), hops + 1)
    mem = {a: addrs[(i + 1) % (hops + 1)] for i, a in enumerate(addrs)}
    lines = ["ADDI R1, R0, %d" % addrs[0], "ADDI R3, R0, 0",
             "ADDI R7, R0, %d" % hops,
             "LOAD R2, R1, 0", "ADD R1, R2, R0", "ADDI R3, R3, 1",
             "ADDI R7, R7, -1", "BNE R7, R0, %d" % (3 - 7),
             "STORE R1, R3, 1001", "HALT"]
    return assemble(lines), mem, {}, f"loop_pointer hops={hops} seed={seed}"


def loop_branch(seed: int = 0, outer: int = 10, inner: int = 8) -> tuple:
    """Outer loop around chaotic inner branches: predictor training signal
    (mostly-taken loop exit + unpredictable inner) with hot I-cache."""
    rng = random.Random(seed)
    lines = ["ADDI R1, R0, 0", "ADDI R2, R0, 500", "ADDI R7, R0, %d" % outer]
    for _ in range(inner):
        v, w = rng.randint(0, 5), rng.randint(0, 5)
        lines += ["ADDI R3, R0, %d" % v, "ADDI R4, R0, %d" % w,
                  "BEQ R3, R4, 2", "ADDI R1, R1, 1", "ADDI R1, R1, 2",
                  "BNE R3, R0, 2", "ADD R2, R2, R1", "SUB R2, R2, R1"]
    lines += ["ADDI R7, R7, -1"]
    bne_idx = len(lines)
    lines += ["BNE R7, R0, %d" % (3 - bne_idx), "STORE R0, R1, 1002", "HALT"]
    return assemble(lines), {}, {}, \
        f"loop_branch outer={outer} inner={inner} seed={seed}"


FAMILIES = {"stream": stream_like, "pointer": pointer_like,
            "branch": branch_like, "mixed": mixed_like, "phase": phase_changing,
            "loop_stream": loop_stream, "loop_pointer": loop_pointer,
            "loop_branch": loop_branch}

HELD_OUT = {"train": (0, 64), "select": (1000, 1032), "test": (10000, 10100)}


def check_disjoint(*ranges: tuple[int, int]) -> None:
    sets = [set(range(a, b)) for a, b in ranges]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise ValueError(f"seed range overlap: {ranges[i]} vs {ranges[j]}")
