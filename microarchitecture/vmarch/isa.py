"""VM-ISA-1: instruction set with formal encoding, assembler, validator.

Superset of the vmicro ISA (all 15 ops, identical semantics) plus:
  FENCE    - memory barrier: drains LSQ (younger mem ops wait for older)
  PREFETCH - cache hint: MEM[rs1+imm] line touched, non-binding, no faults
  ECALL    - halt with exit code regs[rs1] (for workload harnesses)

32-bit encoding: op:6 | rd:5 | rs1:5 | rs2:5 | imm:11 signed (-1024..1023).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

OPCODE_BITS = 6
IMM_MIN, IMM_MAX = -1024, 1023


class Op(str, Enum):
    NOP = "NOP"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    ADDI = "ADDI"
    MOV = "MOV"
    LOAD = "LOAD"
    STORE = "STORE"
    VADD = "VADD"
    VMUL = "VMUL"
    BEQ = "BEQ"
    BNE = "BNE"
    JUMP = "JUMP"
    HALT = "HALT"
    FENCE = "FENCE"
    PREFETCH = "PREFETCH"
    ECALL = "ECALL"


OPCODE = {op: i for i, op in enumerate(Op)}


@dataclass(frozen=True)
class Instruction:
    op: Op
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0

    def __post_init__(self) -> None:
        for name in ("rd", "rs1", "rs2"):
            v = getattr(self, name)
            if not 0 <= v <= 31:
                raise ValueError(f"{name} must be 0..31, got {v}")
        if not IMM_MIN <= self.imm <= IMM_MAX:
            raise ValueError(f"imm must be {IMM_MIN}..{IMM_MAX}, got {self.imm}")

    @property
    def reads(self) -> tuple[int, ...]:
        if self.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.VADD, Op.VMUL):
            return (self.rs1, self.rs2)
        if self.op in (Op.ADDI, Op.MOV):
            return (self.rs1,)
        if self.op in (Op.BEQ, Op.BNE):
            return (self.rs1, self.rs2)
        if self.op is Op.LOAD:
            return (self.rs1,)
        if self.op is Op.STORE:
            return (self.rs1, self.rs2)
        if self.op is Op.PREFETCH:
            return (self.rs1,)
        if self.op is Op.ECALL:
            return (self.rs1,)
        return ()

    @property
    def writes(self) -> tuple[int, ...]:
        if self.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.ADDI, Op.MOV,
                       Op.LOAD, Op.VADD, Op.VMUL):
            return (self.rd,) if self.rd != 0 else ()
        return ()

    @property
    def is_branch(self) -> bool:
        return self.op in (Op.BEQ, Op.BNE, Op.JUMP)

    @property
    def is_mem(self) -> bool:
        return self.op in (Op.LOAD, Op.STORE)

    @property
    def is_vector(self) -> bool:
        return self.op in (Op.VADD, Op.VMUL)

    @property
    def is_halt(self) -> bool:
        return self.op in (Op.HALT, Op.ECALL)

    @property
    def is_fence(self) -> bool:
        return self.op is Op.FENCE

    @property
    def is_prefetch(self) -> bool:
        return self.op is Op.PREFETCH

    def encode(self) -> int:
        return ((OPCODE[self.op] & 0x3F) << 26) | ((self.rd & 31) << 21) | \
               ((self.rs1 & 31) << 16) | ((self.rs2 & 31) << 11) | (self.imm & 0x7FF)

    @classmethod
    def decode(cls, word: int) -> "Instruction":
        ops = list(Op)
        return cls(ops[(word >> 26) & 0x3F], (word >> 21) & 31,
                   (word >> 16) & 31, (word >> 11) & 31,
                   _sign11(word & 0x7FF))


def _sign11(v: int) -> int:
    return v - 0x800 if v & 0x400 else v


def validate_program(prog: list[Instruction]) -> list[str]:
    """Return list of warnings (empty = clean). Errors raise at construction."""
    warns: list[str] = []
    if not prog:
        raise ValueError("program must be non-empty")
    if prog[-1].op not in (Op.HALT, Op.ECALL):
        warns.append("program does not end with HALT/ECALL; execution stops at fall-through")
    for i, ins in enumerate(prog):
        if ins.is_branch:
            tgt = i + ins.imm if ins.op is not Op.JUMP else i + ins.imm
            if not 0 <= tgt < len(prog) and ins.op is not Op.JUMP:
                warns.append(f"branch at {i} targets {tgt} (out of bounds -> fall-through+halt)")
    return warns


def _reg(tok: str) -> int:
    tok = tok.strip().upper()
    if not tok.startswith("R"):
        raise ValueError(f"bad register: {tok}")
    n = int(tok[1:])
    if not 0 <= n <= 31:
        raise ValueError(f"register out of range: {tok}")
    return n


def assemble(lines: list[str] | str) -> list[Instruction]:
    if isinstance(lines, str):
        lines = lines.splitlines()
    prog: list[Instruction] = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        op = Op(parts[0].upper())
        a = parts[1:]
        if op in (Op.NOP, Op.HALT, Op.FENCE):
            prog.append(Instruction(op))
        elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.VADD, Op.VMUL):
            prog.append(Instruction(op, _reg(a[0]), _reg(a[1]), _reg(a[2])))
        elif op is Op.ADDI:
            prog.append(Instruction(op, _reg(a[0]), _reg(a[1]), 0, int(a[2], 0)))
        elif op is Op.MOV:
            prog.append(Instruction(op, _reg(a[0]), _reg(a[1])))
        elif op is Op.LOAD:
            prog.append(Instruction(op, _reg(a[0]), _reg(a[1]), 0, int(a[2], 0)))
        elif op is Op.STORE:
            prog.append(Instruction(op, 0, _reg(a[0]), _reg(a[1]), int(a[2], 0)))
        elif op in (Op.BEQ, Op.BNE):
            prog.append(Instruction(op, 0, _reg(a[0]), _reg(a[1]), int(a[2], 0)))
        elif op is Op.JUMP:
            prog.append(Instruction(op, imm=int(a[0], 0)))
        elif op is Op.PREFETCH:
            prog.append(Instruction(op, 0, _reg(a[0]), 0, int(a[1], 0)))
        elif op is Op.ECALL:
            prog.append(Instruction(op, 0, _reg(a[0])))
        else:
            raise ValueError(f"assembler: unsupported {op}")
    return prog


def disassemble(prog: list[Instruction]) -> list[str]:
    out = []
    for i in prog:
        if i.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.VADD, Op.VMUL):
            out.append(f"{i.op.value} R{i.rd}, R{i.rs1}, R{i.rs2}")
        elif i.op is Op.ADDI:
            out.append(f"ADDI R{i.rd}, R{i.rs1}, {i.imm}")
        elif i.op is Op.MOV:
            out.append(f"MOV R{i.rd}, R{i.rs1}")
        elif i.op is Op.LOAD:
            out.append(f"LOAD R{i.rd}, R{i.rs1}, {i.imm}")
        elif i.op is Op.STORE:
            out.append(f"STORE R{i.rs1}, R{i.rs2}, {i.imm}")
        elif i.op in (Op.BEQ, Op.BNE):
            out.append(f"{i.op.value} R{i.rs1}, R{i.rs2}, {i.imm}")
        elif i.op is Op.JUMP:
            out.append(f"JUMP {i.imm}")
        elif i.op is Op.PREFETCH:
            out.append(f"PREFETCH R{i.rs1}, {i.imm}")
        elif i.op is Op.ECALL:
            out.append(f"ECALL R{i.rs1}")
        else:
            out.append(i.op.value)
    return out
