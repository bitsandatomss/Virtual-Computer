from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Op(str, Enum):
    NOP = "NOP"
    ADD = "ADD"       # rd = rs1 + rs2
    SUB = "SUB"       # rd = rs1 - rs2
    MUL = "MUL"       # rd = rs1 * rs2 (3c)
    DIV = "DIV"       # rd = rs1 // rs2 (6c, div-by-zero -> 0)
    ADDI = "ADDI"     # rd = rs1 + imm
    MOV = "MOV"       # rd = rs1
    LOAD = "LOAD"     # rd = MEM[rs1 + imm]
    STORE = "STORE"   # MEM[rs1 + imm] = rs2
    VADD = "VADD"     # rd = rs1 + rs2 (vector-ish, 4c, accel 2c)
    VMUL = "VMUL"     # rd = rs1 * rs2 (vector-ish, 4c, accel 2c)
    BEQ = "BEQ"       # if rs1 == rs2: pc += imm
    BNE = "BNE"       # if rs1 != rs2: pc += imm
    JUMP = "JUMP"     # pc += imm
    HALT = "HALT"


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
                raise ValueError(f"{name} must be register 0..31, got {v}")

    @property
    def reads(self) -> tuple[int, ...]:
        if self.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.VADD, Op.VMUL):
            return (self.rs1, self.rs2)
        if self.op in (Op.ADDI, Op.MOV, Op.BEQ, Op.BNE):
            # BEQ/BNE read rs1+rs2; ADDI/MOV read rs1
            return (self.rs1, self.rs2) if self.op in (Op.BEQ, Op.BNE) else (self.rs1,)
        if self.op is Op.LOAD:
            return (self.rs1,)
        if self.op is Op.STORE:
            return (self.rs1, self.rs2)
        return ()

    @property
    def writes(self) -> tuple[int, ...]:
        if self.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.ADDI, Op.MOV, Op.LOAD, Op.VADD, Op.VMUL):
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
