"""Tiny assembler: text -> list[Instruction]."""
from __future__ import annotations

from vmicro.isa import Instruction, Op


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
        args = parts[1:]
        if op is Op.NOP or op is Op.HALT:
            prog.append(Instruction(op))
        elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.VADD, Op.VMUL):
            prog.append(Instruction(op, _reg(args[0]), _reg(args[1]), _reg(args[2])))
        elif op is Op.ADDI:
            prog.append(Instruction(op, _reg(args[0]), _reg(args[1]), 0, int(args[2], 0)))
        elif op is Op.MOV:
            prog.append(Instruction(op, _reg(args[0]), _reg(args[1])))
        elif op is Op.LOAD:
            prog.append(Instruction(op, _reg(args[0]), _reg(args[1]), 0, int(args[2], 0)))
        elif op is Op.STORE:
            prog.append(Instruction(op, 0, _reg(args[0]), _reg(args[1]), int(args[2], 0)))
        elif op in (Op.BEQ, Op.BNE):
            prog.append(Instruction(op, 0, _reg(args[0]), _reg(args[1]), int(args[2], 0)))
        elif op is Op.JUMP:
            prog.append(Instruction(op, imm=int(args[0], 0)))
        else:
            raise ValueError(f"unsupported op: {op}")
    return prog


def disassemble(prog: list[Instruction]) -> list[str]:
    return [str(i) for i in prog]
