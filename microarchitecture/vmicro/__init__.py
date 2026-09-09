"""Virtual Microprocessor: executable, branchable surrogate of computation.

Grounding in ``context.txt`` + ``dynamic-execution``:
- context.txt taxonomy: Virtual Computer = Compiler + OS + microarchitecture + I/O;
  Virtual Algorithm = schedulers/TCP/compilers; common operation
  learn -> branch -> search -> act -> validate.
- dynamic-execution (DEA) thesis: a global controller plans how valid computation
  flows through a heterogeneous machine WITHOUT rewriting program semantics;
  hard logic preserves correctness, thermal safety, retirement.
- This package instantiates that for a real ISA: an exact oracle CPU (fetch /
  issue / cache / branch predictor / power / thermal, all causal and
  snapshotable) wrapped in a branchable virtual environment with a learned
  timing surrogate and budget-aware search (the Virtual Chess "killer benchmark":
  finite oracle calls, unlimited surrogate calls, metric = result per oracle call).
"""
from vmicro.isa import Op, Instruction
from vmicro.machine import CPU, ControlBundle, ACTION_LIBRARY, SafetyGovernor
from vmicro.virtual import VirtualMicroprocessor

__all__ = ["Op", "Instruction", "CPU", "ControlBundle", "ACTION_LIBRARY", "SafetyGovernor", "VirtualMicroprocessor"]

__version__ = "0.1.0"
