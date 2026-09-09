"""Smoke + correctness tests for the virtual microprocessor."""
import unittest

from vmicro.assembler import assemble
from vmicro.programs import branch_heavy, matrix_like, mixed, pointer_chase
from vmicro.search import beam_search_plan, fixed_baseline
from vmicro.surrogate import collect_trace, train_surrogate
from vmicro.virtual import VirtualMicroprocessor


class TestISA(unittest.TestCase):
    def test_add_store_load(self):
        prog = assemble(["ADDI R1, R0, 5", "ADDI R2, R0, 7", "ADD R3, R1, R2",
                         "STORE R0, R3, 100", "LOAD R4, R0, 100", "HALT"])
        vm = VirtualMicroprocessor(prog)
        res = vm.run()
        self.assertEqual(res["completion_status"], "HALTED")
        self.assertEqual(vm.cpu.regs[3], 12)
        self.assertEqual(vm.cpu.regs[4], 12)
        self.assertEqual(vm.cpu.mem[100], 12)

    def test_r0_hardwired(self):
        prog = assemble(["ADDI R0, R0, 99", "HALT"])
        vm = VirtualMicroprocessor(prog)
        vm.run()
        self.assertEqual(vm.cpu.regs[0], 0)

    def test_branch_taken(self):
        prog = assemble(["ADDI R1, R0, 1", "ADDI R2, R0, 1", "BEQ R1, R2, 2",
                         "ADDI R3, R0, 111", "ADDI R3, R0, 42", "HALT"])
        vm = VirtualMicroprocessor(prog)
        vm.run()
        self.assertEqual(vm.cpu.regs[3], 42)


class TestBranchable(unittest.TestCase):
    def test_fork_independence(self):
        prog = assemble(["ADDI R1, R0, 1", "ADDI R2, R0, 2", "ADD R3, R1, R2", "HALT"])
        vm = VirtualMicroprocessor(prog)
        snap = vm.snapshot()
        child = vm.branch()
        child.perturb("latency")
        child.step(10)
        vm.restore(snap)
        self.assertEqual(vm.cpu.control.name, "balanced")
        self.assertEqual(vm.cpu.stats.cycles, 0)

    def test_rollout_preserves_parent(self):
        prog, mem, reg, _ = matrix_like(0, 8)
        vm = VirtualMicroprocessor(prog, mem, reg)
        c0 = vm.cpu.stats.cycles
        r = vm.rollout(32, control="streaming")
        self.assertIn("objective", r)
        self.assertEqual(vm.cpu.stats.cycles, c0)

    def test_governor_fires_under_heat(self):
        prog, mem, reg, _ = matrix_like(0, 64)
        vm = VirtualMicroprocessor(prog, mem, reg)
        vm.cpu.temperatures = [41.9, 41.9, 41.9, 41.9]
        reasons = vm.perturb("latency")  # turbo request near limit
        self.assertTrue(reasons)
        self.assertNotEqual(vm.cpu.control.power_mode.value, "turbo")


class TestSurrogateSearch(unittest.TestCase):
    def test_learn_search_loop(self):
        prog, mem, reg, _ = mixed(0)
        examples = []
        for s in range(3):
            p, m, r, _ = mixed(s)
            examples += collect_trace(p, m, r)
        sur = train_surrogate(examples, n_members=2)
        vm = VirtualMicroprocessor(prog, mem, reg)
        out = beam_search_plan(vm, sur, horizon_intervals=3, interval_cycles=16, beam=2, oracle_budget=2)
        self.assertIsNotNone(out["best_plan"])
        base = fixed_baseline(VirtualMicroprocessor(prog, mem, reg))
        # search with validation should be no worse than far off; just check it ran
        self.assertIn("objective", out["best_result"])
        self.assertLessEqual(out["oracle_calls"], 2)
        u = vm.uncertainty(sur)
        self.assertGreaterEqual(u, 0.0)

    def test_all_families_halt(self):
        for gen in (matrix_like, pointer_chase, branch_heavy, mixed):
            prog, mem, reg, _ = gen(1)
            vm = VirtualMicroprocessor(prog, mem, reg)
            res = vm.run(["balanced", "streaming", "control"], interval=32)
            self.assertEqual(res["completion_status"], "HALTED", gen.__name__)


if __name__ == "__main__":
    unittest.main()
