"""VMArch test suite: conservation invariants, determinism, learning loop."""
import unittest

from vmarch import isa
from vmarch.branch import make_predictor
from vmarch.bench import conservation_suite
from vmarch.config import ACTION_LIBRARY, MicroarchConfig
from vmarch.isa import assemble, disassemble
from vmarch.search import beam_search_plan, expert_bundle, mcts_plan
from vmarch.workloads import FAMILIES, stream_like
from vmarch.virtual import VirtualMicroarchitecture


def make_vm(prog=None, mem=None, reg=None, seed=None, family=None, cfg=None):
    cfg = cfg or MicroarchConfig()
    if prog is None:
        prog, mem, reg, _ = FAMILIES[family](seed)
    vm = VirtualMicroarchitecture(prog, cfg=cfg, mem_init=mem, reg_init=reg)
    return vm, 0


def make3(prog, mem, reg):
    return make_vm(prog, mem, reg)


_CACHED_ENS = {}


def get_test_ensemble():
    """Shared cached ensemble so every test class is self-sufficient."""
    if "ens" not in _CACHED_ENS:
        from vmarch.surrogate import train_ensemble
        examples = []
        for s in range(3):
            p, m, r, _ = FAMILIES["phase"](s)
            vm, root = make_vm(p, m, r)
            examples += vm.collect_dataset(
                root, intervals=8,
                bundles=["balanced", "streaming", "control"], interval=16)
        _CACHED_ENS["ens"], _CACHED_ENS["info"] = train_ensemble(
            examples, seeds=(11, 22), epochs=6)
        _CACHED_ENS["examples"] = examples
    return _CACHED_ENS["ens"], _CACHED_ENS["info"], _CACHED_ENS["examples"]


class TestISA(unittest.TestCase):
    def test_roundtrip(self):
        prog = assemble(["ADD R1, R2, R3", "FENCE", "PREFETCH R4, 16",
                         "BEQ R1, R2, -2", "ECALL R5", "HALT"])
        for ins in prog:
            self.assertEqual(isa.Instruction.decode(ins.encode()), ins)
        text = disassemble(prog)
        self.assertEqual(len(text), 6)

    def test_imm_range_enforced(self):
        with self.assertRaises(ValueError):
            isa.Instruction(isa.Op.ADDI, 1, 2, 0, 2000)


class TestBranchPredictors(unittest.TestCase):
    def test_all_kinds(self):
        for kind in ("none", "bimodal", "gshare", "tournament", "tage_lite"):
            p = make_predictor(kind)
            for i in range(50):
                taken = (i % 3 == 0)
                pr = p.predict(0x100 + (i % 7))
                if kind != "none":
                    from vmarch.core import Core  # noqa (commit API varies)
                    if kind == "bimodal":
                        p.commit(0x100, taken, pr)
                    elif kind in ("gshare", "tournament"):
                        p.commit(0x100, taken, pr, p.history)
                    else:
                        p.commit(0x100, taken, pr, p.base[0x100 & p.mask] >= 2)
                else:
                    p.update(0x100, taken)
            self.assertGreaterEqual(p.predictions + getattr(p, "abstentions", 1), 1)


class TestCoreConservation(unittest.TestCase):
    def test_suite(self):
        res = conservation_suite(make3, seeds=[0, 1])
        self.assertTrue(res["ok"], res["failed"])

    def test_fence_orders_memory(self):
        prog = assemble(["ADDI R1, R0, 10", "STORE R0, R1, 50", "FENCE",
                         "LOAD R2, R0, 50", "HALT"])
        vm, root = make_vm(prog, {}, {})
        cid = vm.branch(wid=root)
        res = vm.run_world(cid, ["balanced"])
        self.assertEqual(res["completion_status"], "HALTED")
        self.assertEqual(vm.worlds[cid].sim.core.regs[2], 10)

    def test_snapshot_restore(self):
        p, m, r, _ = stream_like(0, 6)
        vm, root = make_vm(p, m, r)
        cid = vm.branch(wid=root)
        w = vm.worlds[cid]
        w.sim.step(20)
        snap = w.sim.snapshot()
        cyc = w.sim.core.cycle
        w.sim.step(20)
        self.assertGreater(w.sim.core.cycle, cyc)
        w.sim.restore(snap)
        self.assertEqual(w.sim.core.cycle, cyc)

    def test_cross_config_runs(self):
        p, m, r, _ = stream_like(1, 6)
        for name, cfg in MicroarchConfig.design_points().items():
            vm, root = make_vm(p, m, r, cfg=cfg)
            cid = vm.branch(wid=root)
            res = vm.run_world(cid, ["balanced"])
            self.assertEqual(res["completion_status"], "HALTED", name)


class TestLearningLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ens, cls.info, cls.examples = get_test_ensemble()

    def test_parity_finite(self):
        from vmarch.surrogate import evaluate_parity
        par = evaluate_parity(self.ens, self.examples)
        self.assertTrue(0 <= par["mae"] < 10**6)
        self.assertEqual(par["n"], len(self.examples))

    def test_save_load_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.ens.save(d)
            from vmarch.surrogate import SurrogateEnsemble
            ens2 = SurrogateEnsemble.load(d)
            x, b, c, _ = self.examples[0]
            self.assertAlmostEqual(self.ens.predict_cost(x, b),
                                   ens2.predict_cost(x, b), places=9)

    def test_beam_search(self):
        p, m, r, _ = FAMILIES["mixed"](0)
        vm, root = make_vm(p, m, r)
        out = beam_search_plan(vm, root, self.ens, horizon=2, interval=16,
                               beam=2, oracle_budget=2)
        self.assertIsNotNone(out["best"])
        self.assertLessEqual(out["oracle_calls"], 2)

    def test_mcts_search(self):
        p, m, r, _ = FAMILIES["mixed"](1)
        vm, root = make_vm(p, m, r)
        out = mcts_plan(vm, root, self.ens, horizon=2, interval=16,
                        iterations=6, oracle_budget=2)
        self.assertIsNotNone(out["best"])

    def test_expert_runs(self):
        from vmarch.search import run_expert
        p, m, r, _ = FAMILIES["branch"](0)
        vm, root = make_vm(p, m, r)
        res = run_expert(vm, root, interval=16)
        self.assertIn("objective", res)
        self.assertEqual(res["completion_status"], "HALTED")

    def test_agents_smoke(self):
        from vmarch import agents
        from vmarch.surrogate import evaluate_divergence, evaluate_parity
        p, m, r, _ = FAMILIES["stream"](0)
        vm, root = make_vm(p, m, r)
        wid = vm.branch(wid=root)
        vm.run_world(wid, ["balanced"] * 2, interval=16)
        f = agents.bottleneck_analyst(vm, wid)
        self.assertIn("bottleneck", f)
        par = evaluate_parity(self.ens, self.examples[:8])
        div = evaluate_divergence(self.ens, self.examples[0][0], ["balanced"] * 4)
        crit = agents.surrogate_critic(self.ens, par, div,
                                       {"rate": 0.1})
        self.assertIn("verdict", crit)
        rep = agents.synthesize_report("smoke", [f, crit])
        self.assertIn("Lab report", rep)


class TestKillerBenchmarkTiny(unittest.TestCase):
    def test_killer(self):
        from vmarch.search import killer_benchmark
        ens, _, _ = get_test_ensemble()

        def mk(seed):
            p, m, r, _ = FAMILIES["phase"](10000 + seed)
            return make_vm(p, m, r)
        kb = killer_benchmark(mk, ens, seeds=[0, 1], horizon=2, interval=16, budget=2)
        self.assertIn("delta_surrogate_beam", kb)
        self.assertEqual(len(kb["per_seed"]), 2)


class TestSkillGates(unittest.TestCase):
    """T4: skill gates fail constants by construction, pass oracles."""

    @staticmethod
    def _examples():
        import random
        rng = random.Random(7)
        return [([float(i)], "balanced", 10.0 + i + rng.uniform(-1, 1), {})
                for i in range(12)]

    def test_perfect_stub_passes(self):
        from vmarch.bench import maturity_gate, skill_scores

        class Perfect:
            def predict_cost(self, x, b):
                return 10.0 + x[0]
        ex = [(x, b, 10.0 + x[0], m) for x, b, _, m in self._examples()]
        skill = skill_scores(ex, Perfect())
        self.assertAlmostEqual(skill["r2"], 1.0)
        self.assertAlmostEqual(skill["skill_vs_mean"], 1.0)
        kb = {"delta_surrogate_beam": {"mean": 5.0, "ci95": 1.0}}
        mat = maturity_gate({}, {"rate": 0.0}, {"monotone": True}, kb, skill)
        self.assertTrue(mat["L1_exemplar"])
        self.assertTrue(mat["L2_interpolative"])

    def test_mean_stub_fails(self):
        from vmarch.bench import maturity_gate, skill_scores

        class Mean:
            def __init__(self, mu):
                self.mu = mu
            def predict_cost(self, x, b):
                return self.mu
        ex = self._examples()
        mu = sum(t[2] for t in ex) / len(ex)
        skill = skill_scores(ex, Mean(mu))
        self.assertAlmostEqual(skill["skill_vs_mean"], 0.0)
        self.assertLessEqual(skill["r2"], 0.0)
        kb = {"delta_surrogate_beam": {"mean": 5.0, "ci95": 1.0}}
        mat = maturity_gate({}, {"rate": 0.0}, {"monotone": True}, kb, skill)
        self.assertFalse(mat["L1_exemplar"])
        self.assertFalse(mat["L2_interpolative"])

    def test_no_skill_no_gates(self):
        from vmarch.bench import maturity_gate
        kb = {"delta_surrogate_beam": {"mean": 5.0, "ci95": 1.0}}
        mat = maturity_gate({}, {"rate": 0.0}, {"monotone": True}, kb, None)
        self.assertFalse(mat["L1_exemplar"])
        self.assertFalse(mat["L2_interpolative"])


class TestMimicExperiment(unittest.TestCase):
    """T5: renderer hallucinates phantom pcs OOD; hybrid exact by construction."""

    def test_mimic_falsified(self):
        from vmarch.mimic import run_mimic_experiment
        res = run_mimic_experiment(make3, train_family="stream",
                                   test_family="loop_branch",
                                   train_seeds=(0, 1), test_seeds=(100,),
                                   epochs=15)
        self.assertEqual(res["verdict"], "RENDERER FALSIFIED")
        # symbolic channel broken even where costs are memorized ...
        self.assertLess(res["in_family"]["top1_hit"], 0.10)
        # ... and the scalar channel degrades out-of-family ...
        self.assertGreaterEqual(
            res["ood"]["cost_mae"] / max(res["in_family"]["cost_mae"], 1e-9), 2.0)
        # ... while the hybrid's symbolic channel is exact by construction.
        self.assertEqual(res["hybrid_by_construction"]["phantom_mass"], 0.0)
        self.assertEqual(res["hybrid_by_construction"]["top1_hit"], 1.0)


if __name__ == "__main__":
    unittest.main()
