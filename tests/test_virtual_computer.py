"""Cross-layer integration tests v2: science, not plumbing.

Every test asserts a checkable property of the measurement chain
(equivalence on two CPU oracles, determinism, conservation, budget
exactness, honest grading, surrogate honesty on synthetic data).
Nothing here can pass for a random surrogate.
"""
import unittest

from virtual_computer import lowering
from virtual_computer.computer import VirtualComputer, layer_available
from virtual_computer.levels import certify_unified
from virtual_computer.vertical import run_vertical_slice
from virtual_computer.xsurrogate import loo_skill

SMALL = dict(seeds=(0, 1), ns=(4, 8), controls=("balanced",),
             kernel_steps=2, gains=(1.0,), ens_members=2,
             ens_episodes=2, ens_steps=4, ens_epochs=2)


class TestLayersPresent(unittest.TestCase):
    def test_all_layers_importable(self):
        for layer in ("compiler", "kernel", "microarchitecture"):
            ok, reason = layer_available(layer)
            self.assertTrue(ok, f"{layer} unavailable: {reason}")


class TestLowering(unittest.TestCase):
    def test_structural_gap_every_n(self):
        from vmicro.assembler import assemble
        for n in (4, 8, 12):
            h0 = lowering.op_histogram(assemble(lowering.lower("O0", n)))
            h3 = lowering.op_histogram(assemble(lowering.lower("O3", n)))
            self.assertLess(h3.get("mem", 0), h0.get("mem", 0))
            self.assertGreater(h3.get("vector", 0), 0)

    def test_vectors_deterministic(self):
        self.assertEqual(lowering.vector_memory(3, 8), lowering.vector_memory(3, 8))

    def test_unknown_preset_rejected(self):
        with self.assertRaises(ValueError):
            lowering.lower("Ofast", 8)

    def test_vmarch_translation_preserves_program(self):
        from vmicro.assembler import assemble
        prog = assemble(lowering.lower("O3", 4))
        arch = lowering.translate_to_vmarch(prog)
        self.assertEqual(len(arch), len(prog))
        self.assertEqual([i.op.name for i in arch],
                         [i.op.name for i in prog])


class TestVerticalScience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_vertical_slice(**SMALL)

    def test_all_gates_pass(self):
        self.assertEqual(self.report["gate_problems"], [])
        for gate, passed in self.report["gates"].items():
            self.assertTrue(passed, f"gate {gate} failed")

    def test_budget_vector_exact(self):
        S, Nn, Nc, H, G = 2, 2, 1, 2, 1
        self.assertEqual(self.report["budget"], {
            "compiler_stub_views": 2,
            "micro_cpu_runs": 2 * S * Nn * Nc + 4,
            "vmarch_runs": 2 * S * Nn,
            "kernel_sim_steps": S * 2 * Nn * G * 2 * H + 2 * 4})

    def test_stratified_stats_present(self):
        v = self.report
        self.assertIn("balanced", v["micro"]["median_by_control"])
        self.assertIn(4, v["micro"]["median_by_n"])
        for w in ("O0", "O3"):
            self.assertIn(w, v["kernel"]["stats"])
            self.assertIn(1.0, v["kernel"]["sensitivity_verdicts"][w])

    def test_paired_stats_sane(self):
        for st in [self.report["micro"]["stats"], self.report["vmarch"]["stats"],
                   *self.report["kernel"]["stats"].values()]:
            self.assertLessEqual(st["ci_low"], st["median"])
            self.assertLessEqual(st["median"], st["ci_high"])
            self.assertIn(st["verdict"], ("improves", "regresses", "within-noise"))

    def test_vmarch_cross_oracle_agreement_reported(self):
        rate = self.report["vmarch"]["sign_agreement_rate"]
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_end_to_end_deterministic(self):
        again = run_vertical_slice(**SMALL)
        self.assertEqual(again["digest"], self.report["digest"])

    def test_versions_bound(self):
        self.assertEqual(self.report["versions"]["lowering"], "lowering-v2")
        self.assertEqual(self.report["versions"]["coupling"], "coupling-v2")

    def test_ensemble_scored(self):
        ens = self.report["ensemble"]
        self.assertGreater(ens["n"], 0)
        self.assertGreaterEqual(ens["mse"], 0.0)

    def test_xsurrogate_reported(self):
        for tab in ("micro_loo", "kernel_loo"):
            self.assertIn("skill", self.report["xsurrogate"][tab])


class TestXSurrogateHonesty(unittest.TestCase):
    def test_learns_linear_structure(self):
        import random
        rng = random.Random(0)
        X = [[rng.uniform(-5, 5), rng.uniform(-5, 5)] for _ in range(30)]
        y = [3 * r[0] - 2 * r[1] for r in X]
        self.assertGreater(loo_skill(X, y)["skill"], 0.9)

    def test_reports_no_skill_on_noise(self):
        import random
        rng = random.Random(1)
        X = [[rng.uniform(-5, 5), rng.uniform(-5, 5)] for _ in range(30)]
        y = [rng.uniform(-5, 5) for _ in range(30)]
        self.assertLess(loo_skill(X, y)["skill"], 0.5)

    def test_constant_target_scores_zero(self):
        X = [[float(i), float(i)] for i in range(10)]
        out = loo_skill(X, [1.0] * 10)
        self.assertEqual(out["skill"], 0.0)
        self.assertEqual(out["reason"], "constant-target")

    def test_insufficient_units_degrades_gracefully(self):
        out = loo_skill([[1.0], [2.0]], [1.0, 2.0])
        self.assertEqual(out["skill"], 0.0)
        self.assertEqual(out["reason"], "insufficient-units")


class TestBudgetLedger(unittest.TestCase):
    def test_unknown_unit_rejected(self):
        with self.assertRaises(ValueError):
            VirtualComputer()._spend("gpu_hours", 1)

    def test_overspend_blocked_per_unit(self):
        vc = VirtualComputer({"compiler_stub_views": 0, "micro_cpu_runs": 1,
                              "vmarch_runs": 1, "kernel_sim_steps": 1})
        with self.assertRaises(RuntimeError):
            vc._spend("compiler_stub_views", 1)
        vc._spend("vmarch_runs", 1)
        with self.assertRaises(RuntimeError):
            vc._spend("vmarch_runs", 1)

    def test_uncertainty_unmeasured_not_invented(self):
        self.assertEqual(VirtualComputer().uncertainty()["status"], "unmeasured")

    def test_gcc_key_nests_under_vertical_only_when_requested(self):
        rep = VirtualComputer().run_vertical_slice(
            seeds=(0,), ns=(4,), controls=("balanced",), kernel_steps=2,
            gains=(1.0,), gcc=False)
        self.assertNotIn("gcc", rep)


class TestHonestGrade(unittest.TestCase):
    def test_grade_capped_by_weakest_layer(self):
        cert = certify_unified(None)
        for layer in ("microarchitecture", "compiler", "kernel"):
            self.assertLessEqual(cert["certified_level"],
                                 cert["layers"][layer]["grade"])

    def test_current_headline_is_L0_with_named_blocker(self):
        cert = certify_unified(None)
        self.assertEqual(cert["certified_level"], 0)
        self.assertIn("micro", " ".join(cert["blockers"]))

    def test_live_integration_does_not_inflate(self):
        rep = run_vertical_slice(**SMALL)
        cert = certify_unified(rep)
        lo = min(cert["layers"][n]["grade"]
                 for n in ("microarchitecture", "compiler", "kernel"))
        self.assertLessEqual(cert["certified_level"], lo)
        self.assertEqual(cert["layers"]["integration"]["grade"], 1)


class TestToolchainProbe(unittest.TestCase):
    def test_probe_returns_bool(self):
        from virtual_computer.gcc_leg import toolchain_available
        self.assertIsInstance(toolchain_available(), bool)


class TestEvidenceFreshness(unittest.TestCase):
    def test_kernel_freshness_structure(self):
        from virtual_computer.levels import kernel_evidence_freshness
        f = kernel_evidence_freshness()
        for key in ("fresh", "config_reproduces", "recorded_config_hash",
                    "reproduced_config_hash", "sources_predate_artifact",
                    "newest_source", "code_hash"):
            self.assertIn(key, f)
        self.assertIsInstance(f["fresh"], bool)
        self.assertRegex(f["code_hash"], r"^[0-9a-f]{12}$")

    def test_kernel_config_hash_reproduces(self):
        # the mechanism's own assumption, pinned: current code replays
        # the committed full-mode config hash exactly
        from virtual_computer.levels import kernel_evidence_freshness
        f = kernel_evidence_freshness()
        self.assertTrue(f["config_reproduces"])
        self.assertEqual(f["recorded_config_hash"], f["reproduced_config_hash"])

    def test_fresh_evidence_carries_no_stale_blocker(self):
        from virtual_computer.levels import grade_kernel
        g = grade_kernel()
        self.assertTrue(g["freshness"]["fresh"])
        self.assertNotIn("blocker", g)

    def test_staleness_tripwire_fires(self):
        # prove the mechanism bites: a source newer than the artifact
        # must flip fresh to False (uses a temp file; repo untouched)
        import os
        import tempfile
        import time
        from virtual_computer import levels
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x")
            tmp = f.name
        future = time.time() + 100000
        os.utime(tmp, (future, future))
        old = levels._KERNEL_GRADE_SOURCES
        levels._KERNEL_GRADE_SOURCES = old + (tmp,)
        try:
            f2 = levels.kernel_evidence_freshness()
            self.assertFalse(f2["fresh"])
            self.assertFalse(f2["sources_predate_artifact"])
        finally:
            levels._KERNEL_GRADE_SOURCES = old
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
