import json
import tempfile
import unittest
from pathlib import Path

from learned_kernel.policy.schemas import CPUMetrics, SchedulerState
from learned_kernel.policy.schemas import KernelIntermediateRepresentation
from learned_kernel.trainer.rl_trainer import (
    LearnedLinearPolicy,
    OfflineTrainer,
    extract_features,
)
from learned_kernel.validator.core import PolicyValidator


def kir(util, runnable=6, latency_ms=8.0, switches=0):
    return KernelIntermediateRepresentation(
        timestamp=100.0,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=util, runnable_tasks=runnable, irq_time=0.01)},
            total_context_switches=switches,
            avg_latency_ms=latency_ms),
    )


def synthetic_history(steps=60):
    """Regime-shift telemetry like the demo generators."""
    out = []
    for i in range(steps):
        util = 0.2 if i < 30 else 0.9
        out.append(kir(util, runnable=int(util * 10), latency_ms=2 + util * 12))
        out[-1] = out[-1].model_copy(update={"timestamp": 100.0 + i})
    return out


class TrainerCausalityTests(unittest.TestCase):
    def test_training_improves_holdout_reward(self):
        """THE core regression: gradients must actually flow (simulator rollouts)."""
        tr = OfflineTrainer(rounds=6, seed=11)
        pol = tr.train([synthetic_history()], verbose=False)
        self.assertTrue(tr.last_report.improved,
                        f"no holdout improvement: {tr.last_report.summary()}")

    def test_deterministic_across_runs(self):
        h = [synthetic_history()]
        p1 = OfflineTrainer(seed=42).train(h, verbose=False)
        p2 = OfflineTrainer(seed=42).train(h, verbose=False)
        self.assertEqual(p1.weights, p2.weights)
        self.assertEqual(p1.bias, p2.bias)

    def test_learned_actions_respect_validator(self):
        pol = OfflineTrainer(rounds=3, seed=5).train([synthetic_history()], verbose=False)
        v = PolicyValidator()
        for u in (0.1, 0.4, 0.7, 0.95):
            a = pol.decide(kir(u))
            self.assertTrue(v.validate_action(kir(u), a))

    def test_features_are_six_and_normalised(self):
        f = extract_features(kir(0.9))
        self.assertEqual(len(f), 6)
        for x in f:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.05)


class ArtifactTests(unittest.TestCase):
    def test_artifact_roundtrip(self):
        pol = LearnedLinearPolicy(weights=[0.1, -0.2, 0.3, 0, 0, 0], bias=7000)
        art = pol.to_artifact(dataset_sha256="abc")
        loaded = LearnedLinearPolicy.from_artifact(art)
        self.assertEqual(loaded.weights, pol.weights)
        self.assertEqual(loaded.bias, pol.bias)

    def test_tampered_artifact_rejected(self):
        pol = LearnedLinearPolicy(weights=[0, 0, 0, 0, 0, 0], bias=6000)
        art = pol.to_artifact()
        art["payload"]["bias"] = 99999.0          # tamper after signing
        with self.assertRaises(ValueError):
            LearnedLinearPolicy.from_artifact(art)

    def test_save_load_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "policy.json"
            tr = OfflineTrainer(rounds=2, seed=1)
            pol = tr.train([synthetic_history(30)], verbose=False)
            tr.save_artifact(path, pol)
            back = OfflineTrainer.load_artifact(path)
            self.assertEqual(back.weights, pol.weights)
            blob = json.loads(path.read_text())
            self.assertIn("sha256", blob)
            self.assertIn("dataset_sha256", blob["payload"])


if __name__ == "__main__":
    unittest.main()
