import unittest

from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
from learned_kernel.simulator.env import KernelSimulator, WorkloadProfile
import random


class SimulatorTests(unittest.TestCase):
    def test_deterministic_given_seed(self):
        a = KernelSimulator(seed=7)
        b = KernelSimulator(seed=7)
        for _ in range(20):
            act = PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=4000))
            ka, kb = a.step(act), b.step(act)
            self.assertEqual(ka.model_dump_json(), kb.model_dump_json())

    def test_shorter_target_reduces_latency_at_moderate_load(self):
        lo = KernelSimulator(seed=3)._effective_latency(0.6, 12_000)
        hi = KernelSimulator(seed=3)._effective_latency(0.6, 2_000)
        self.assertLess(hi, lo)

    def test_extreme_target_overhead_creates_interior_optimum(self):
        """Latency must be non-monotonic: absurdly tiny T raises overhead again."""
        lat = lambda t: KernelSimulator(seed=1)._effective_latency(0.9, t)
        self.assertGreater(lat(1000), lat(3000))

    def test_switch_rate_grows_as_target_shrinks(self):
        rate = lambda t: KernelSimulator(seed=1)._switch_rate(0.8, t)
        self.assertGreater(rate(2000), rate(12000))

    def test_reset_from_state_preserves_counters(self):
        env = KernelSimulator(seed=5)
        for _ in range(5):
            env.step(None)
        snap = env.current_kir()
        other = KernelSimulator(seed=99)
        other.reset(from_state=snap)
        self.assertEqual(other.current_kir().scheduler.total_context_switches,
                         snap.scheduler.total_context_switches)

    def test_workload_stays_in_bounds(self):
        w = WorkloadProfile(rng=random.Random(0))
        for _ in range(500):
            d = w.advance()
            self.assertGreaterEqual(d, 0.05)
            self.assertLessEqual(d, 1.0)


if __name__ == "__main__":
    unittest.main()
