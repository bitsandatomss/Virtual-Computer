import random
import unittest

from learned_kernel.policy.schemas import CPUMetrics, SchedulerState
from learned_kernel.policy.schemas import KernelIntermediateRepresentation
from learned_kernel.runtime.adaptation import WorkloadDriftManager, welch_t_statistic


def kir(ts, switches, latency_ms):
    return KernelIntermediateRepresentation(
        timestamp=ts,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=0.5, runnable_tasks=2, irq_time=0.01)},
            total_context_switches=switches,
            avg_latency_ms=latency_ms),
    )


class LatencyFeeder:
    """Stateful KIR stream generator; push(lat_ms) feeds exactly one step."""

    def __init__(self, jitter_rng=None):
        self.ts = 1000.0
        self.sw = 100_000
        self.prev = kir(self.ts, self.sw, 2.0)
        self.rng = jitter_rng

    def _jitter(self):
        return self.rng.gauss(0, 0.3) if self.rng else 0.0

    def push(self, mgr: WorkloadDriftManager, lat_ms: float) -> float:
        self.ts += 1.0
        self.sw += 3000
        curr = kir(self.ts, self.sw, max(0.05, lat_ms + self._jitter()))
        r = mgr.add_step(self.prev, curr)
        self.prev = curr
        return r


class WelchTests(unittest.TestCase):
    def test_sign_and_magnitude(self):
        self.assertLess(welch_t_statistic([1.0, 1.1, 0.9], [5.0, 5.1, 4.9]), 0)
        self.assertGreater(welch_t_statistic([9.0, 8.8, 9.2], [1.0, 1.2, 0.8]), 0)

    def test_equal_means_zero(self):
        self.assertEqual(welch_t_statistic([3.0, 3.2], [3.1, 3.1]), 0.0)

    def test_undersized_windows_return_zero(self):
        self.assertEqual(welch_t_statistic([1.0], [2.0]), 0.0)


class DriftDetectionTests(unittest.TestCase):
    def _mgr(self, ws=6, tau=2.5, eff=0.08):
        return WorkloadDriftManager(window_size=ws, t_threshold=tau,
                                    min_effect_size=eff, verbose=False)

    def test_quiet_on_stationary_noise(self):
        rng = random.Random(7)
        mgr = self._mgr()
        feeder = LatencyFeeder(rng)
        fires = 0
        for _ in range(400):
            feeder.push(mgr, rng.gauss(6.0, 0.8))
            if mgr.check_drift(consume=True):
                fires += 1
        self.assertLessEqual(fires, 3, f"false-positive rate too high: {fires}/400")

    def test_fires_on_regime_drop_quickly(self):
        rng = random.Random(3)
        mgr = self._mgr(ws=5, tau=2.0, eff=0.15)
        feeder = LatencyFeeder(rng)
        fired_at = None
        step = 0
        while step < 120 and fired_at is None:
            level = 2.0 if step < 40 else 16.0          # healthy → degraded
            feeder.push(mgr, level)
            step += 1
            if mgr.check_drift(consume=True):
                fired_at = step
        self.assertIsNotNone(fired_at, "drift never detected")
        self.assertLess(fired_at, 60, f"detection too slow ({fired_at})")

    def test_consume_prevents_refire(self):
        rng = random.Random(11)
        mgr = self._mgr(ws=4, tau=2.0, eff=0.1)
        feeder = LatencyFeeder(rng)
        for lat in [2.0] * 4 + [16.0] * 4:
            feeder.push(mgr, lat)
        first = mgr.check_drift(consume=True)
        self.assertTrue(first)
        self.assertFalse(mgr.check_drift(), "re-fired on cleared windows")

    def test_no_effect_size_blocks_alarm(self):
        mgr = self._mgr(ws=5, tau=0.1, eff=99.0)       # impossible effect size
        feeder = LatencyFeeder()
        for _ in range(10):
            feeder.push(mgr, 16.0)
        self.assertFalse(mgr.check_drift())

    def test_high_variance_stable_system_does_not_fire(self):
        rng = random.Random(19)
        mgr = self._mgr(ws=6, tau=2.5, eff=0.08)
        feeder = LatencyFeeder(rng)
        fires = 0
        for _ in range(300):
            feeder.push(mgr, rng.uniform(1.0, 12.0))
            if mgr.check_drift(consume=True):
                fires += 1
        self.assertLessEqual(fires, 3, f"high-variance false alarms: {fires}/300")


if __name__ == "__main__":
    unittest.main()
