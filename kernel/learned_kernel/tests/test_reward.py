import unittest

from learned_kernel.policy.schemas import CPUMetrics, SchedulerState
from learned_kernel.policy.schemas import KernelIntermediateRepresentation, MemMetrics, NetMetrics
from learned_kernel.trainer.reward import RewardCalculator


def kir(ts=1.0, util=0.5, runnable=4, switches=10_000, latency_ms=6.0,
        pressure=None, rx=None):
    return KernelIntermediateRepresentation(
        timestamp=ts,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=util, runnable_tasks=runnable, irq_time=0.01)},
            total_context_switches=switches,
            avg_latency_ms=latency_ms,
        ),
        memory=MemMetrics(pressure=pressure, reclaim_rate_mb_s=1.0, cache_hit_rate=0.9)
        if pressure is not None else None,
        network=NetMetrics(rx_queue_len=rx, tx_queue_len=0, tcp_retransmits=0)
        if rx is not None else None,
    )


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.rc = RewardCalculator()

    def test_higher_latency_lowers_reward(self):
        base = self.rc.calculate_step_reward(kir(switches=0), kir(ts=2.0, latency_ms=2.0))
        worse = self.rc.calculate_step_reward(kir(switches=0), kir(ts=2.0, latency_ms=18.0))
        self.assertGreater(base, worse)


    def test_thrash_penalty_engages_only_above_tau(self):
        # 5k switches/s → T̄ = 0.5 < τ ⇒ no thrash penalty.
        mild = self.rc.explain_step_reward(kir(ts=1.0, switches=0), kir(ts=2.0, switches=5000))["thrash"]
        self.assertEqual(mild, 0.0)
        # 10k switches/s → T̄ = 1.0 ⇒ full thrash penalty.
        storm = self.rc.explain_step_reward(kir(ts=1.0, switches=0), kir(ts=2.0, switches=10_000))["thrash"]
        self.assertGreater(storm, 0.99)

    def test_thrash_penalty_can_dominate_throughput_gain(self):
        """Perverse-incentive regression: a preemption storm must not out-reward healthy load."""
        calm = self.rc.calculate_step_reward(kir(ts=1.0, switches=0), kir(ts=2.0, switches=6000))
        storm = self.rc.calculate_step_reward(kir(ts=1.0, switches=0), kir(ts=2.0, switches=10_000))
        self.assertGreater(calm, storm)

    def test_explain_matches_calculate(self):
        prev, curr = kir(switches=100), kir(ts=2.0, switches=8000, latency_ms=3.3, util=0.7)
        self.assertAlmostEqual(
            self.rc.calculate_step_reward(prev, curr),
            self.rc.explain_step_reward(prev, curr)["reward"], places=12)

    def test_counter_reset_yields_zero_delta(self):
        r = RewardCalculator(alpha=0, beta=1, gamma=0).calculate_step_reward(
            kir(switches=999), kir(ts=2.0, switches=1))
        self.assertEqual(r, 0.0)

    def test_invalid_tau_rejected(self):
        with self.assertRaises(ValueError):
            RewardCalculator(thrash_tau=1.0)


if __name__ == "__main__":
    unittest.main()
