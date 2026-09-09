import unittest

from learned_kernel.policy.schemas import (
    CPUMetrics,
    KernelIntermediateRepresentation,
    MemMetrics,
    NetMetrics,
    SchedulerState,
)


def kir(**kw):
    base = dict(
        timestamp=1.0,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=0.5, runnable_tasks=1, irq_time=0.1)},
            total_context_switches=10, avg_latency_ms=2.0),
    )
    base.update(kw)
    return KernelIntermediateRepresentation(**base)


class IngestionBoundaryTests(unittest.TestCase):
    """Bad telemetry must be rejected at the KIR boundary (data-quality defence)."""

    def test_utilization_below_zero_rejected(self):
        with self.assertRaises(Exception):
            kir(scheduler=SchedulerState(
                cpus={0: CPUMetrics(utilization=-0.5, runnable_tasks=1)},
                total_context_switches=1, avg_latency_ms=1.0))

    def test_pressure_above_one_rejected(self):
        with self.assertRaises(Exception):
            kir(memory=MemMetrics(pressure=999.0, reclaim_rate_mb_s=0.0,
                                  cache_hit_rate=0.5))

    def test_negative_counters_rejected(self):
        with self.assertRaises(Exception):
            kir(scheduler=SchedulerState(
                cpus={0: CPUMetrics(utilization=0.5, runnable_tasks=-3)},
                total_context_switches=-5, avg_latency_ms=1.0))

    def test_negative_timestamp_rejected(self):
        with self.assertRaises(Exception):
            kir(timestamp=-1.0)

    def test_valid_snapshot_accepted(self):
        k = kir(memory=MemMetrics(pressure=0.4, reclaim_rate_mb_s=2.0, cache_hit_rate=0.9),
                network=NetMetrics(rx_queue_len=100, tx_queue_len=50, tcp_retransmits=0))
        self.assertEqual(k.memory.pressure, 0.4)

    def test_policy_id_pattern_enforced(self):
        with self.assertRaises(Exception):
            PolicyAction(policy_id="bad id with spaces")


if __name__ == "__main__":
    from learned_kernel.policy.schemas import PolicyAction
    unittest.main()
