import unittest

from learned_kernel.policy.schemas import (
    CPUMetrics,
    KernelIntermediateRepresentation,
    MemMetrics,
    NetAction,
    PolicyAction,
    SchedulerAction,
    SchedulerState,
    VMAction,
)
from learned_kernel.trainer.reward import RewardCalculator
from learned_kernel.validator.core import PolicyValidator, ValidatorError


def state(memory=None, switches=100, timestamp=1.0):
    return KernelIntermediateRepresentation(
        timestamp=timestamp,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=0.5, runnable_tasks=1, irq_time=0.0)},
            total_context_switches=switches,
            avg_latency_ms=2.0,
        ),
        memory=memory,
    )


class SafetyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.validator = PolicyValidator()

    def test_regular_scheduler_action_no_longer_crashes(self):
        action = PolicyAction(policy_id="test", scheduler=SchedulerAction(target_latency_us=6000))
        self.assertTrue(self.validator.validate_action(state(), action))

    def test_maximum_network_queue_requires_memory_telemetry(self):
        action = PolicyAction(policy_id="test", network=NetAction(queue_limit=5000))
        with self.assertRaises(ValidatorError):
            self.validator.validate_action(state(), action)
        self.assertTrue(self.validator.validate_action(state(MemMetrics(pressure=0.2, reclaim_rate_mb_s=1, cache_hit_rate=0.9)), action))

    def test_inconsistent_vm_watermarks_rejected(self):
        action = PolicyAction(policy_id="test", vm=VMAction(dirty_ratio=20, dirty_background_ratio=30))
        with self.assertRaises(ValidatorError):
            self.validator.validate_action(state(), action)

    def test_counter_reset_cannot_create_negative_throughput(self):
        reward = RewardCalculator(alpha=0, beta=1, gamma=0).calculate_step_reward(
            state(switches=100, timestamp=1.0), state(switches=1, timestamp=2.0)
        )
        self.assertEqual(reward, 0)


if __name__ == "__main__":
    unittest.main()
