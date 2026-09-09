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
from learned_kernel.validator.core import ALLOWED_CONGESTION_CONTROL, PolicyValidator, ValidatorError


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
        from learned_kernel.trainer.reward import RewardCalculator
        reward = RewardCalculator(alpha=0, beta=1, gamma=0).calculate_step_reward(
            state(switches=100, timestamp=1.0), state(switches=1, timestamp=2.0)
        )
        self.assertEqual(reward, 0)


class ValidatorBoundTests(unittest.TestCase):
    def setUp(self):
        self.v = PolicyValidator()

    # ---- scheduler ----
    def test_target_latency_below_floor_rejected(self):
        a = PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=999))
        with self.assertRaises(ValidatorError):
            self.v.validate_action(state(), a)

    def test_target_latency_above_ceiling_rejected(self):
        a = PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=24001))
        with self.assertRaises(ValidatorError):
            self.v.validate_action(state(), a)

    def test_zero_wakeup_granularity_rejected_even_without_target(self):
        a = PolicyAction(policy_id="t", scheduler=SchedulerAction(wakeup_granularity_us=0))
        with self.assertRaises(ValidatorError):
            self.v.validate_action(state(), a)

    def test_negative_migration_cost_rejected(self):
        a = PolicyAction(policy_id="t", scheduler=SchedulerAction(migration_cost_ns=-1))
        with self.assertRaises(ValidatorError):
            self.v.validate_action(state(), a)

    def test_oversized_migration_cost_rejected(self):
        a = PolicyAction(policy_id="t", scheduler=SchedulerAction(migration_cost_ns=10**9))
        with self.assertRaises(ValidatorError):
            self.v.validate_action(state(), a)

    # ---- injection defence ----
    def test_unknown_congestion_control_rejected(self):
        a = PolicyAction(policy_id="t", network=NetAction(congestion_control="bogus_algo"))
        with self.assertRaises(ValidatorError):
            self.v.validate_action(state(), a)

    def test_sysctl_injection_string_rejected(self):
        payload = "cubic\nvm.swappiness=1"
        a = PolicyAction(policy_id="t", network=NetAction(congestion_control=payload))
        with self.assertRaises(ValidatorError):
            self.v.validate_action(state(), a)

    def test_allowlisted_congestion_control_accepted(self):
        for cc in ALLOWED_CONGESTION_CONTROL:
            a = PolicyAction(policy_id="t", network=NetAction(congestion_control=cc))
            self.assertTrue(self.v.validate_action(state(), a), cc)

    # ---- vm ----
    def test_swappiness_out_of_bounds_rejected(self):
        for bad in (-1, 101):
            a = PolicyAction(policy_id="t", vm=VMAction(swappiness=bad))
            with self.assertRaises(ValidatorError):
                self.v.validate_action(state(), a)

    def test_valid_full_action_accepted(self):
        a = PolicyAction(
            policy_id="t",
            scheduler=SchedulerAction(target_latency_us=6000, wakeup_granularity_us=1500,
                                      migration_cost_ns=250000),
            vm=VMAction(swappiness=60, dirty_ratio=30, dirty_background_ratio=10),
            network=NetAction(queue_limit=1000, congestion_control="cubic"),
        )
        self.assertTrue(self.v.validate_action(state(), a))


if __name__ == "__main__":
    unittest.main()
