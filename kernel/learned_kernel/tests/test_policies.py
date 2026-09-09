import unittest

from learned_kernel.policy.core import HeuristicLatencyPolicy, LinuxDefaultPolicy
from learned_kernel.policy.joint_policy import JointHeuristicPolicy
from learned_kernel.policy.schemas import (
    CPUMetrics,
    KernelIntermediateRepresentation,
    MemMetrics,
    NetMetrics,
    SchedulerState,
)


def kir(util=0.5, pressure=None, rx=None):
    return KernelIntermediateRepresentation(
        timestamp=100.0,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=util, runnable_tasks=2, irq_time=0.01)},
            total_context_switches=10, avg_latency_ms=2.0),
        memory=MemMetrics(pressure=pressure, reclaim_rate_mb_s=1, cache_hit_rate=0.9)
        if pressure is not None else None,
        network=NetMetrics(rx_queue_len=rx or 0, tx_queue_len=0, tcp_retransmits=0)
        if rx is not None else None,
    )


class AbiContractTests(unittest.TestCase):
    def test_concrete_policies_declare_schemas(self):
        for pol in (HeuristicLatencyPolicy(), LinuxDefaultPolicy(),
                    JointHeuristicPolicy()):
            self.assertTrue(pol.policy_id)
            self.assertTrue(pol.version)

    def test_decide_returns_none_or_policy_action(self):
        from learned_kernel.policy.schemas import PolicyAction
        for pol in (HeuristicLatencyPolicy(), LinuxDefaultPolicy(),
                    JointHeuristicPolicy()):
            out = pol.decide(kir())
            self.assertTrue(out is None or isinstance(out, PolicyAction))


class HeuristicHysteresisTests(unittest.TestCase):
    def setUp(self):
        self.p = HeuristicLatencyPolicy()

    def test_engages_above_enter_threshold(self):
        first = self.p.decide(kir(util=0.5))
        self.assertEqual(first.scheduler.target_latency_us, 12000)   # loose baseline
        a = self.p.decide(kir(util=0.85))
        self.assertEqual(a.scheduler.target_latency_us, 2000)

    def test_deadband_prevents_flapping(self):
        self.p.decide(kir(util=0.5))
        self.p.decide(kir(util=0.85))                        # → tight
        # Hovering inside the deadband must NOT flip back to loose:
        # either None (unchanged) or an explicit tight action.
        for u in (0.79, 0.75, 0.71):
            a = self.p.decide(kir(util=u))
            if a is not None:
                self.assertEqual(a.scheduler.target_latency_us, 2000, f"flapped at {u}")
        a = self.p.decide(kir(util=0.65))
        self.assertIsNotNone(a)
        self.assertEqual(a.scheduler.target_latency_us, 12000)

    def test_identical_state_suppresses_actuator_churn(self):
        s = kir(util=0.5)
        first = self.p.decide(s)
        self.assertIsNotNone(first)
        second = self.p.decide(s)                            # same params again
        self.assertIsNone(second)

    def test_reset_restores_initial_behaviour(self):
        s_hi = kir(util=0.9)
        self.p.decide(s_hi)                                  # → tight
        self.p.reset()
        a = self.p.decide(s_hi)
        self.assertEqual(a.scheduler.target_latency_us, 2000)


class JointPolicyTests(unittest.TestCase):
    def test_network_spike_scenario(self):
        p = JointHeuristicPolicy()
        a = p.decide(kir(rx=4500))
        self.assertEqual(a.vm.swappiness, 10)
        self.assertEqual(a.network.queue_limit, 4000)

    def test_memory_pressure_overrides_network_scenario(self):
        p = JointHeuristicPolicy()
        a = p.decide(kir(rx=4500, pressure=0.95))
        self.assertEqual(a.network.queue_limit, 1000)
        self.assertEqual(a.vm.swappiness, 100)

    def test_all_emitted_values_pass_validator(self):
        from learned_kernel.validator.core import PolicyValidator
        v = PolicyValidator()
        for kwargs in ({}, {"rx": 4500}, {"pressure": 0.95},
                       {"rx": 4500, "pressure": 0.95}):
            a = JointHeuristicPolicy().decide(kir(**kwargs))
            self.assertTrue(v.validate_action(kir(**kwargs), a))


if __name__ == "__main__":
    unittest.main()
