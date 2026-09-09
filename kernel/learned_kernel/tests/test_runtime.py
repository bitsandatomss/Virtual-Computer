import unittest

from learned_kernel.observation.monitors import SimulationMonitor
from learned_kernel.policy.core import HeuristicLatencyPolicy, LinuxDefaultPolicy
from learned_kernel.policy.schemas import CPUMetrics, NetAction, PolicyAction, SchedulerAction, SchedulerState
from learned_kernel.policy.schemas import KernelIntermediateRepresentation
from learned_kernel.runtime.loop import KernelActuator, LearnedKernelRuntime
from learned_kernel.validator.core import PolicyValidator


def kir(util=0.9):
    return KernelIntermediateRepresentation(
        timestamp=1.0,
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=util, runnable_tasks=3, irq_time=0.01)},
            total_context_switches=100,
            avg_latency_ms=5.0),
    )


class ActuatorGateTests(unittest.TestCase):
    def test_unapproved_action_never_reaches_actuator(self):
        a = KernelActuator(execute_real=False)
        act = PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=6000))
        self.assertEqual(a.apply(act, approved=False), [])

    def test_none_action_is_noop(self):
        self.assertEqual(KernelActuator().apply(None, approved=True), [])

    def test_plan_renders_expected_sysctls(self):
        a = KernelActuator()
        cmds = a.plan(PolicyAction(
            policy_id="t",
            scheduler=SchedulerAction(target_latency_us=2000, wakeup_granularity_us=500),
            vm=None, network=NetAction(queue_limit=3000)))
        self.assertIn("sysctl kernel.sched_latency_ns=2000000", cmds)
        self.assertIn("sysctl kernel.sched_wakeup_granularity_ns=500000", cmds)
        self.assertIn("sysctl net.core.netdev_max_backlog=3000", cmds)

    def test_malformed_value_refused(self):
        a = KernelActuator()
        with self.assertRaises(Exception):
            a.plan(PolicyAction(policy_id="t",
                                network=NetAction(congestion_control="cubic; rm -rf /")))


class RuntimeLoopTests(unittest.TestCase):
    def test_step_returns_structured_result_in_sim_mode(self):
        rt = LearnedKernelRuntime(policy=HeuristicLatencyPolicy(),
                                  monitor=SimulationMonitor(seed=42))
        res = rt.step(verbose=False)
        self.assertIsNotNone(res.kir)
        self.assertIsNotNone(res.action)
        self.assertTrue(res.approved)

    def test_baseline_policy_yields_no_change(self):
        rt = LearnedKernelRuntime(policy=LinuxDefaultPolicy(),
                                  monitor=SimulationMonitor(seed=1))
        res = rt.step(verbose=False)
        self.assertIsNone(res.action)
        self.assertTrue(res.approved)

    def test_policy_exception_contained(self):
        class Boom:
            policy_id = "boom"
            version = "0"
            observation_schema = []

            @property
            def action_schema(self):
                return []

            def decide(self, _):
                raise ValueError("exploded")

        rt = LearnedKernelRuntime(policy=Boom(), monitor=SimulationMonitor(seed=2))
        res = rt.step(verbose=False)
        self.assertIn("exploded", res.error or "")

    def test_simulation_monitor_forms_closed_loop(self):
        """Actuated target latency must influence subsequent simulated state."""
        m1 = SimulationMonitor(seed=9)
        m1.get_kir_state()
        m1.submit_action(PolicyAction(policy_id="t",
                                      scheduler=SchedulerAction(target_latency_us=12000)))
        after_long = m1.get_kir_state()

        m2 = SimulationMonitor(seed=9)
        m2.get_kir_state()
        m2.submit_action(PolicyAction(policy_id="t",
                                      scheduler=SchedulerAction(target_latency_us=1500)))
        after_tight = m2.get_kir_state()

        self.assertNotEqual(after_long.scheduler.avg_latency_ms,
                            after_tight.scheduler.avg_latency_ms)
        self.assertLess(after_tight.scheduler.avg_latency_ms,
                        after_long.scheduler.avg_latency_ms)


class ValidatorIntegrationTests(unittest.TestCase):
    def test_runtime_blocks_rogue_policy_end_to_end(self):
        """A compromised/buggy policy proposing out-of-band values must be
        blocked by the runtime pipeline (validator + actuator gate)."""
        class Rogue:
            policy_id = "rogue"
            version = "0"
            observation_schema = []
            action_schema = ["scheduler.target_latency_us"]

            def decide(self, _):
                return PolicyAction(policy_id=self.policy_id,
                                    scheduler=SchedulerAction(target_latency_us=999_999))

        rt = LearnedKernelRuntime(policy=Rogue(), monitor=SimulationMonitor(seed=4))
        res = rt.step(verbose=False)
        self.assertIsNotNone(res.action)
        self.assertFalse(res.approved)
        self.assertEqual(res.applied, [])
        self.assertIn("Safety Violation", res.error or "")

    def test_learned_policy_output_always_in_band(self):
        from learned_kernel.trainer.rl_trainer import LearnedLinearPolicy
        pol = LearnedLinearPolicy(weights=[-50.0, 0, 0, 0, 0, 0], bias=-40000.0)
        v = PolicyValidator()
        for u in (0.05, 0.5, 1.0):
            a = pol.decide(kir(u))
            self.assertTrue(v.validate_action(kir(u), a))


if __name__ == "__main__":
    unittest.main()
