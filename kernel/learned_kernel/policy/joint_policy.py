"""
policy/joint_policy.py — Multi-subsystem joint heuristic policy.
"""
from typing import List, Optional
from .core import PolicyBase
from .schemas import (
    KernelIntermediateRepresentation,
    PolicyAction, SchedulerAction, VMAction, NetAction,
)


class JointHeuristicPolicy(PolicyBase):
    """
    Simultaneously controls Scheduler + VM + Network based on unified KIR state.

    Observation contracts
    ---------------------
    scheduler.cpus.*.utilization
    memory.pressure
    network.rx_queue_len

    Action contracts
    ----------------
    scheduler.target_latency_us
    vm.swappiness
    network.queue_limit
    """

    @property
    def policy_id(self) -> str:       return "joint-heuristic-v1"
    @property
    def version(self)   -> str:       return "1.0"

    @property
    def observation_schema(self) -> List[str]:
        return [
            "scheduler.cpus.*.utilization",
            "memory.pressure",
            "network.rx_queue_len",
        ]

    @property
    def action_schema(self) -> List[str]:
        return [
            "scheduler.target_latency_us",
            "vm.swappiness",
            "network.queue_limit",
        ]

    def decide(self, kstate: KernelIntermediateRepresentation) -> Optional[PolicyAction]:
        # Defaults: Linux-conservative starting point
        swappiness = 60
        queue_lim  = 1000
        lat_us     = 6000

        # SCENARIO A: Network I/O spike
        # High rx backlog → allocate socket buffers from RAM (lower swappiness)
        # and shorten scheduler tick so network_softirq threads get CPU quickly.
        if kstate.network and kstate.network.rx_queue_len > 1000:
            swappiness = 10
            queue_lim  = 4000   # well below the 5000 joint-safety ceiling
            lat_us     = 2000

        # SCENARIO B: Memory pressure dominates
        # System-survival priority: constrain network buffers, reclaim aggressively.
        # Note: this branch intentionally overwrites SCENARIO A values if both
        # conditions are true simultaneously, because OOM risk trumps rx latency.
        if kstate.memory and kstate.memory.pressure > 0.8:
            queue_lim  = 1000
            swappiness = 100
            lat_us     = 12000

        return PolicyAction(
            policy_id=self.policy_id,
            scheduler=SchedulerAction(target_latency_us=lat_us),
            vm=VMAction(swappiness=swappiness),
            network=NetAction(queue_limit=queue_lim),
        )
