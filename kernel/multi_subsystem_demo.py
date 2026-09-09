import time
from learned_kernel.policy.schemas import SchedulerState, CPUMetrics, MemMetrics, NetMetrics, KernelIntermediateRepresentation
from learned_kernel.policy.joint_policy import JointHeuristicPolicy
from learned_kernel.runtime.loop import LearnedKernelRuntime

def simulate_workload(workload_name: str) -> KernelIntermediateRepresentation:
    if workload_name == "web_server": # High Network Spikes, Low Mem Pressure
        return KernelIntermediateRepresentation(
            timestamp=time.time(),
            scheduler=SchedulerState(
                cpus={0: CPUMetrics(utilization=0.6, runnable_tasks=50, irq_time=0.1)},
                total_context_switches=10000, avg_latency_ms=1.5
            ),
            memory=MemMetrics(pressure=0.2, reclaim_rate_mb_s=0.5, cache_hit_rate=0.95),
            network=NetMetrics(rx_queue_len=4500, tx_queue_len=4000, tcp_retransmits=10)
        )
    elif workload_name == "database": # High Disk/Memory pressure, Low Network
        return KernelIntermediateRepresentation(
            timestamp=time.time(),
            scheduler=SchedulerState(
                cpus={0: CPUMetrics(utilization=0.9, runnable_tasks=5, irq_time=0.0)},
                total_context_switches=200, avg_latency_ms=8.0
            ),
            memory=MemMetrics(pressure=0.95, reclaim_rate_mb_s=250.0, cache_hit_rate=0.4),
            network=NetMetrics(rx_queue_len=100, tx_queue_len=150, tcp_retransmits=1)
        )
    elif workload_name == "malicious_action": # A fake state to trick the policy into bad parameters to test the verifier
        return KernelIntermediateRepresentation(
            timestamp=time.time(),
            scheduler=SchedulerState(
                cpus={0: CPUMetrics(utilization=1.0, runnable_tasks=500, irq_time=0.5)},
                total_context_switches=50000, avg_latency_ms=15.0
            ),
            memory=MemMetrics(pressure=0.99, reclaim_rate_mb_s=800.0, cache_hit_rate=0.01),
            network=NetMetrics(rx_queue_len=9000, tx_queue_len=9000, tcp_retransmits=500)
        )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("=== MULTI-SUBSYSTEM JOINT CONTROL DEMO ===")
    print("="*70)
    
    runtime = LearnedKernelRuntime()
    runtime.policy = JointHeuristicPolicy()
    
    for w_name in ["web_server", "database", "malicious_action"]:
        print(f"\n[Injecting Workload: {w_name.upper()}]")
        state = simulate_workload(w_name)
        
        # Policy Decides Actions
        action = runtime.policy.decide(state)
        
        # Validator Checks Joint Constraints
        try:
            runtime.validator.validate_action(state, action)
            runtime.actuator.apply(action)
            print(">>> VALIDATOR: CROSS-DOMAIN APPROVED.")
        except Exception as e:
            print(f">>> VALIDATOR INTERVENTION: {e}")
