import time
from learned_kernel.policy.schemas import SchedulerState, CPUMetrics, KernelIntermediateRepresentation
from learned_kernel.runtime.loop import LearnedKernelRuntime
from learned_kernel.runtime.adaptation import WorkloadDriftManager
from learned_kernel.telemetry.provenance import ProvenanceLogger

def get_simulated_state(step: int) -> KernelIntermediateRepresentation:
    # Simulates workload phase shift abruptly at step 10
    is_cpu_bound = step >= 8
    
    util = 0.92 if is_cpu_bound else 0.15
    switches = 250 if is_cpu_bound else 80
    latency_val = 14.0 if is_cpu_bound else 1.5
    
    return KernelIntermediateRepresentation(
        timestamp=time.time() + (step * 0.1),
        scheduler=SchedulerState(
            cpus={0: CPUMetrics(utilization=util, runnable_tasks=int(util*20), irq_time=0.01)},
            total_context_switches=switches * step, # Cumulative metric
            avg_latency_ms=latency_val
        )
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("=== SELF-KERNEL ONLINE ADAPTATION & PROVENANCE DEMO ===")
    print("="*70 + "\n")
    
    runtime = LearnedKernelRuntime()
    # 3 cycle window to detect abrupt massive reward degradation
    drift_manager = WorkloadDriftManager(window_size=3, t_threshold=0.5)
    logger = ProvenanceLogger("demo_audit.jsonl")
    
    history_buffer = []
    
    for step in range(16):
        print(f"\n[Cycle {step+1}/16] Workload State: {'CPU-BOUND' if step >= 8 else 'IDLE-BOUND'}")
        
        curr_state = get_simulated_state(step)
        history_buffer.append(curr_state)
        
        # 1. Evaluate Reward & Detect Drift
        reward = 0.0
        if step > 0:
            prev_state = history_buffer[-2]
            reward = drift_manager.add_step(prev_state, curr_state)
            print(f"  -> Evaluated System Reward: {reward:.2f}")
            
            if drift_manager.check_drift():
                print(f"  >>> DRIFT ALARM! Model Reward fell below thresholds.")
                # Force synchronous training hook
                new_policy = drift_manager.trigger_adaptation([history_buffer[-8:]])
                runtime.policy = new_policy
                print(f"  >>> HOT-SWAP COMPLETE! Adopted: {new_policy.policy_id} v{new_policy.version}")
        
        # 2. Policy Decides bounded actions
        action = runtime.policy.decide(curr_state)
        
        # 3. Validator Ensures Kernel Survival
        is_valid = False
        val_error = ""
        if action:
            try:
                is_valid = runtime.validator.validate_action(curr_state, action)
                if is_valid:
                    runtime.actuator.apply(action)
            except Exception as e:
                val_error = str(e)
                print(f"  -> Validator BLOCKED action: {e}")
                
        # 4. Provenance Cryptographic Write
        state_hash = logger.record_decision(
            kstate=curr_state,
            action=action,
            policy_id=runtime.policy.policy_id,
            policy_version=runtime.policy.version,
            is_validated=is_valid,
            validator_error=val_error,
            reward=reward
        )
        print(f"  -> Provenance Audit Logged | Hash: {state_hash}")
        
        time.sleep(0.05)
        
    print("\n[Simulation Complete] Audit trail persisted to 'demo_audit.jsonl'")
