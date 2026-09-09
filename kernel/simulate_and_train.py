import time
from learned_kernel.observation.monitors import SchedulerMonitor
from learned_kernel.policy.schemas import SchedulerState, CPUMetrics, KernelIntermediateRepresentation
from learned_kernel.trainer.rl_trainer import OfflineTrainer
from learned_kernel.runtime.loop import LearnedKernelRuntime

def generate_synthetic_history(steps=100) -> list:
    history = []
    base_switches = 1000
    for i in range(steps):
        # Fake workload transitions: initially low util, later high util
        util = 0.2 if i < 50 else 0.95
        switches = base_switches + (i * 20) if util > 0.5 else base_switches + (i * 2)
        
        state = KernelIntermediateRepresentation(
            timestamp=time.time() + i,
            scheduler=SchedulerState(
                cpus={0: CPUMetrics(utilization=util, runnable_tasks=int(util*10), irq_time=0.0)},
                total_context_switches=switches,
                avg_latency_ms=2.0 + (util * 10)
            )
        )
        history.append(state)
    return history

if __name__ == "__main__":
    print("\n" + "="*50)
    print("=== PHASE A: KERNEL TELEMETRY & OFFLINE RL TRAINER ===")
    print("="*50)
    history = generate_synthetic_history()
    
    trainer = OfflineTrainer()
    # The learned artifact is yielded from offline historical training
    learned_policy = trainer.train([history])
    
    print("\n" + "="*50)
    print("=== PHASE B: KERNEL DEPLOYMENT & ACTUATION LOOP ===")
    print("="*50)
    runtime = LearnedKernelRuntime()
    
    print(f"[Deployment] Replacing baseline policy with: {learned_policy.policy_id} (v{learned_policy.version})")
    
    # Critical step: Re-assign the active policy safely via the boundary interface
    runtime.policy = learned_policy
    
    # Run the runtime loop
    for _ in range(2):
        runtime.step()
        time.sleep(0.1)
