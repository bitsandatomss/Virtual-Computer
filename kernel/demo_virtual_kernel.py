"""demo_virtual_kernel.py — end-to-end Virtual Kernel demonstration.

Pipeline (context.txt: learn -> branch -> search -> act -> validate):
  1. Train ensemble surrogate from exact-oracle (s, a, s') data     [learn]
  2. Wrap in branchable VirtualKernel, fork counterfactual worlds   [branch]
  3. Virtual screening over sysctl trajectories (0 oracle calls)    [search]
  4. Agent harness proposes / critiques / validates                 [act asap]
  5. TruthOracle grounds top-k + reports strength per oracle call   [validate]
"""
import time

from learned_kernel.simulator.env import KernelSimulator
from virtual_kernel import EnsembleDynamics, TruthOracle, VirtualKernel, VirtualLab
from virtual_kernel.search import validate_topk, virtual_screen
from virtual_kernel.active import rank_probes


def main():
    t0 = time.time()
    print("=" * 64)
    print("VIRTUAL KERNEL — executable surrogate of kernel dynamics")
    print("=" * 64)

    print("\n[1/5] learn — training ensemble surrogate from oracle transitions ...")
    ens = EnsembleDynamics.train(n_members=3, n_episodes=16, steps_per_episode=30,
                                 epochs=150, verbose=True)
    print(f"        trained {len(ens)} members on {len(ens.dataset)} (s,a,s') samples")

    sim = KernelSimulator(seed=42)
    kir0 = sim.current_kir()
    vk = VirtualKernel(ens, kir0)
    print(f"\n[2/5] branch — observe + fork counterfactual worlds ...")
    print(f"        observe: latency={kir0.scheduler.avg_latency_ms}ms")
    vk.branch("tight-world")
    vk.branch("loose-world")
    from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
    vk.intervene(PolicyAction(policy_id="d", scheduler=SchedulerAction(target_latency_us=2000)),
                 branch="tight-world")
    vk.intervene(PolicyAction(policy_id="d", scheduler=SchedulerAction(target_latency_us=12000)),
                 branch="loose-world")
    print(f"        compare: {vk.compare('tight-world', 'loose-world')}")

    print("\n[3/5] search — virtual screening (surrogate only, 0 oracle calls) ...")
    results = virtual_screen(ens, kir0, horizon=3, beam_width=6)
    for r in results[:3]:
        print(f"        {r.latencies} predicted_reward={r.predicted_reward:.4f}")

    print("\n[4/5] agents — harness operating inside the virtual kernel ...")
    oracle = TruthOracle(seed=123, n_cpus=2)
    lab = VirtualLab(vk, oracle=oracle)
    report = lab.run(horizon=3, beam_width=6, validate_k=1)
    for f in report.findings:
        print(f"        [{f.agent}] {f.claim}")

    print("\n[5/5] validate — oracle grounding + active-learning probe ...")
    probes = rank_probes(ens, kir0)
    print(f"        top info-gain probe: {probes[0].target_latency_us}us "
          f"(u={probes[0].uncertainty:.4f})")
    print(f"        summary: {report.summary()}")
    print(f"\nDone in {time.time()-t0:.1f}s. "
          f"Validator remains safety authority; surrogate never actuates directly.")


if __name__ == "__main__":
    main()
