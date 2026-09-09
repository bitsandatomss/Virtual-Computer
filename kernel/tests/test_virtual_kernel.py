"""tests/test_virtual_kernel.py — hermetic checks for the virtual kernel layer."""
import math

from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
from learned_kernel.simulator.env import KernelSimulator
from virtual_kernel import EnsembleDynamics, TruthOracle, VirtualKernel, VirtualLab
from virtual_kernel.search import virtual_screen
from virtual_kernel.active import rank_probes


def _small_ensemble():
    return EnsembleDynamics.train(n_members=2, n_episodes=6, steps_per_episode=15,
                                  epochs=60)


def test_dynamics_learns_better_than_chance():
    import numpy as np
    ens = _small_ensemble()
    F, Y = ens.dataset.feats, ens.dataset.targets
    acts = ens.dataset.acts
    m = ens.members[0]
    errs = []
    for i in range(0, len(F), 5):
        f = F[i]
        a = np.array([acts[i]])
        pred_o = m.decode(m.transition(m.encode(f), a))
        errs.append(float(((pred_o - Y[i]) ** 2).mean()))
    mse = sum(errs) / len(errs)
    var = float(Y.var())
    assert mse < var, f"surrogate did not learn: mse={mse:.4f} var={var:.4f}"


def test_surrogate_is_action_sensitive():
    """Regression: transition must not collapse to E[y|z] ignoring the action."""
    import numpy as np
    ens = _small_ensemble()
    kir0 = KernelSimulator(seed=6).current_kir()
    a_tight = PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=1000))
    a_loose = PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=24000))
    o1 = ens.predict_mean_obs(kir0, a_tight)
    o2 = ens.predict_mean_obs(kir0, a_loose)
    assert float(np.abs(o1 - o2).max()) > 1e-3, f"action-insensitive: {o1} vs {o2}"
    # direction: looser window must predict higher latency (queueing term)
    assert o2[2] > o1[2], f"wrong action direction: {o1[2]} vs {o2[2]}"


def test_branching_isolation():
    ens = _small_ensemble()
    kir0 = KernelSimulator(seed=1).current_kir()
    vk = VirtualKernel(ens, kir0)
    vk.branch("a")
    vk.branch("b")
    vk.intervene(PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=1000)), branch="a")
    vk.intervene(PolicyAction(policy_id="t", scheduler=SchedulerAction(target_latency_us=24000)), branch="b")
    cmp = vk.compare("a", "b")
    assert cmp["kir_distance"] >= 0.0
    assert vk.observe("a").timestamp == vk.observe("b").timestamp  # same steps


def test_rollout_sequential_chain():
    from learned_kernel.policy.core import HeuristicLatencyPolicy
    ens = _small_ensemble()
    kir0 = KernelSimulator(seed=2).current_kir()
    vk = VirtualKernel(ens, kir0)
    traj = vk.rollout(HeuristicLatencyPolicy(), steps=4)
    assert len(traj) == 4
    assert len(vk.observe().scheduler.cpus) == 2


def test_search_and_oracle_validation():
    ens = _small_ensemble()
    kir0 = KernelSimulator(seed=3).current_kir()
    results = virtual_screen(ens, kir0, horizon=2, beam_width=3)
    assert len(results) == 3
    assert len(results[0].latencies) == 2
    oracle = TruthOracle(seed=3)
    from virtual_kernel.search import validate_topk
    validate_topk(results, oracle, kir0, k=1)
    assert results[0].oracle_reward is not None
    assert math.isfinite(results[0].oracle_reward)


def test_active_probes_ranked():
    ens = _small_ensemble()
    kir0 = KernelSimulator(seed=4).current_kir()
    probes = rank_probes(ens, kir0)
    assert len(probes) == 6
    us = [p.uncertainty for p in probes]
    assert us == sorted(us, reverse=True)
    assert all(p.uncertainty >= 0 for p in probes)


def test_lab_report():
    ens = _small_ensemble()
    kir0 = KernelSimulator(seed=5).current_kir()
    vk = VirtualKernel(ens, kir0)
    lab = VirtualLab(vk, oracle=TruthOracle(seed=5))
    rep = lab.run(horizon=2, beam_width=3, validate_k=1)
    assert rep.best_oracle_reward is not None
    assert rep.oracle_calls > 0
    assert rep.virtual_worlds > rep.oracle_calls  # leverage: many virtual, few real
    agents = {f.agent for f in rep.findings}
    assert {"perturbation", "adversarial", "validation"} <= agents
