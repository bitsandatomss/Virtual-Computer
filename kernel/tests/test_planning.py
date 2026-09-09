"""Tests for planning: MCTS shape/determinism, robust maximin semantics."""
from learned_kernel.simulator.env import KernelSimulator
from virtual_kernel import EnsembleDynamics, fast_config, mcts_search, robust_search


def _ens():
    cfg = fast_config("test-planning")
    return EnsembleDynamics.train(n_members=2, n_episodes=4, steps_per_episode=8,
                                  base_seed=7000, epochs=20, n_cpus=2), cfg


def test_mcts_full_horizon_and_deterministic():
    ens, cfg = _ens()
    kir0 = KernelSimulator(seed=11, n_cpus=2).current_kir()
    r1 = mcts_search(ens, kir0, horizon=2, sims=30, seed=0, arms=tuple(cfg.arms))
    r2 = mcts_search(ens, kir0, horizon=2, sims=30, seed=0, arms=tuple(cfg.arms))
    assert len(r1.best_latencies) == 2
    assert r1.best_latencies == r2.best_latencies
    assert sum(r1.visits.values()) > 0
    assert all(lat in cfg.arms for lat in r1.best_latencies)


def test_robust_selects_maximin():
    ens, cfg = _ens()
    kirs = [KernelSimulator(seed=s, n_cpus=2).current_kir() for s in (11, 12)]
    cands = [[2000, 2000], [12000, 12000], [6000, 6000]]
    out = robust_search(ens, kirs, cands)
    assert out["n_scenarios"] == 2
    worsts = [r["worst_case"] for r in out["ranking"]]
    assert worsts == sorted(worsts, reverse=True)
    assert out["selected"]["worst_case"] == max(worsts)
