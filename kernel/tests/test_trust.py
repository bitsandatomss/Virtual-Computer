"""Tests for the trust layer: conservation, OOD, fidelity, propagation, levels."""
import numpy as np

from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
from learned_kernel.simulator.env import KernelSimulator
from virtual_kernel import (
    OODDetector,
    STANDARD_SPECS,
    check_transition,
    conservation_report,
    fidelity_report,
    propagation_law,
    regime_coverage,
    score_trajectory,
)
from virtual_kernel.levels import _l3_sequences, assess_L1, certify
from virtual_kernel import EnsembleDynamics, fast_config
from virtual_kernel.datasets import capture_traces, to_transitions


def _oracle_traj(seed=11, steps=8):
    sim = KernelSimulator(seed=seed)
    kir = sim.current_kir()
    trajs = [kir]
    for i in range(steps):
        kir = sim.step(PolicyAction(
            policy_id="t",
            scheduler=SchedulerAction(target_latency_us=6000)))
        trajs.append(kir)
    return trajs


def test_conservation_clean_on_oracle():
    s = score_trajectory(_oracle_traj())
    assert s["violations"] == 0, s


def test_conservation_flags_time_reversal_and_counter_reset():
    kirs = _oracle_traj()
    bad = kirs[1].model_copy(deep=True)
    bad.scheduler.total_context_switches = kirs[0].scheduler.total_context_switches - 5
    bad.timestamp = kirs[0].timestamp  # no advance
    vs = check_transition(kirs[0], bad)
    kinds = {v["invariant"] for v in vs}
    assert "I1-monotonic-switches" in kinds
    assert "I2-time-advance" in kinds


def test_conservation_parity_oracle_vs_oracle():
    rep = conservation_report([_oracle_traj(seed=11)], [_oracle_traj(seed=12)])
    assert rep["parity"] is True
    assert rep["oracle"]["violation_rate"] == 0.0


def test_ood_flags_far_point_and_separates_shift():
    rng = np.random.default_rng(0)
    train = rng.uniform(0.1, 0.6, size=(200, 6))
    det = OODDetector().fit(train)
    assert bool(det.flagged(np.array([[5.0] * 6]))[0]) is True
    assert float(det.flagged(train).mean()) <= 0.05
    shifted = rng.uniform(0.8, 1.0, size=(100, 6))
    rep = det.report(train, shifted)
    assert rep["separation"] > 0.9
    assert rep["test_flag_rate"] > rep["train_flag_rate"]


def test_regime_coverage_names_unseen_band():
    cov = regime_coverage(["idle", "mid"], ["mid", "saturated"])
    assert cov["unseen_bands"] == ["saturated"]
    assert cov["coverage"] == 0.5


def test_fidelity_report_pass_and_fail():
    ok = fidelity_report(STANDARD_SPECS, {
        "spearman_rank": 0.8, "topk_hit_rate": 1.0, "test_mse": 0.01,
        "divergence_step": 6, "worst_case_gap": -0.1,
        "conservation_parity": 1.0})
    assert ok["all_passed"] is True
    bad = fidelity_report(STANDARD_SPECS, {
        "spearman_rank": 0.1, "topk_hit_rate": 0.0, "test_mse": 0.5,
        "divergence_step": 1, "worst_case_gap": -0.9,
        "conservation_parity": 0.0})
    assert bad["all_passed"] is False
    assert bad["n_passed"] < bad["n_decided"]


def test_propagation_law_recovers_growth():
    h = np.arange(8)
    curves = [0.01 * (1.5 ** h) for _ in range(3)]
    law = propagation_law(curves)
    assert abs(law["growth_rate"] - 1.5) < 0.05
    assert abs(law["doubling_horizon"] - (np.log(2) / np.log(1.5))) < 0.2
    assert law["r_squared"] > 0.99


def test_l3_panel_shape():
    seqs = _l3_sequences((1000, 6000, 24000), 4)
    assert len(seqs) == 6
    assert all(len(s) == 4 for s in seqs)
    assert [1000] * 4 in seqs and [24000] * 4 in seqs


def test_levels_structure_smoke():
    cfg = fast_config("levels-test")
    cfg.train_seeds = list(range(7000, 7004))
    cfg.test_seeds = list(range(9000, 9002))
    cfg.steps_per_episode = 10
    cfg.n_members, cfg.epochs = 2, 25
    out = certify(cfg, levels=(1, 2))
    assert [r["level"] for r in out["results"]] == [1, 2]
    assert isinstance(out["certified_level"], int)
    assert out["results"][0]["metrics"]["train_mse"] < 0.05


def test_assess_L1_passes_on_train_distribution():
    eps = capture_traces(4, 12, base_seed=7000)
    ens = EnsembleDynamics.train(n_members=2, epochs=30,
                                 dataset=to_transitions(eps))
    r = assess_L1(ens, eps)
    assert r["passed"] is True


def test_unrolled_fit_reduces_multistep_error():
    """Mixed single+unrolled training is wired correctly: unrolled loss falls."""
    from virtual_kernel.datasets import to_transitions, windows_from_episodes
    eps = capture_traces(4, 12, base_seed=7100)
    ds = to_transitions(eps)
    Fw, Aw, Yw = windows_from_episodes(eps, 2)
    ens = EnsembleDynamics.train(n_members=1, epochs=10, dataset=ds)
    m = ens.members[0]
    l_before, _ = m._unrolled_forward_backward(Fw, Aw, Yw)
    m.fit(ds.feats, ds.acts, ds.targets, epochs=15,
          windows=(Fw, Aw, Yw), unroll_weight=2.0)
    l_after, _ = m._unrolled_forward_backward(Fw, Aw, Yw)
    assert l_after < l_before, f"{l_before} -> {l_after}"
