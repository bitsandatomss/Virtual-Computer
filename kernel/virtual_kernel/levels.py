"""levels.py — the five-level surrogate hierarchy as an operational certification.

context.txt (lines 660-684) defines the ladder every virtual environment must
climb:

    L1 Exemplar                 reproduce previously observed trajectories
    L2 Interpolative surrogate  predict unseen combinations of known conditions
    L3 Counterfactual surrogate intervene on variables, obtain useful predictions
    L4 Mechanistically grounded  latent mechanisms; generalize substantially OOD
    L5 Scientific surrogate      propose experiments, predict outcomes, quantify
                                 uncertainty, actively decide physical probes

Each level has a PASS/FAIL gate with a measured number — a surrogate is
certified at the highest level it clears CONSECUTIVELY from L1. Skipping is
forbidden: L3 ranking means nothing if L1 replay fails. This is the honest
answer to "how far can a learned surrogate replace explicit simulation as
the substrate on which we conduct experiments" (lines 688-693): exactly as
far as its certified level, on exactly the workloads it was tested on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

LEVEL_NAMES = {
    1: "L1-exemplar",
    2: "L2-interpolative",
    3: "L3-counterfactual",
    4: "L4-mechanistic",
    5: "L5-scientific",
}


def assess_L1(ensemble, train_episodes) -> Dict[str, Any]:
    """Replay: single-step MSE on held-in training trajectories."""
    from .datasets import to_transitions
    ds = to_transitions(train_episodes)
    ev = ensemble.evaluate(ds)
    passed = bool(ev["mse"] <= 0.05)
    return {"level": 1, "name": LEVEL_NAMES[1], "passed": passed,
            "metrics": {"train_mse": ev["mse"]}, "threshold": 0.05}


def assess_L2(ensemble, train_episodes) -> Dict[str, Any]:
    """Interpolation: unseen (state, action) combos, SAME workloads.

    Time-split within training episodes: first 70% of steps are eligible for
    training, last 30% are the test combos. Same demand trajectories, novel
    pairings — the interpolative regime, not memorization (which would ace
    L1 and fail here if the model only memorized).
    """
    from .datasets import Episode, to_transitions
    tr, te = [], []
    for ep in train_episodes:
        cut = max(1, int(0.7 * len(ep.steps)))
        tr.append(Episode(seed=ep.seed, steps=ep.steps[:cut], regimes=ep.regimes[:cut]))
        te.append(Episode(seed=ep.seed, steps=ep.steps[cut:], regimes=ep.regimes[cut:]))
    from .datasets import to_transitions as _tt
    ev_tr = ensemble.evaluate(_tt(tr))
    ev_te = ensemble.evaluate(_tt(te))
    gap = ev_te["mse"] / max(ev_tr["mse"], 1e-12)
    passed = bool(ev_te["mse"] <= 0.06 and gap <= 3.0)
    return {"level": 2, "name": LEVEL_NAMES[2], "passed": passed,
            "metrics": {"interp_mse": ev_te["mse"], "train_mse": ev_tr["mse"],
                        "interp_gap": gap},
            "threshold": {"interp_mse<=0.06": True, "gap<=3.0": True}}


def _l3_sequences(arms: Sequence[int], horizon: int) -> List[List[int]]:
    """Fixed candidate panel: constant policies per arm + representative mixes."""
    a = [int(arms[0]), int(arms[len(arms) // 2]), int(arms[-1])]
    seqs = [[x] * horizon for x in a]
    seqs.append(([a[0], a[1]] * ((horizon + 1) // 2))[:horizon])
    seqs.append(([a[1], a[0]] * ((horizon + 1) // 2))[:horizon])
    seqs.append(([a[2], a[1]] * ((horizon + 1) // 2))[:horizon])
    return seqs


def assess_L3(ensemble, kir0_list, arms: Sequence[int] = (2000, 6000, 12000),
              horizon: int = 4, topk: int = 2, seed: int = 0,
              n_cpus: int = 2, dt_s: float = 0.5) -> Dict[str, Any]:
    """Counterfactual: rank intervention sequences like the oracle, in
    EXPECTATION over workload realizations.

    Metrology note (measured during development): at horizon 2 the exact
    single-trajectory reward is dominated by workload stochasticity, so any
    rank comparison there measures noise, not fidelity. Action effects
    compound past the noise only at longer horizons; hence this gate compares
    MEAN predicted vs MEAN exact rewards over K start states at horizon 4 —
    the estimand a planner actually needs ("which configuration is best on
    average over the workloads it will face").
    """
    from .metrics import spearman_rank, topk_hit_rate
    from .oracle import TruthOracle
    from .planning import rollout_sequence_reward
    kir0s = list(kir0_list)
    oracle = TruthOracle(seed=seed, n_cpus=n_cpus, dt_s=dt_s)
    seqs = _l3_sequences(arms, horizon)
    pred, exact = [], []
    for s in seqs:
        pred.append(float(np.mean([rollout_sequence_reward(ensemble, k, s,
                                                           dt_s=dt_s)[0]
                                   for k in kir0s])))
        exact.append(float(np.mean([oracle.sequence_reward(k, s) for k in kir0s])))
    rho = spearman_rank(pred, exact)
    hit = topk_hit_rate(pred, exact, k=min(topk, len(seqs)))
    passed = bool(rho >= 0.5 and hit >= 1.0)
    return {"level": 3, "name": LEVEL_NAMES[3], "passed": passed,
            "metrics": {"spearman": rho, f"top{topk}_hit": hit,
                        "n_sequences": len(seqs), "n_start_states": len(kir0s),
                        "horizon": horizon,
                        "oracle_calls": len(seqs) * horizon * len(kir0s)},
            "threshold": {"spearman>=0.5": True, "topk_hit>=1.0": True}}


def assess_L4(train_ds, test_ds, train_episodes, test_episodes,
              n_members: int = 3, epochs: int = 100,
              dt_s: float = 0.5) -> Dict[str, Any]:
    """Mechanistic: substantial OOD generalization with conserved invariants.

    Three simultaneous demands (any one failing fails L4):
    (a) zero-shot error gap test/train stays bounded,
    (b) virtual test rollouts break no more invariants than oracle rollouts,
    (c) the OOD detector demonstrably separates train from test regimes.
    """
    from .conservation import conservation_report
    from .datasets import regime_histogram
    from .ensemble import EnsembleDynamics
    from .ood import OODDetector, regime_coverage

    ens = EnsembleDynamics.train(n_members=n_members, epochs=epochs,
                                 dataset=train_ds)
    ev_tr = ens.evaluate(train_ds)
    ev_te = ens.evaluate(test_ds)
    gap = ev_te["mse"] / max(ev_tr["mse"], 1e-12)

    # conservation parity on a few test-regime rollouts
    from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
    from learned_kernel.simulator.env import KernelSimulator
    virt_trajs, orac_trajs = [], []
    for ep in test_episodes[:3]:
        kir0 = ep.steps[0].kir
        seq = [2000, 6000, 12000][:3]
        kv, kir = kir0.model_copy(deep=True), kir0.model_copy(deep=True)
        vt, ot = [kir0.model_copy(deep=True)], [kir0.model_copy(deep=True)]
        sim = KernelSimulator(seed=ep.seed, n_cpus=len(kir0.scheduler.cpus), dt_s=dt_s)
        sim.reset(from_state=kir0)
        for lat in seq:
            a = PolicyAction(policy_id="l4",
                             scheduler=SchedulerAction(target_latency_us=lat))
            kv = ens.predict_kir(kv, a, dt_s=dt_s)
            vt.append(kv)
            ot.append(sim.step(a))
        virt_trajs.append(vt)
        orac_trajs.append(ot)
    cons = conservation_report(virt_trajs, orac_trajs, dt_s)

    det = OODDetector().fit(np.asarray(train_ds.feats))
    ood_rep = det.report(np.asarray(train_ds.feats), np.asarray(test_ds.feats))
    all_reg = [r for ep in train_episodes for r in ep.regimes]
    tst_reg = [r for ep in test_episodes for r in ep.regimes]
    cov = regime_coverage(all_reg, tst_reg)

    passed = bool(gap <= 3.0 and ev_te["mse"] <= 0.08
                  and cons["parity"] and ood_rep["separation"] >= 0.5)
    return {"level": 4, "name": LEVEL_NAMES[4], "passed": passed,
            "metrics": {"test_mse": ev_te["mse"], "train_mse": ev_tr["mse"],
                        "zero_shot_gap": gap,
                        "conservation_parity": cons["parity"],
                        "conservation": cons,
                        "ood_separation": ood_rep["separation"],
                        "ood_test_flag_rate": ood_rep["test_flag_rate"],
                        "regime_coverage": cov,
                        "train_regimes": regime_histogram(train_episodes),
                        "test_regimes": regime_histogram(test_episodes)},
            "threshold": {"gap<=3.0": True, "test_mse<=0.08": True,
                          "parity": True, "separation>=0.5": True}}


def assess_L5(ensemble, kir0, oracle, test_ds=None, rounds: int = 2,
              probes_per_round: int = 3, epochs: int = 30,
              dt_s: float = 0.5) -> Dict[str, Any]:
    """Scientific: propose probes, reduce error, know uncertainty.

    The surrogate must (a) lower its own error by acting on its uncertainty
    (active loop mse_after <= mse_before on probed transitions) and
    (b) be non-anti-calibrated on UNSEEN workloads: uncertainty-error
    correlation > 0 (never over the training pool — see note there).

    Why correlation-floor, not tercile-monotonicity: with single-step errors
    near the noise floor (MSE ~0.008), rank-ordering tiny errors by
    uncertainty is inherently low-signal, and strict monotonicity flips run
    to run on identical configs — a coin-flip gate. corr > 0 stably measures
    "knows when it doesn't know, better than chance"; the full terciles are
    still reported for scrutiny.
    """
    from .active import active_loop
    res = active_loop(ensemble, kir0, oracle, rounds=rounds,
                      probes_per_round=probes_per_round, epochs=epochs, dt_s=dt_s)
    last = res.rounds[-1] if res.rounds else {}
    improved = bool(last.get("mse_after", float("inf")) <= last.get("mse_before", 0.0))
    cal_ds = test_ds if test_ds is not None else ensemble.dataset
    cal = ensemble.calibration(cal_ds) if cal_ds is not None else {}
    corr = cal.get("uncertainty_error_corr", float("nan"))
    import math
    calibrated = bool(math.isfinite(corr) and corr > 0.0)
    passed = bool(improved and calibrated)
    return {"level": 5, "name": LEVEL_NAMES[5], "passed": passed,
            "metrics": {"rounds": res.rounds, "oracle_calls": res.n_oracle_calls,
                        "final_mse": res.final_mse, "improved": improved,
                        "calibration": cal, "calibrated": calibrated},
            "threshold": {"improved": True, "calibrated": True}}


def certify(config, levels: Sequence[int] = (1, 2, 3, 4, 5),
            verbose: bool = False) -> Dict[str, Any]:
    """Train once on the config's train split, then gate level by level."""
    from .datasets import train_holdout_split
    from .ensemble import EnsembleDynamics
    from .fidelity import STANDARD_SPECS, fidelity_report
    from .oracle import TruthOracle

    train_ds, test_ds, train_eps, test_eps = train_holdout_split(
        config.train_seeds, config.test_seeds, config.steps_per_episode,
        config.arms, config.n_cpus, config.dt_s)
    ens = EnsembleDynamics.train_config(config, dataset=train_ds,
                                          verbose=verbose, episodes=train_eps)
    test_kir0s = [ep.steps[len(ep.steps) // 2].kir for ep in test_eps[:4]]
    oracle = TruthOracle(seed=config.seed, n_cpus=config.n_cpus, dt_s=config.dt_s)

    results: List[Dict[str, Any]] = []
    if 1 in levels:
        results.append(assess_L1(ens, train_eps))
    if 2 in levels:
        results.append(assess_L2(ens, train_eps))
    if 3 in levels:
        results.append(assess_L3(ens, test_kir0s, arms=tuple(config.arms[:3]),
                                 horizon=4, seed=config.seed, n_cpus=config.n_cpus,
                                 dt_s=config.dt_s))
    if 4 in levels:
        results.append(assess_L4(train_ds, test_ds, train_eps, test_eps,
                                 n_members=config.n_members,
                                 epochs=max(config.epochs // 3, 20),
                                 dt_s=config.dt_s))
    if 5 in levels:
        results.append(assess_L5(ens, test_kir0s[0], oracle, test_ds=test_ds,
                                 dt_s=config.dt_s))

    certified = 0
    for r in sorted(results, key=lambda d: d["level"]):
        if r["passed"] and r["level"] == certified + 1:
            certified = r["level"]
        else:
            break

    by_level = {r["level"]: r for r in results}
    l3m = by_level.get(3, {}).get("metrics", {})
    topk_val = next((v for k, v in l3m.items()
                     if k.startswith("top") and k.endswith("_hit")), None)
    measurements = {
        "spearman_rank": l3m.get("spearman"),
        "topk_hit_rate": topk_val,
        "test_mse": by_level.get(4, {}).get("metrics", {}).get("test_mse"),
        "divergence_step": None,  # measured by run_degradation horizon sweeps
        "worst_case_gap": None,   # measured by run_compositional
        "conservation_parity": (1.0 if by_level.get(4, {}).get("metrics", {})
                                .get("conservation_parity") else 0.0) if 4 in by_level else None,
    }
    fib = fidelity_report(STANDARD_SPECS, measurements)
    return {"results": results, "certified_level": certified,
            "certified_name": LEVEL_NAMES.get(certified, "L0-uncertified"),
            "fidelity": fib, "config_hash": config.hash,
            "context": {"n_train": len(train_ds), "n_test": len(test_ds)}}
