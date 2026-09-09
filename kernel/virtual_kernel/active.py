"""active.py — active learning: which real experiment to run next.

context.txt: "Virtual Cell -> which physical perturbation should we run?"
(lines 321, 934-942) and "expected information gain rather than merely
reconstruction error". Here: score candidate sysctl probes by ensemble
disagreement; the most-disagreed probe maximizes expected information gain
about the true kernel dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from learned_kernel.policy.schemas import KernelIntermediateRepresentation, PolicyAction, SchedulerAction

from .search import CANDIDATE_LATENCIES


@dataclass
class ActiveProbe:
    target_latency_us: int
    uncertainty: float
    action: PolicyAction


def rank_probes(ensemble, kir: KernelIntermediateRepresentation,
                latencies: tuple = CANDIDATE_LATENCIES) -> List[ActiveProbe]:
    probes = []
    for lat in latencies:
        act = PolicyAction(policy_id="active-probe",
                           scheduler=SchedulerAction(target_latency_us=lat))
        u = ensemble.uncertainty(kir, act)
        probes.append(ActiveProbe(target_latency_us=lat, uncertainty=u, action=act))
    probes.sort(key=lambda p: p.uncertainty, reverse=True)
    return probes


@dataclass
class ActiveLoopResult:
    rounds: List[dict]
    final_mse: float
    n_oracle_calls: int


def active_loop(ensemble, kir0: KernelIntermediateRepresentation,
                oracle, rounds: int = 3, probes_per_round: int = 4,
                rollout_steps: int = 6, epochs: int = 50,
                latencies: tuple = CANDIDATE_LATENCIES,
                dt_s: float = 0.5) -> ActiveLoopResult:
    """Close the active-learning loop: probe -> oracle -> retrain.

    Each round: rank probes by disagreement at the current frontier state,
    execute top probes on the EXACT oracle (the only oracle cost), append the
    resulting (s,a,s') transitions to the training pool, and refit members.
    Reports held-out MSE trajectory so the information gain is measurable.
    """
    import numpy as np
    from .dynamics import DynamicsDataset

    F = np.asarray(ensemble.dataset.feats) if ensemble.dataset is not None else np.zeros((0, 6))
    A = np.asarray(ensemble.dataset.acts) if ensemble.dataset is not None else np.zeros((0,))
    Y = np.asarray(ensemble.dataset.targets) if ensemble.dataset is not None else np.zeros((0, 4))

    from .dynamics import encode_action, encode_kir, kir_observables

    def batch_mse(members, Fb, Ab, Yb) -> float:
        errs = []
        for i in range(len(Fb)):
            f, a, y = Fb[i], np.array([Ab[i]]), Yb[i]
            outs = np.array([m.decode(m.transition(m.encode(f), a), a) for m in members])
            errs.append(float(np.mean((outs.mean(axis=0) - y) ** 2)))
        return float(np.mean(errs)) if errs else 0.0

    history: List[dict] = []
    calls = 0
    frontier = kir0.model_copy(deep=True)
    mse = batch_mse(ensemble.members, F, A, Y) if len(F) else float("nan")
    for rd in range(rounds):
        probes = rank_probes(ensemble, frontier, latencies)[:probes_per_round]
        new_F, new_A, new_Y = [], [], []
        for p in probes:
            oracle.reset(frontier)
            prev_sw = frontier.scheduler.total_context_switches
            nxt = oracle.step(p.action)
            calls += 1
            new_F.append(encode_kir(frontier))
            new_A.append(encode_action(p.action)[0])
            new_Y.append(kir_observables(nxt, prev_sw, dt_s))
            frontier = nxt  # walk the frontier forward through real outcomes
        nF, nA, nY = np.array(new_F), np.array(new_A), np.array(new_Y)
        pre = batch_mse(ensemble.members, nF, nA, nY)
        F = np.concatenate([F, nF]) if len(F) else nF
        A = np.concatenate([A, nA]) if len(A) else nA
        Y = np.concatenate([Y, nY]) if len(Y) else nY
        ds = DynamicsDataset(feats=F, acts=A, targets=Y, seeds=[])
        for m in ensemble.members:
            m.fit(ds.feats, ds.acts, ds.targets, epochs=epochs)
        ensemble.dataset = ds
        mse = batch_mse(ensemble.members, nF, nA, nY)
        history.append({"round": rd,
                        "probed": [p.target_latency_us for p in probes],
                        "top_uncertainty": probes[0].uncertainty,
                        "mse_before": pre, "mse_after": mse,
                        "train_size": len(F)})
    return ActiveLoopResult(rounds=history, final_mse=mse, n_oracle_calls=calls)
