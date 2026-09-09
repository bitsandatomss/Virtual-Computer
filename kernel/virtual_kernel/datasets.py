"""datasets.py — workload-regime traces, transition datasets, held-out splits.

context.txt lessons encoded here:
- Zero-shot generalization (held-out cell lines -> held-out workloads): the
  train/test split is by workload SEED, so test episodes come from demand
  trajectories the surrogate never saw. Memorizing training workloads scores
  well on train but fails here — exactly the pressure test demanded.
- Partial observability (Vesuvius lesson): workload demand is hidden state.
  Traces store only (kir, action, next_kir); demand is never leaked to the
  learner. Regime labels are derived from OBSERVABLES for analysis only.
- Transition-function evidence: the dataset is intervention->response pairs,
  not observational telemetry (the bug the old self-kernel trainer had).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from learned_kernel.policy.schemas import (
    KernelIntermediateRepresentation,
    PolicyAction,
    SchedulerAction,
)
from learned_kernel.simulator.env import KernelSimulator

from .dynamics import DynamicsDataset, encode_action, encode_kir, kir_observables


# --------------------------------------------------------------------------- #
# Regime labels (observable-side only)                                        #
# --------------------------------------------------------------------------- #

def demand_band(mean_util: float) -> str:
    """Coarse workload regime from observable mean utilization."""
    if mean_util < 0.35:
        return "idle"
    if mean_util < 0.7:
        return "mid"
    return "saturated"


def regime_of(kir: KernelIntermediateRepresentation) -> str:
    cpus = list(kir.scheduler.cpus.values())
    u = sum(c.utilization for c in cpus) / len(cpus) if cpus else 0.0
    return demand_band(u)


# --------------------------------------------------------------------------- #
# Trace capture                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    kir: KernelIntermediateRepresentation
    action: Optional[PolicyAction]
    next_kir: KernelIntermediateRepresentation


@dataclass
class Episode:
    seed: int
    steps: List[Step] = field(default_factory=list)
    regimes: List[str] = field(default_factory=list)


def _random_action(rng: np.random.Generator, arms: Tuple[int, ...]) -> PolicyAction:
    return PolicyAction(policy_id="trace",
                        scheduler=SchedulerAction(target_latency_us=int(rng.choice(arms))))


def capture_traces(n_episodes: int, steps_per_episode: int, base_seed: int,
                   arms: Tuple[int, ...] = (1000, 2000, 4000, 6000, 12000, 24000),
                   n_cpus: int = 2, dt_s: float = 0.5,
                   policy_fn: Optional[Callable[[KernelIntermediateRepresentation],
                                                Optional[PolicyAction]]] = None,
                   switch_prob: float = 0.35) -> List[Episode]:
    """Roll the EXACT oracle to produce intervention->response episodes.

    policy_fn, if given, is rolled out (on-policy traces); otherwise random
    arms with action repeats (coverage traces). Either way every step records
    the action actually taken — no observational-trajectory shortcut.
    """
    rng = np.random.default_rng(base_seed ^ 0x7A1C)
    episodes: List[Episode] = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        sim = KernelSimulator(seed=seed, n_cpus=n_cpus, dt_s=dt_s)
        kir = sim.current_kir()
        cur_lat = int(rng.choice(np.array(arms)))
        episode = Episode(seed=seed)
        for _ in range(steps_per_episode):
            if policy_fn is not None:
                act = policy_fn(kir)
            else:
                if rng.random() < switch_prob:
                    cur_lat = int(rng.choice(np.array(arms)))
                act = PolicyAction(policy_id="trace",
                                   scheduler=SchedulerAction(target_latency_us=cur_lat))
            nxt = sim.step(act)
            episode.steps.append(Step(kir=kir, action=act, next_kir=nxt))
            episode.regimes.append(regime_of(nxt))
            kir = nxt
        episodes.append(episode)
    return episodes


def to_transitions(episodes: List[Episode], dt_s: float = 0.5) -> DynamicsDataset:
    F, A, Y, seeds = [], [], [], []
    for ep in episodes:
        prev_sw = ep.steps[0].kir.scheduler.total_context_switches
        for s in ep.steps:
            F.append(encode_kir(s.kir))
            A.append(encode_action(s.action)[0])
            Y.append(kir_observables(s.next_kir, prev_sw, dt_s))
            prev_sw = s.next_kir.scheduler.total_context_switches
        seeds.append(ep.seed)
    return DynamicsDataset(feats=np.array(F), acts=np.array(A),
                           targets=np.array(Y), seeds=seeds)


def windows_from_episodes(episodes: List[Episode], horizon: int,
                          dt_s: float = 0.5):
    """Overlapping H-step windows for unrolled (multi-step) training.

    Returns Fw (N,H,6), Aw (N,H), Yw (N,H,4): the H true features, action
    codes and observable targets of each window. Switch-delta targets use
    the running counter within the episode, exactly like to_transitions.
    """
    Fw, Aw, Yw = [], [], []
    for ep in episodes:
        prev_sw = ep.steps[0].kir.scheduler.total_context_switches
        feats, acts, tgts = [], [], []
        for s in ep.steps:
            feats.append(encode_kir(s.kir))
            acts.append(encode_action(s.action)[0])
            tgts.append(kir_observables(s.next_kir, prev_sw, dt_s))
            prev_sw = s.next_kir.scheduler.total_context_switches
        for t in range(len(ep.steps) - horizon + 1):
            Fw.append(feats[t:t + horizon])
            Aw.append(acts[t:t + horizon])
            Yw.append(tgts[t:t + horizon])
    import numpy as np
    return np.array(Fw), np.array(Aw), np.array(Yw)


def train_holdout_split(train_seeds: List[int], test_seeds: List[int],
                        steps_per_episode: int, arms, n_cpus: int = 2,
                        dt_s: float = 0.5) -> Tuple[DynamicsDataset, DynamicsDataset,
                                                   List[Episode], List[Episode]]:
    """Zero-shot split: disjoint workload seeds for train vs test."""
    train_eps = capture_traces(len(train_seeds), steps_per_episode,
                               base_seed=train_seeds[0], arms=tuple(arms),
                               n_cpus=n_cpus, dt_s=dt_s)
    # re-seed episodes to the requested seed sets exactly
    train_eps = _reseed(train_eps, train_seeds, steps_per_episode, arms, n_cpus, dt_s)
    test_eps = _reseed([], test_seeds, steps_per_episode, arms, n_cpus, dt_s)
    return to_transitions(train_eps, dt_s), to_transitions(test_eps, dt_s), train_eps, test_eps


def _reseed(_ignored, seeds: List[int], steps: int, arms, n_cpus: int,
            dt_s: float) -> List[Episode]:
    out: List[Episode] = []
    for s in seeds:
        eps = capture_traces(1, steps, base_seed=s, arms=tuple(arms),
                             n_cpus=n_cpus, dt_s=dt_s)
        eps[0].seed = s
        out.append(eps[0])
    return out


def dataset_hash(ds: DynamicsDataset) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ds.feats).tobytes())
    h.update(np.ascontiguousarray(ds.acts).tobytes())
    h.update(np.ascontiguousarray(ds.targets).tobytes())
    h.update(json.dumps(sorted(ds.seeds)).encode())
    return h.hexdigest()[:16]


def regime_histogram(episodes: List[Episode]) -> dict:
    from collections import Counter
    c: Counter = Counter()
    for ep in episodes:
        c.update(ep.regimes)
    return dict(c)
