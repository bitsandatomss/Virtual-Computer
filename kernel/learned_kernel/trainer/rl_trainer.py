"""
trainer/rl_trainer.py — Learned Kernel offline policy optimiser.

Fixes over the previous revision
--------------------------------
1. CAUSAL evaluation: candidate policies are scored by *rolling them out
   through KernelSimulator*, so reward genuinely depends on the actions the
   candidate would take. The old code scored observational trajectories whose
   rewards were independent of the policy, making every gradient ~0.
2. Working optimiser: annealed multi-scale coordinate search with paired
   common random numbers and accept-only-if-better moves. (The old numerical
   gradient scheme was doubly broken: eps=200 on dimensionless weights was
   astronomically large, and the /2*eps normalisation made bias steps ~0.01us,
   so nothing ever moved.)
3. Temporal train/holdout split + best-checkpoint tracking, so "improvement"
   claims generalise beyond the data being optimised.
4. Seeded, deterministic runs; persisted JSON artifacts with checksums.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from ..policy.schemas import KernelIntermediateRepresentation, PolicyAction, SchedulerAction
from ..policy.core import PolicyBase
from ..simulator.env import KernelSimulator
from .reward import RewardCalculator
from ..validator.core import PolicyValidator

# --------------------------------------------------------------------------- #
# Feature extraction                                                           #
# --------------------------------------------------------------------------- #
_MAX_RUNNABLE = 64          # normalisation constant
_NORM_LATENCY_MS = 20.0
_NORM_NET_QUEUE = 10_000


def extract_features(kstate: KernelIntermediateRepresentation) -> List[float]:
    """Return a normalised 6-D feature vector from a KIR snapshot."""
    cpus = list(kstate.scheduler.cpus.values())
    mean_util = sum(c.utilization for c in cpus) / len(cpus) if cpus else 0.0
    total_runnable = sum(c.runnable_tasks for c in cpus)
    runnable_norm = min(total_runnable, _MAX_RUNNABLE) / _MAX_RUNNABLE
    lat_norm = min(kstate.scheduler.avg_latency_ms, _NORM_LATENCY_MS) / _NORM_LATENCY_MS
    mem_pressure = kstate.memory.pressure if kstate.memory else 0.0
    net_q = (min(kstate.network.rx_queue_len, _NORM_NET_QUEUE) / _NORM_NET_QUEUE
             if kstate.network else 0.0)
    irq = sum(c.irq_time for c in cpus) / len(cpus) if cpus else 0.0
    return [mean_util, runnable_norm, lat_norm, mem_pressure, net_q, irq]


# KIR dot-paths each feature derives from — keeps the Policy ABI honest:
# introspection must reveal which kernel state a policy actually reads.
FEATURE_KIR_PATHS: List[str] = [
    "scheduler.cpus.*.utilization",
    "scheduler.cpus.*.runnable_tasks",
    "scheduler.avg_latency_ms",
    "memory.pressure",
    "network.rx_queue_len",
    "scheduler.cpus.*.irq_time",
]


# --------------------------------------------------------------------------- #
# Learned policy                                                               #
# --------------------------------------------------------------------------- #
_MIN_LAT_US = 1_000
_MAX_LAT_US = 24_000


class LearnedLinearPolicy(PolicyBase):
    """
    Verifiable linear policy:  lat_us = clip(w · f + bias, MIN, MAX)

    All weights and bias are learned from telemetry via OfflineTrainer;
    nothing is hardcoded in production paths.
    """

    def __init__(self, weights: List[float], bias: float):
        if len(weights) != 6:
            raise ValueError(f"LearnedLinearPolicy requires 6 weights, got {len(weights)}")
        self.weights = [float(w) for w in weights]
        self.bias = float(bias)
        self._rng = random.Random(hash((tuple(self.weights), self.bias)) & 0xFFFF)

    @property
    def policy_id(self) -> str: return "learned-linear-v3"

    @property
    def version(self) -> str: return "3.0"

    @property
    def observation_schema(self) -> List[str]:
        return list(FEATURE_KIR_PATHS)

    @property
    def action_schema(self) -> List[str]:
        return ["scheduler.target_latency_us"]

    def decide(self, kstate: KernelIntermediateRepresentation) -> PolicyAction:
        feats = extract_features(kstate)
        raw = sum(w * f for w, f in zip(self.weights, feats)) + self.bias
        lat = int(max(_MIN_LAT_US, min(_MAX_LAT_US, raw)))
        return PolicyAction(
            policy_id=self.policy_id,
            scheduler=SchedulerAction(target_latency_us=lat),
        )

    # ---- artifact persistence ------------------------------------------ #

    def to_artifact(self, dataset_sha256: str = "", report: dict | None = None) -> dict:
        payload = {
            "kind": "learned-linear-policy",
            "policy_id": self.policy_id,
            "version": self.version,
            "feature_paths": FEATURE_KIR_PATHS,
            "weights": self.weights,
            "bias": self.bias,
            "bounds": [_MIN_LAT_US, _MAX_LAT_US],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": dataset_sha256,
            "report": report or {},
        }
        checksum = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return {"payload": payload, "sha256": checksum}

    @classmethod
    def from_artifact(cls, artifact: dict) -> "LearnedLinearPolicy":
        payload = artifact["payload"]
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if expected != artifact.get("sha256"):
            raise ValueError("Policy artifact checksum mismatch — refusing to load")
        pol = cls(weights=payload["weights"], bias=payload["bias"])
        return pol

    def __repr__(self) -> str:
        names = ["util", "runn", "lat", "mem", "netq", "irq"]
        terms = ", ".join(f"{n}: {w:+.3f}" for n, w in zip(names, self.weights))
        return f"LearnedLinearPolicy(bias={self.bias:.1f}, [{terms}])"


# --------------------------------------------------------------------------- #
# Trainer                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class TrainingReport:
    baseline_train: float = 0.0
    baseline_val: float = 0.0
    final_train: float = 0.0
    final_val: float = 0.0
    best_round: int = -1
    improved: bool = False
    train_curve: List[float] = field(default_factory=list)
    val_curve: List[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "baseline_train": round(self.baseline_train, 4),
            "baseline_val": round(self.baseline_val, 4),
            "final_train": round(self.final_train, 4),
            "final_val": round(self.final_val, 4),
            "best_round": self.best_round,
            "improved": self.improved,
        }


_ROUNDS = 25


class OfflineTrainer:
    """
    Derivative-free coordinate optimiser over LearnedLinearPolicy's
    (weights, bias) space, scored by simulator rollouts.

    Why not numerical gradients: the previous revision estimated dR/dw via
    (R(w+eps)-R(w-eps))/(2*eps) and applied it as lr*grad. For the bias,
    measured in microseconds, that normalization yields steps of ~0.01 us —
    the optimiser never moved (bias stayed pinned at its initialisation,
    "learning" was cosmetic). For a 7-dimensional objective evaluated with
    paired common random numbers, annealed multi-scale coordinate search is
    simpler, monotone on train (accept-only-if-better), and free of
    learning-rate pathologies:

      • proposal scales anneal geometrically: s_r = s0 * decay^round
      • each coordinate tries {-s, -s/4, +s/4, +s}; best strict improvement wins
      • every round's candidate is scored on a held-out split; the
        best-holdout checkpoint is what gets returned
    """

    def __init__(
        self,
        rounds: int = _ROUNDS,
        horizon: int = 20,
        val_fraction: float = 0.3,
        seed: int = 2024,
        weight_clip: float = 12.0,
        w_step0: float = 2.0,
        b_step0_us: float = 5000.0,
        decay: float = 0.75,
    ):
        self.rounds = rounds
        self.horizon = horizon
        self.val_fraction = val_fraction
        self.seed = seed
        self.weight_clip = weight_clip
        self.w_step0 = w_step0
        self.b_step0_us = b_step0_us
        self.decay = decay
        self.reward_fn = RewardCalculator()
        self.validator = PolicyValidator()

    # ------------------------------------------------------------------ #

    def dataset_hash(self, trajectories) -> str:
        h = hashlib.sha256()
        for traj in trajectories:
            for ks in traj:
                h.update(ks.model_dump_json().encode())
        return h.hexdigest()

    def _split(self, trajectories):
        """Temporal split inside each trajectory (train = earliest fraction)."""
        train_trajs, val_starts = [], []
        for traj in trajectories:
            cut = max(1, int(len(traj) * (1.0 - self.val_fraction)))
            train_trajs.append(traj[:cut])
            if len(traj) - cut > 0:
                val_starts.append(traj[cut:])
        return train_trajs, val_starts

    def _rollout(self, weights: List[float], bias: float,
                 seed_states: List[KernelIntermediateRepresentation],
                 eval_seed: int) -> float:
        policy = LearnedLinearPolicy(weights=weights, bias=bias)
        total = 0.0
        env = KernelSimulator(seed=self.seed ^ eval_seed)
        for i, start in enumerate(seed_states):
            env.reset(from_state=start)
            prev = env.current_kir()
            for _ in range(self.horizon):
                try:
                    action = policy.decide(prev)
                    self.validator.validate_action(prev, action)
                except Exception:
                    total -= 50.0
                    action = None
                curr = env.step(action)
                total += self.reward_fn.calculate_step_reward(prev, curr)
                prev = curr
        return total

    def _evaluate(self, weights, bias, eval_seed: int, starts) -> float:
        if not starts:
            return 0.0
        return self._rollout(weights, bias, starts, eval_seed)

    # ------------------------------------------------------------------ #

    def train(
        self,
        historical_trajectories: List[List[KernelIntermediateRepresentation]],
        verbose: bool = True,
    ) -> LearnedLinearPolicy:
        """Optimise and return the best-holdout checkpoint as a policy object."""
        log = print if verbose else (lambda *a, **k: None)
        flat = [t for t in historical_trajectories if len(t) >= 2]
        if not flat:
            raise ValueError("OfflineTrainer needs at least one trajectory of length >= 2")

        train_starts, val_states = [], []
        for traj in flat:
            cut = max(1, int(len(traj) * (1.0 - self.val_fraction)))
            train_starts.extend(traj[:cut])
            val_states.extend(traj[cut:])
        # Cap evaluation cost deterministically.
        rng = random.Random(self.seed)
        if len(train_starts) > 24:
            train_starts = rng.sample(train_starts, 24)
        if len(val_states) > 16:
            val_states = rng.sample(val_states, 16)
        if not val_states:                      # degenerate tiny input
            val_states = train_starts[-1:]

        # Common random numbers: every candidate is scored on the SAME
        # workload realisation, so pairwise comparisons are paired and the
        # search is deterministic.
        TRAIN_SEED, VAL_SEED = 555, 556

        weights = [0.0] * 6
        bias = 6_000.0

        base_train = self._evaluate(weights, bias, TRAIN_SEED, train_starts)
        base_val = self._evaluate(weights, bias, VAL_SEED, val_states)
        log(f"[Trainer] Baseline reward  train={base_train:.4f}  holdout={base_val:.4f}")

        best = {
            "weights": list(weights), "bias": bias,
            "val": base_val, "round": -1,
        }
        report = TrainingReport(baseline_train=base_train, baseline_val=base_val)
        cur_train = base_train

        for rnd in range(self.rounds):
            w_s = max(self.w_step0 * (self.decay ** rnd), 1e-3)
            b_s = max(self.b_step0_us * (self.decay ** rnd), 25.0)

            # --- coordinate sweep over weights -------------------------
            for d in range(6):
                cand_best, r_best = weights[d], cur_train
                for delta in (w_s, -w_s, w_s / 4.0, -w_s / 4.0):
                    cand = weights[d] + delta
                    if abs(cand) > self.weight_clip:
                        continue
                    trial_w = list(weights)
                    trial_w[d] = cand
                    r = self._evaluate(trial_w, bias, TRAIN_SEED, train_starts)
                    if r > r_best:
                        cand_best, r_best = cand, r
                if cand_best != weights[d]:
                    weights[d] = cand_best
                    cur_train = r_best

            # --- bias sweep ---------------------------------------------
            cand_best, r_best = bias, cur_train
            for delta in (b_s, -b_s, b_s / 4.0, -b_s / 4.0):
                cand = min(23_000.0, max(500.0, bias + delta))
                if cand == bias:
                    continue
                r = self._evaluate(weights, cand, TRAIN_SEED, train_starts)
                if r > r_best:
                    cand_best, r_best = cand, r
            if cand_best != bias:
                bias = cand_best
                cur_train = r_best

            va = self._evaluate(weights, bias, VAL_SEED, val_states)
            report.train_curve.append(cur_train)
            report.val_curve.append(va)
            if va > best["val"]:
                best = {"weights": list(weights), "bias": bias, "val": va, "round": rnd}
            log(f"[Trainer] round {rnd:02d}  train={cur_train:.4f}  holdout={va:.4f}")

        weights, bias = best["weights"], best["bias"]
        final_train = self._evaluate(weights, bias, TRAIN_SEED, train_starts)
        report.final_train = final_train
        report.final_val = best["val"]
        report.best_round = best["round"]
        report.improved = best["val"] > base_val + 1e-9

        log(f"[Trainer] Done. best_round={report.best_round} "
            f"holdout {base_val:.4f} -> {best['val']:.4f}")
        log(f"[Trainer] {LearnedLinearPolicy(weights=weights, bias=bias)!r}")

        self.last_report = report
        self.last_dataset_hash = self.dataset_hash(historical_trajectories)
        return LearnedLinearPolicy(weights=weights, bias=bias)

    # ------------------------------------------------------------------ #

    def save_artifact(self, path, policy: LearnedLinearPolicy) -> dict:
        artifact = policy.to_artifact(
            dataset_sha256=getattr(self, "last_dataset_hash", ""),
            report=getattr(self, "last_report", TrainingReport()).summary(),
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        return artifact

    @staticmethod
    def load_artifact(path) -> LearnedLinearPolicy:
        with open(path, "r", encoding="utf-8") as f:
            return LearnedLinearPolicy.from_artifact(json.load(f))
