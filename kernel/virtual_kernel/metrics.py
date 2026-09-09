"""metrics.py — scoring the surrogate like a scientist, not a demo.

Implements the measurement battery demanded by context.txt:
- "How far can it roll out before divergence?"   -> divergence curve/step
- "How much does it hallucinate?"                -> hallucination rate
- "Does MCTS compensate for model error?"        -> ranking fidelity (if the
  surrogate ranks candidates like the oracle, search works despite bias)
- "When does uncertainty become dangerous?"      -> calibration: does
  ensemble disagreement predict actual error?
- "Strength achieved per oracle interaction"     -> oracle efficiency
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def divergence_stats(curve: Sequence[float], threshold: float) -> Dict:
    """Summarize an open-loop error-vs-horizon curve."""
    c = [float(x) for x in curve]
    if not c:
        return {"horizon": 0, "final": 0.0, "mean": 0.0,
                "divergence_step": None, "auc": 0.0}
    div = next((i for i, v in enumerate(c) if v > threshold), None)
    return {"horizon": len(c), "final": c[-1], "mean": float(np.mean(c)),
            "divergence_step": div, "auc": float(np.trapezoid(c))}


def hallucination_rate(curves: List[Sequence[float]], threshold: float,
                       at_step: int = -1) -> float:
    """Fraction of rollouts whose error exceeds threshold at the probe step."""
    if not curves:
        return 0.0
    bad = 0
    for c in curves:
        c = list(c)
        v = c[at_step] if len(c) > abs(at_step) or at_step >= 0 else c[-1]
        if at_step >= 0:
            v = c[min(at_step, len(c) - 1)]
        if v > threshold:
            bad += 1
    return bad / len(curves)


def spearman_rank(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation in [-1, 1]; nan-safe. Core search-fidelity metric."""
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c)


def topk_hit_rate(pred_ranking: Sequence, oracle_rewards: Sequence[float],
                  k: int = 3) -> float:
    """Does the surrogate's top-k contain the oracle's true best?"""
    order_pred = np.argsort(np.argsort(list(pred_ranking)))
    best_true = int(np.argmax(list(oracle_rewards)))
    # pred_ranking is a score list aligned with candidates; top-k by score
    topk = set(np.argsort(list(pred_ranking))[-k:])
    return 1.0 if best_true in topk else 0.0


def calibration(uncertainties: Sequence[float],
                errors: Sequence[float]) -> Dict:
    """Does the surrogate know what it doesn't know?"""
    u = np.asarray(list(uncertainties), dtype=float)
    e = np.asarray(list(errors), dtype=float)
    corr = float(np.corrcoef(u, e)[0, 1]) if len(u) > 2 and np.std(u) > 0 else float("nan")
    # binned reliability: mean error per uncertainty tercile should increase
    order = np.argsort(u)
    thirds = np.array_split(e[order], 3)
    terciles = [float(np.mean(t)) if len(t) else float("nan") for t in thirds]
    monotonic = bool(terciles[0] <= terciles[1] <= terciles[2])
    return {"uncertainty_error_corr": corr, "error_terciles": terciles,
            "monotonic": monotonic}


def strength_per_oracle(candidate_oracle_reward: float,
                        best_oracle_reward: float,
                        oracle_calls: int) -> Dict:
    """The killer-benchmark efficiency: signed gap to the best-known reward.

    Convention: gap = (candidate - best) / |best|, so a candidate worse than
    the best yields a NEGATIVE value and the best itself scores 0.0.
    """
    denom = abs(best_oracle_reward) if best_oracle_reward != 0 else 1.0
    gap = (candidate_oracle_reward - best_oracle_reward) / denom
    return {"regret_frac": float(gap),
            "calls": int(oracle_calls),
            "reward_per_call": float(candidate_oracle_reward / max(oracle_calls, 1))}


def summarize_oracle_report(single_step_mse: float,
                            rollout_curves: List[List[float]],
                            threshold: float) -> Dict:
    aucs, finals, divs = [], [], []
    for c in rollout_curves:
        s = divergence_stats(c, threshold)
        aucs.append(s["auc"])
        finals.append(s["final"])
        divs.append(s["divergence_step"] if s["divergence_step"] is not None else len(c))
    return {"single_step_mse": float(single_step_mse),
            "mean_rollout_auc": float(np.mean(aucs)) if aucs else 0.0,
            "mean_final_drift": float(np.mean(finals)) if finals else 0.0,
            "mean_divergence_step": float(np.mean(divs)) if divs else 0.0,
            "hallucination_rate": hallucination_rate(rollout_curves, threshold),
            "propagation": propagation_law(rollout_curves),
            "n_rollouts": len(rollout_curves)}


def propagation_law(curves: List[Sequence[float]]) -> Dict:
    """Trust requirement #4: uncertainty propagation over long rollouts.

    context.txt (lines 595-597): "If uncertainty compounds over 1,000
    simulated steps, does the system know that?"

    Fits mean-error growth e(h) ~= e0 * r^h on the observed horizon and
    reports the growth rate r plus the doubling horizon (steps for error to
    double). A planner must refuse horizons far beyond the doubling point;
    the number is a property of THIS surrogate on THESE workloads, measured,
    not assumed.
    """
    arr = [np.asarray(c, dtype=float) for c in curves if len(c) > 1]
    if not arr:
        return {"growth_rate": float("nan"), "doubling_horizon": None,
                "r_squared": float("nan"), "n_curves": 0}
    h_max = min(len(c) for c in arr)
    mean_curve = np.mean([c[:h_max] for c in arr], axis=0) + 1e-12
    h = np.arange(len(mean_curve), dtype=float)
    log_e = np.log(mean_curve)
    # least squares: log e = log e0 + h * log r
    A = np.vstack([np.ones_like(h), h]).T
    coef, *_ = np.linalg.lstsq(A, log_e, rcond=None)
    log_r = float(coef[1])
    r = float(np.exp(log_r))
    pred = A @ coef
    ss_res = float(((log_e - pred) ** 2).sum())
    ss_tot = float(((log_e - log_e.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    doubling = float(np.log(2) / log_r) if log_r > 1e-9 else None
    return {"growth_rate": r, "doubling_horizon": doubling,
            "r_squared": float(r2), "n_curves": len(arr),
            "horizon_fit": h_max,
            "mean_final_error": float(mean_curve[-1])}
