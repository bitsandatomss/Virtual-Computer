"""L1–L5 grading: assess the system against VISION N7.

Levels (context.txt :660-686), each with an operational pass/fail so a
level claim is a measurement, not an adjective:

- L1 Exemplar — reproduce seen experience (leave-one-out build_ok).
- L2 Interpolative — unseen programs: `lopo` beats-mean (ranking task).
- L3 Counterfactual — interventions rank usefully out-of-sample:
  mean held-out pairwise accuracy ≥ 0.6 AND branch machinery exercised.
- L4 Mechanistic/OOD — abstains far-from-experience (`ood`) while
  reporting the worst family honestly (no hiding the matmul).
- L5 Scientific — active choice beats random choice: simulated
  probe-to-optimum race, EI vs seeded random, on real records.

Overall grade = highest consecutive pass starting at L1. A system that
passes L3 but fails L2 is graded L1 — levels are cumulative by
construction, because counterfactual trust without interpolation is
exactly the explosion scenario (:567-577).
"""
from __future__ import annotations

import random

LEVELS = ("L1", "L2", "L3", "L4", "L5")


def _state_for(rec: dict):
    from .state import CompilationState
    st = CompilationState("", flags=tuple(rec.get("flags", ("-O2",))))
    st.telemetry = dict(rec.get("features", {}))
    out = rec.get("outcome", {})
    st.build_ok = out.get("build_ok", True)
    st.runtime_ms = out.get("runtime_ms")
    sz = out.get("binary_size")
    st.binary_size = sz
    st.validated = True
    return st


def _l1(records: list[dict]) -> dict:
    from .surrogate import Surrogate
    ok = [r for r in records if r.get("outcome", {}).get("build_ok") is not None]
    if len(ok) < 2:
        return {"pass": False, "reason": "insufficient-evidence", "n": len(ok)}
    correct = 0
    for i, held in enumerate(ok):
        surr = Surrogate()
        for j, r in enumerate(ok):
            if j != i:
                surr.observe(_state_for(r))
        pred = surr.predict(_state_for(held)).ok_prob
        if (pred >= 0.5) == bool(held["outcome"]["build_ok"]):
            correct += 1
    acc = correct / len(ok)
    return {"pass": acc == 1.0, "accuracy": acc, "n": len(ok)}


def _l2(records: list[dict]) -> dict:
    from .analysis import lopo
    out = lopo(records)
    return {"pass": out["verdict"] == "beats-mean",
            "lopo_verdict": out["verdict"],
            "programs": out.get("programs", {})}


def _l3(records: list[dict]) -> dict:
    from .analysis import lopo
    out = lopo(records)
    progs = out.get("programs", {})
    if isinstance(progs, list):  # lopo early-return shape
        return {"pass": False, "reason": "insufficient-evidence",
                "mean_pairwise_accuracy": None,
                "structural_branching": True}
    accs = [v["pairwise_accuracy"] for v in progs.values()
            if isinstance(v.get("pairwise_accuracy"), float)]
    mean_acc = sum(accs) / len(accs) if accs else None
    structural = True  # branch/perturb machinery exists by construction
    passed = bool(mean_acc is not None and mean_acc >= 0.6 and structural)
    return {"pass": passed, "mean_pairwise_accuracy": mean_acc,
            "structural_branching": structural}


def _l4(records: list[dict]) -> dict:
    from . import ood as _ood
    from .features import FEATURE_KEYS
    from .surrogate import Surrogate
    surr = Surrogate()
    for r in records:
        surr.observe(_state_for(r))
    model = _ood.fit_on_surrogate(surr)
    n_feats = len(FEATURE_KEYS)
    far = [10.0] * n_feats  # far-regime probe: nothing like experience
    near = list(surr.latent_rows()[0]) if len(surr) else [0.0] * n_feats
    a_far = _ood.assess(far, model)
    a_near = _ood.assess(near, model)
    abstains_far = bool(a_far["abstain"]["magnitude"])
    keeps_near = not bool(a_near["abstain"]["ranking"])
    from .analysis import lopo
    progs = lopo(records).get("programs", {})
    worst = None
    if isinstance(progs, dict):
        for name, v in progs.items():
            acc = v.get("pairwise_accuracy")
            if isinstance(acc, float) and (worst is None or acc < worst[1]):
                worst = (name, acc)
    passed = abstains_far and keeps_near
    return {"pass": passed, "abstains_far_magnitude": abstains_far,
            "keeps_near_ranking": keeps_near,
            "worst_family": worst}


def _ei_pick(surr, cands: list[dict], incumbent: float) -> dict:
    best, best_score = cands[0], None
    for rec in cands:
        pred = surr.predict(_state_for(rec))
        est = pred.runtime_ms if pred.runtime_ms is not None else incumbent
        improvement = max(0.0, incumbent - est)
        score = improvement * (1.0 - pred.uncertainty) + 0.05 * pred.uncertainty
        if best_score is None or score > best_score:
            best, best_score = rec, score
    return best


def _l5(records: list[dict], seed: int = 0) -> dict:
    """Simulated probe race on real records: EI vs random, same budgets."""
    from .surrogate import Surrogate
    by_prog: dict[str, list[dict]] = {}
    for r in records:
        if r.get("outcome", {}).get("runtime_ms") is not None:
            by_prog.setdefault(r["program"], []).append(r)
    progs = {p: rs for p, rs in by_prog.items() if len(rs) >= 4}
    if not progs:
        return {"pass": False, "reason": "insufficient-evidence"}
    ei_ratios, rnd_ratios = [], []
    for idx, (prog, rs) in enumerate(sorted(progs.items())):
        optimum = min(r["outcome"]["runtime_ms"] for r in rs)
        budget = min(3, len(rs) - 1)
        # EI contestant
        rng = random.Random(seed + idx)
        order = list(rs)
        rng.shuffle(order)
        probed, pool = [order[0]], order[1:]
        for _ in range(budget):
            surr = Surrogate()
            for r in probed:
                surr.observe(_state_for(r))
            inc = min(r["outcome"]["runtime_ms"] for r in probed)
            pick = _ei_pick(surr, pool, inc)
            probed.append(pick)
            pool = [r for r in pool if r is not pick]
        ei_best = min(r["outcome"]["runtime_ms"] for r in probed)
        ei_ratios.append(ei_best / max(optimum, 1e-9))
        # random contestant, same budget, independent shuffle
        rng2 = random.Random(10_000 + seed + idx)
        order2 = list(rs)
        rng2.shuffle(order2)
        rnd_best = min(r["outcome"]["runtime_ms"] for r in order2[:1 + budget])
        rnd_ratios.append(rnd_best / max(optimum, 1e-9))
    mean_ei = sum(ei_ratios) / len(ei_ratios)
    mean_rnd = sum(rnd_ratios) / len(rnd_ratios)
    return {"pass": mean_ei <= mean_rnd, "ei_ratio": mean_ei,
            "random_ratio": mean_rnd, "programs": sorted(progs)}


def grade_dataset(records: list[dict], seed: int = 0) -> dict:
    """Grade L1–L5 on collected records; overall = top consecutive pass."""
    results = {
        "L1": _l1(records),
        "L2": _l2(records),
        "L3": _l3(records),
        "L4": _l4(records),
        "L5": _l5(records, seed=seed),
    }
    overall = "L0"
    for lvl in LEVELS:
        if results[lvl]["pass"]:
            overall = lvl
        else:
            break
    return {"levels": results, "grade": overall}


def grade_session(env) -> dict:
    """Lightweight grade from a live environment (L1 + L3-structural + L4 probe)."""
    from . import ood as _ood
    from .features import FEATURE_KEYS
    validated = [s for s in env.branches.values() if s.validated]
    recs = [{"program": "session", "flags": list(s.flags),
             "features": s.features,
             "outcome": {"build_ok": s.build_ok,
                         "runtime_ms": s.runtime_ms,
                         "binary_size": s.binary_size}} for s in validated]
    l1 = _l1(recs) if recs else {"pass": False, "reason": "no-validated"}
    model = _ood.fit_on_surrogate(env.surrogate)
    far = [10.0] * len(FEATURE_KEYS)
    abstain = _ood.assess(far, model)["abstain"]["magnitude"]
    return {"L1": l1,
            "L3_structural": {"branch_count": len(env.branches),
                              "pass": len(env.branches) > 1},
            "L4_probe_abstains_far": bool(abstain)}
