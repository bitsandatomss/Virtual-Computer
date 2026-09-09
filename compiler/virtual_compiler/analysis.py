"""Degradation studies: answer T5 with numbers, not adjectives.

context.txt:652-664 — "How much of the exact game can a surrogate
recover? How much does it hallucinate? How far can it roll out before
divergence? Does MCTS compensate for model error? When does model
uncertainty become strategically dangerous?"

Each function takes an explicit oracle callable so studies run against
the real toolchain *or* a stub with equal rigor.
"""
from __future__ import annotations

from typing import Any, Callable


def audit_surrogate(env, oracle_fn: Callable[[Any], Any] | None = None,
                    threshold: float = 0.7) -> dict:
    """Agreement, hallucination, and calibration of the surrogate.

    - *recover*: fraction of oracle-validated branches the surrogate
      ranked correctly pairwise (build_ok ordering).
    - *hallucinate*: fraction of high-confidence (ok_prob ≥ threshold)
      predictions contradicted by the oracle.
    - *calibration*: mean |ok_prob − empirical build_ok| over validated.
    """
    validated = [s for s in env.branches.values() if s.validated]
    if len(validated) < 2:
        return {"n": len(validated), "verdict": "insufficient-evidence"}
    if oracle_fn is not None:
        for st in validated:
            oracle_fn(st)
    correct = total = 0
    for i in range(len(validated)):
        for j in range(i + 1, len(validated)):
            a, b = validated[i], validated[j]
            if a.build_ok == b.build_ok:
                continue
            total += 1
            pa = env.surrogate.predict(a).ok_prob
            pb = env.surrogate.predict(b).ok_prob
            if (pa >= pb) == bool(a.build_ok):
                correct += 1
    confident = [s for s in validated
                 if env.surrogate.predict(s).ok_prob >= threshold]
    hallucinations = sum(
        1 for s in confident if not s.build_ok) if confident else 0
    cal = sum(abs(env.surrogate.predict(s).ok_prob
                  - (1.0 if s.build_ok else 0.0)) for s in validated
              ) / len(validated)
    return {
        "n": len(validated),
        "pairwise_recovery": correct / total if total else None,
        "hallucination_rate": (hallucinations / len(confident)
                               if confident else 0.0),
        "n_confident": len(confident),
        "calibration_error": cal,
        "verdict": ("trustworthy" if cal < 0.25 and
                    (hallucinations == 0) else "unreliable"),
    }


def rollout_divergence(env, base: str,
                       actions: list[tuple[str, str]]) -> list[dict]:
    """Error vs rollout depth: how far before the virtual world diverges.

    Compares surrogate runtime estimates along a trajectory against
    oracle-validated runtimes where branches happen to be validated;
    unvalidated depths report uncertainty instead of error (honest
    divergence accounting — no invented ground truth).
    """
    rows: list[dict] = []
    name = base
    for depth, (action, arg) in enumerate(actions, 1):
        try:
            name = env.perturb(action, arg, branch=name)
        except ValueError as exc:
            rows.append({"depth": depth, "error": str(exc)})
            break
        pred = env.surrogate.predict(env.branches[name])
        row: dict = {"depth": depth, "branch": name,
                     "predicted_runtime": pred.runtime_ms,
                     "uncertainty": pred.uncertainty}
        st = env.branches[name]
        if st.validated and st.runtime_ms and pred.runtime_ms:
            row["oracle_runtime"] = st.runtime_ms
            row["rel_error"] = abs(pred.runtime_ms - st.runtime_ms
                                   ) / max(st.runtime_ms, 1e-9)
        rows.append(row)
    return rows


def ablation_mcts_vs_beam(make_env, source: str, top_k: int = 2,
                          seed: int = 0) -> dict:
    """T5's "does MCTS compensate for model error?" as a number."""
    from .search import run_strategy

    out = {}
    for arm in ("beam", "mcts"):
        env = make_env(source)
        try:
            env.validate("root")
        except (RuntimeError, OSError):
            out[arm] = {"error": "oracle-unavailable"}
            continue
        res = run_strategy(env, arm, top_k=top_k, seed=seed)
        out[arm] = {"best_runtime_ms": res.best_runtime_ms,
                    "oracle_calls": res.oracle_calls,
                    "virtual_calls": res.virtual_calls}
    if "best_runtime_ms" in out.get("beam", {}) and \
            "best_runtime_ms" in out.get("mcts", {}):
        b, m = out["beam"]["best_runtime_ms"], out["mcts"]["best_runtime_ms"]
        out["delta"] = (None if b is None or m is None or b <= 0
                        else (b - m) / b)
        out["verdict"] = ("mcts-wins" if (out["delta"] or 0) > 0.02
                          else "within-noise")
    return out


def lopo(records: list[dict], k: int = 5) -> dict:
    """Leave-one-program-out generalization (CRITIQUE G1/G2/G8).

    Trains a fresh `Surrogate` on all programs but one, predicts the
    held-out program's runtimes, and compares against the mean-baseline
    (predict the train mean). Metrics use **per-program-normalized**
    error (G2): absolute ms do not transfer, relative effects might.

    Each record: ``{"program": str, "features": {...},
    "outcome": {"runtime_ms": float|None, "build_ok": bool}}``.
    Verdict ``beats-mean`` requires normalized MAE below baseline AND
    pairwise accuracy above 0.5 on every held-out program with ≥2
    measurable configs — the falsifiable bar from CRITIQUE §2.G1.
    """
    from .state import CompilationState
    from .surrogate import Surrogate

    by_prog: dict[str, list[dict]] = {}
    for rec in records:
        by_prog.setdefault(rec["program"], []).append(rec)
    progs = sorted(by_prog)
    if len(progs) < 2:
        return {"verdict": "insufficient-evidence", "programs": progs}
    per_prog: dict[str, dict] = {}
    for held in progs:
        train = [r for p, rs in by_prog.items() if p != held for r in rs]
        test = [r for r in by_prog[held]
                if r.get("outcome", {}).get("runtime_ms") is not None]
        if len(test) < 2:
            per_prog[held] = {"verdict": "insufficient-evidence",
                              "n_test": len(test)}
            continue
        surr = Surrogate()
        for r in train:
            out = r.get("outcome", {})
            st = CompilationState("", flags=tuple(r.get("flags", ("-O2",))))
            # query/train states carry the RECORDED features via the
            # telemetry overlay, so train and test rows live in the same
            # feature space (no empty-source mismatch).
            st.telemetry = dict(r.get("features", {}))
            st.build_ok = out.get("build_ok", True)
            st.runtime_ms = out.get("runtime_ms")
            sz = out.get("binary_size")
            st.binary_size = sz
            st.validated = True
            surr.observe(st)
        train_rts = [x for x in surr._rt if x is not None]
        train_mean = sum(train_rts) / max(1, len(train_rts))
        errs, base_errs, correct, total = [], [], 0, 0
        for r in test:
            actual = r["outcome"]["runtime_ms"]
            st = CompilationState("", flags=tuple(r.get("flags", ("-O2",))))
            st.telemetry = dict(r.get("features", {}))
            pred = surr.predict(st, k=k).runtime_ms
            pred = actual if pred is None else pred
            errs.append(abs(pred - actual) / max(actual, 1e-9))
            base_errs.append(abs(train_mean - actual) / max(actual, 1e-9))
        for i in range(len(test)):
            for j in range(i + 1, len(test)):
                ai = test[i]["outcome"]["runtime_ms"]
                aj = test[j]["outcome"]["runtime_ms"]
                if ai == aj:
                    continue
                total += 1
                si = CompilationState("", flags=tuple(
                    test[i].get("flags", ("-O2",))))
                si.telemetry = dict(test[i].get("features", {}))
                sj = CompilationState("", flags=tuple(
                    test[j].get("flags", ("-O2",))))
                sj.telemetry = dict(test[j].get("features", {}))
                pi = surr.predict(si, k=k).runtime_ms or 0.0
                pj = surr.predict(sj, k=k).runtime_ms or 0.0
                if (pi <= pj) == (ai <= aj):
                    correct += 1
        mae = sum(errs) / len(errs)
        bmae = sum(base_errs) / len(base_errs)
        acc = correct / total if total else None
        per_prog[held] = {
            "n_test": len(test), "mae_norm": mae,
            "baseline_mae_norm": bmae,
            "pairwise_accuracy": acc,
            "verdict": ("beats-mean" if mae < bmae and
                        (acc or 0) > 0.5 else "within-noise"),
        }
    overall = ("beats-mean" if per_prog and all(
        v.get("verdict") == "beats-mean" for v in per_prog.values())
        else "within-noise")
    return {"verdict": overall, "programs": per_prog}


def propagate(env, base: str, actions: list[tuple[str, str]],
              rho: float = 0.15) -> list[dict]:
    """Uncertainty propagation over a virtual rollout (VISION N3.4).

    Per-step uncertainty compounds: not knowing step 1 makes step 2
    less trustworthy even before measuring it —
    ``u_d = max(fresh_d, 1 - (1 - u_{d-1}) * (1 - rho))``.
    Where branches are oracle-validated, measured error is reported
    alongside; where not, the propagated number stands alone (honest:
    no invented ground truth).
    """
    rows: list[dict] = []
    name = base
    u_prev = env.uncertainty(base)
    for depth, (action, arg) in enumerate(actions, 1):
        try:
            name = env.perturb(action, arg, branch=name)
        except ValueError as exc:
            rows.append({"depth": depth, "error": str(exc)})
            break
        u_fresh = env.uncertainty(name)
        u_prop = max(u_fresh, 1.0 - (1.0 - u_prev) * (1.0 - rho))
        row: dict = {"depth": depth, "branch": name,
                     "predicted_runtime":
                         env.surrogate.predict(env.branches[name]).runtime_ms,
                     "uncertainty_fresh": u_fresh,
                     "uncertainty_propagated": u_prop}
        st = env.branches[name]
        if st.validated and st.runtime_ms:
            pred = row["predicted_runtime"]
            if pred:
                row["oracle_runtime"] = st.runtime_ms
                row["rel_error"] = abs(pred - st.runtime_ms) / max(
                    st.runtime_ms, 1e-9)
        rows.append(row)
        u_prev = u_prop
    return rows


def lopo_from_dataset(path: str) -> dict:
    """Run `lopo` over a collected dataset file."""
    import json as _json

    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(_json.loads(line))
    return lopo(records)
