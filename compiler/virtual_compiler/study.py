"""Study aggregation: dataset stability, killer table, LOPO, levels.

Reads run artifacts (study.jsonl, killer.json, levels.json, audit.json)
and prints the paper tables. All estimates prefer the guarded median of
per-row samples where available (ANALYSIS P5).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def guarded_median(xs: list[float]) -> tuple[float, bool]:
    xs = sorted(xs)
    n = len(xs)
    if n < 3:
        return (xs[n // 2] if xs else 0.0), False
    inner = xs[1:-1]
    med = statistics.median(inner)
    dev = max(abs(x - med) / med for x in inner) if med else 0.0
    return med, dev <= 0.05


def row_estimate(row: dict) -> tuple[float | None, str]:
    s = row["outcome"].get("samples") or []
    if len(s) >= 3:
        med, _ = guarded_median([float(x) for x in s])
        return med, "guarded"
    return row["outcome"]["runtime_ms"], "median"


def stability(path: str) -> dict:
    rows = [json.loads(line) for line in open(path)]
    spreads = []
    for r in rows:
        s = r["outcome"].get("samples") or []
        m = r["outcome"]["runtime_ms"]
        if len(s) >= 3 and m:
            spreads.append((max(s) - min(s)) / m)
    spreads.sort()
    n = len(spreads)
    return {
        "rows": len(rows),
        "with_samples": n,
        "spread_p50": round(statistics.median(spreads), 3),
        "spread_p90": round(spreads[int(n * 0.9)], 3),
        "frac_gt10": round(sum(1 for x in spreads if x > 0.10) / n, 3),
    }


def killer_table(path: str) -> dict:
    rep = json.load(open(path))
    cells = defaultdict(list)   # arm -> [gain_per_oracle]
    gains = defaultdict(list)   # arm -> [gain]
    verdicts = defaultdict(lambda: defaultdict(int))
    winners = defaultdict(int)
    for p in rep["programs"]:
        winners[p.get("winner")] += 1
        for a in p["arms"]:
            cells[a["arm"]].append(a.get("gain_per_oracle") or 0.0)
            gains[a["arm"]].append(a.get("gain") or 0.0)
            verdicts[a["arm"]][a.get("verdict")] += 1
    out = {}
    for arm, xs in sorted(cells.items()):
        xs = sorted(xs)
        n = len(xs)
        out[arm] = {
            "n": n,
            "mean_gain_per_oracle": round(statistics.mean(xs), 4),
            "median_gain_per_oracle": round(statistics.median(xs), 4),
            "min": round(xs[0], 4), "max": round(xs[-1], 4),
            "mean_gain": round(statistics.mean(gains[arm]), 4),
            "verdicts": dict(verdicts[arm]),
        }
    return {"arms": out, "winners": dict(winners),
            "worst_case_gain": rep.get("worst_case_gain"),
            "elapsed_s": round(rep.get("elapsed_s", 0))}


def degradation(path: str, fracs: tuple = (0.25, 0.5, 0.75, 1.0),
                seed: int = 0) -> dict:
    """P4: how much training data before cross-program ranking works?

    Seeded global subsample at each fraction, LOPO pairwise accuracy.
    Programs falling below 2 test rows drop out (reported in n_prog).
    """
    import random
    from .analysis import lopo
    rows = [json.loads(line) for line in open(path)]
    out = {}
    for f in fracs:
        rng = random.Random(seed)
        sub = [r for r in rows if rng.random() < f]
        res = lopo(sub)
        progs = res.get("programs", {})
        accs = [v.get("pairwise_accuracy") for v in progs.values()
                if isinstance(v, dict)
                and v.get("pairwise_accuracy") is not None]
        out[str(f)] = {
            "n_rows": len(sub),
            "n_prog": len(accs),
            "mean_pairwise": (round(sum(accs) / len(accs), 4)
                              if accs else None),
        }
    return out


def main() -> None:
    import time
    ds = sys.argv[1] if len(sys.argv) > 1 else (
        r"C:\Users\aacer\AppData\Local\Temp\opencode\study.jsonl")
    kj = sys.argv[2] if len(sys.argv) > 2 else (
        r"C:\Users\aacer\AppData\Local\Temp\opencode\killer.json")
    print("== stability ==")
    print(json.dumps(stability(ds), indent=1))
    print("== killer ==")
    print(json.dumps(killer_table(kj), indent=1))
    print("== levels ==")
    from .levels import _l1, _l2, _l3, _l4, _l5
    records = [json.loads(line) for line in open(ds)]
    for name, fn in [("L1", _l1), ("L2", _l2), ("L3", _l3), ("L4", _l4)]:
        t = time.time()
        out = fn(records)
        keep = {k: v for k, v in out.items()
                if k in ("accuracy", "mean_pairwise_accuracy",
                         "worst_family", "lopo_verdict", "reason")}
        print(name, "pass=", out.get("pass"), f"{time.time() - t:.1f}s",
              json.dumps(keep)[:300])
    t = time.time()
    out = _l5(records)
    print("L5", "pass=", out.get("pass"), f"{time.time() - t:.1f}s",
          {k: v for k, v in out.items() if k in ("ei_ratio", "random_ratio",
                                                "programs")})
    t = time.time()
    print("== degradation ==")
    print(json.dumps(degradation(ds), indent=1), f"{time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
