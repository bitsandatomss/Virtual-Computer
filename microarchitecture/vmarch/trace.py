"""Per-line memory event trace: summarize, compare, attribute (T3).

The trace answers "where did the cycles go" at line granularity: every
L1/L2/DRAM/MSHR/prefetch event is stamped with (cycle, kind, line, info).
`attribute_delta` decomposes a wall-clock delta between two runs of the same
program under different bundles into counter categories plus an explicit
overlap residual — the residual is reported, not hidden, because OoO overlap
means categories need not sum to the wall delta.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path


def summarize(events: list) -> dict:
    kinds = collections.Counter(e[1] for e in events)
    per_line = collections.Counter((e[1], e[2]) for e in events)
    dram = [e for e in events if e[1] == "dram"]
    return {"n_events": len(events), "by_kind": dict(kinds),
            "dram_row_hits": sum("rowhit" in e[3] for e in dram),
            "dram_row_misses": sum("rowmiss" in e[3] for e in dram),
            "hot_lines": per_line.most_common(5)}


def compare(a_events: list, b_events: list) -> dict:
    sa, sb = summarize(a_events), summarize(b_events)
    keys = sorted(set(sa["by_kind"]) | set(sb["by_kind"]))
    return {k: {"a": sa["by_kind"].get(k, 0), "b": sb["by_kind"].get(k, 0),
                "delta_b_minus_a": sb["by_kind"].get(k, 0) - sa["by_kind"].get(k, 0)}
            for k in keys}


def attribute_delta(res_a: dict, core_a, res_b: dict, core_b) -> dict:
    """Decompose wall-clock delta into stall-counter categories (T3 exit).

    Returns category deltas and the overlap residual:
    residual = wall_delta - sum(modeled categories). A large residual means
    overlap hides the effect and the attribution is incomplete — reported,
    not rounded away.
    """
    cats = {}
    for k in ("icache_stall", "fetch_starve", "dep_stall", "mshr_stall",
              "flush_bubbles", "commit_idle", "rob_full", "rs_full", "lsq_full"):
        cats[k] = core_b.st[k] - core_a.st[k]
    cats["mispredict_extra"] = (core_b.predictor.mispredictions -
                                core_a.predictor.mispredictions) * 2
    wall = res_b["cycles"] - res_a["cycles"]
    modeled = (cats["icache_stall"] + cats["fetch_starve"] + cats["dep_stall"] +
               cats["mshr_stall"] + cats["flush_bubbles"])
    return {"wall_delta": wall, "categories": cats, "modeled_sum": modeled,
            "overlap_residual": wall - modeled,
            "residual_share": abs(wall - modeled) / max(abs(wall), 1)}


def save(path: str | Path, events: list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(events) + "\n")


def load(path: str | Path) -> list:
    return [tuple(e) for e in json.loads(Path(path).read_text())]
