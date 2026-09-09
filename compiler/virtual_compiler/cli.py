"""CLI for the virtual compiler: screen, swarm, benchmark, campaign, audit.

Each subcommand is one rung of the vision: single screens are the inner
step, the campaign is the experimental engine (T15), the benchmark is
the killer evaluation (T2), and audit answers T5.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import run_swarm
from .analysis import ablation_mcts_vs_beam, audit_surrogate, lopo_from_dataset
from .benchmark import ARMS, run_killer_benchmark
from .campaign import run_campaign
from .environment import FLAG_ACTIONS, VirtualCompiler
from .evidence import EvidenceLog
from .features import SCHEMA as FEATURE_SCHEMA
from .search import STRATEGIES, gain_per_oracle, run_strategy
from .surrogate import Surrogate

DEMO_SOURCE = """\
#include <stdio.h>
long slow_search(long *a, long n, long key) {
    long found = -1;
    for (long i = 0; i < n; i++) {
        if (a[i] == key) found = i;
    }
    return found;
}
int main(void) {
    long a[1000];
    for (long i = 0; i < 1000; i++) a[i] = i * 2;
    volatile long s = 0;
    for (long r = 0; r < 200; r++) s += slow_search(a, 1000, 1998);
    printf("%ld\\n", s);
    return 0;
}
"""


def _load_source(args) -> str:
    if getattr(args, "source", None):
        return Path(args.source).read_text(encoding="utf-8",
                                           errors="replace")
    return DEMO_SOURCE


def _make_env(source: str, args, evidence=None,
              keep_dir: str | None = None) -> VirtualCompiler:
    from .oracle import Oracle
    oracle = Oracle(
        compiler=getattr(args, "compiler", "gcc"),
        test_cmd=getattr(args, "test_cmd", None),
        benchmark_runs=getattr(args, "runs", 3),
        budget=getattr(args, "budget", 8),
        evidence=evidence or EvidenceLog(getattr(args, "evidence", None)),
        keep_dir=keep_dir,
    )
    return VirtualCompiler(
        source,
        oracle_budget=oracle.budget,
        compiler=getattr(args, "compiler", "gcc"),
        evidence=evidence or oracle.evidence,
        oracle=oracle,
        benchmark_runs=getattr(args, "runs", 3))


def _emit(args, result: dict) -> None:
    print(json.dumps(result, indent=2))
    if getattr(args, "emit", None):
        Path(args.emit).write_text(json.dumps(result, indent=2),
                                   encoding="utf-8")


def cmd_screen(args) -> int:
    env = _make_env(_load_source(args), args,
                    keep_dir=getattr(args, "keep_dir", None))
    try:
        env.validate("root")
        baseline = env.branches["root"].runtime_ms
    except (RuntimeError, OSError) as exc:
        print(f"oracle unavailable: {exc}", file=sys.stderr)
        baseline = None
    res = run_strategy(env, args.strategy, top_k=args.top_k,
                       objective=args.objective, seed=args.seed)
    if args.objective == "size":
        # size is deterministic: report byte-deltas, not wall-clock.
        base_sz = env.branches["root"].binary_size
        best_sz = env.branches[res.best_branch].binary_size
        gain = (None if not base_sz or not best_sz
                else (base_sz - best_sz) / base_sz)
        metric = ("size_gain_per_oracle", gain_per_oracle(
            float(base_sz) if base_sz else None,
            float(best_sz) if best_sz else None, res.oracle_calls))
        extra = {"baseline_bytes": base_sz, "best_bytes": best_sz,
                 "size_gain": gain}
    else:
        baseline = (env.branches["root"].runtime_ms if "root" in env.branches
                    else None)
        metric = ("gain_per_oracle", gain_per_oracle(
            baseline, res.best_runtime_ms, res.oracle_calls))
        extra = {"baseline_runtime_ms": baseline}
    _emit(args, {
        "mode": "screen", "strategy": args.strategy,
        "best_branch": res.best_branch,
        "best_runtime_ms": res.best_runtime_ms,
        metric[0]: metric[1],
        "oracle_used": env.oracle.used,
        "virtual_calls": res.virtual_calls,
        "evaluated": res.evaluated,
        "branches": sorted(env.branches), **extra})
    return 0


def cmd_swarm(args) -> int:
    env = _make_env(_load_source(args), args,
                    keep_dir=getattr(args, "keep_dir", None))
    try:
        env.validate("root")
        baseline = env.branches["root"].runtime_ms
    except (RuntimeError, OSError) as exc:
        print(f"oracle unavailable: {exc}", file=sys.stderr)
        baseline = None
    out = run_swarm(env, oracle_rounds=args.rounds, top_k=args.top_k,
                    rule=args.rule)
    best_rt = env.branches[env.current].runtime_ms
    _emit(args, {
        "mode": "swarm", "best_branch": env.current,
        "best_runtime_ms": best_rt, "baseline_runtime_ms": baseline,
        "gain_per_oracle": gain_per_oracle(
            baseline, best_rt, env.oracle.used),
        "oracle_used": env.oracle.used,
        "consensus": out.get("consensus"),
        "log": [vars(v) for v in out["log"]]})
    return 0


def cmd_campaign(args) -> int:
    env = _make_env(_load_source(args), args,
                    keep_dir=getattr(args, "keep_dir", None))
    try:
        env.validate("root")
    except (RuntimeError, OSError) as exc:
        print(f"oracle unavailable: {exc}", file=sys.stderr)
        return 1
    out = run_campaign(env, rounds=args.rounds, fanout=args.fanout,
                       physics_k=args.top_k, rule=args.rule)
    out["mode"] = "campaign"
    _emit(args, out)
    return 0


def cmd_benchmark(args) -> int:
    wdir = Path(args.workloads) if args.workloads else None
    programs: dict[str, str] = {}
    if wdir is not None and wdir.is_dir():
        for f in sorted(wdir.glob("*.c")):
            programs[f.stem] = f.read_text(encoding="utf-8",
                                           errors="replace")
    if not programs:
        programs = {"demo": _load_source(args)}

    def make_env(source: str) -> VirtualCompiler:
        return VirtualCompiler(source, oracle_budget=args.budget,
                               compiler=args.compiler,
                               benchmark_runs=args.runs)

    report = run_killer_benchmark(
        make_env, programs,
        arms=tuple(args.arms) if args.arms else ARMS,
        budget_per_program=args.budget, top_k=args.top_k,
        objective=args.objective, seed=args.seed,
        seeds=getattr(args, "seeds", 1),
        verify_winners=getattr(args, "verify_winners", False),
        compiler=args.compiler)
    _emit(args, report.to_dict())
    return 0


def cmd_collect(args) -> int:
    """Measure workloads × configs into a dataset for LOPO (CRITIQUE G8).

    Each row: program, flags, features (static + observed IR telemetry),
    outcome (build_ok, runtime, size). Costs programs × configs oracle
    builds — data collection is the honest price of a transfer claim.
    """
    import json as _json

    from .features import extract_features
    from .oracle import Oracle
    from .state import CompilationState

    wdir = Path(args.workloads) if args.workloads else None
    programs: dict[str, str] = {}
    if wdir is not None and wdir.is_dir():
        for f in sorted(wdir.glob("*.c")):
            programs[f.stem] = f.read_text(encoding="utf-8",
                                           errors="replace")
    if not programs:
        programs = {"demo": _load_source(args)}
    configs: list[tuple[str, ...]] = [v for v in FLAG_ACTIONS.values()]
    if getattr(args, "full_space", False):
        from .environment import FLAG_TUNE_POOL
        for flag in FLAG_TUNE_POOL:
            configs.append(("-O2", "-" + flag))
            core = flag[1:] if flag.startswith("f") else flag
            configs.append(("-O2", "-fno-" + core))
    oracle = Oracle(compiler=args.compiler, benchmark_runs=args.runs,
                    budget=10 ** 9)
    out = Path(args.emit) if args.emit else None
    if out is None:
        print("--emit <dataset.jsonl> is required for collect",
              file=sys.stderr)
        return 2
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        def _row(prog: str, src_text: str, flags: tuple[str, ...],
                 xform: str | None) -> None:
            nonlocal n
            st = CompilationState(src_text, flags=tuple(flags))
            try:
                res = oracle.measure(st)
            except (RuntimeError, OSError) as exc:
                print(f"collect failed {prog} {flags} xform={xform}: {exc}",
                      file=sys.stderr)
                return
            feats = extract_features(src_text, tuple(flags))
            feats.update(res.telemetry)
            fh.write(_json.dumps({
                "schema": Surrogate.DATASET_SCHEMA,
                "feature_schema": FEATURE_SCHEMA,
                "program": prog,
                "flags": list(flags),
                "xform": xform,
                "features": feats,
                "outcome": {
                    "build_ok": res.build_ok,
                    "runtime_ms": res.runtime_ms,
                    "binary_size": res.binary_size,
                    "samples": res.run_samples,
                },
            }) + "\n")
            n += 1
        for prog, source in programs.items():
            for flags in configs:
                _row(prog, source, flags, None)
            if getattr(args, "xforms", False):
                from .transforms import TRANSFORMS, apply_transform
                for name in TRANSFORMS:
                    try:
                        new_src = apply_transform(source, name)
                    except ValueError:
                        continue
                    if new_src is None:
                        continue
                    for flags in (("-O0",), ("-O2",), ("-O3",)):
                        _row(prog, new_src, flags, name)
    print(json.dumps({"mode": "collect", "rows": n,
                      "dataset": str(out)}, indent=2))
    if getattr(args, "lopo", False):
        print(json.dumps(lopo_from_dataset(str(out)), indent=2))
    _ = STRATEGIES
    return 0


def cmd_levels(args) -> int:
    """Grade the system L1-L5 on a collected dataset (VISION N7)."""
    import json as _json

    from .levels import grade_dataset
    ds = getattr(args, "dataset", None)
    if not ds:
        print("--dataset <dataset.jsonl> is required for levels",
              file=sys.stderr)
        return 2
    records = []
    with open(ds, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(_json.loads(line))
    result = {"mode": "levels", "dataset": ds,
              **grade_dataset(records, seed=getattr(args, "seed", 0))}
    _emit(args, result)
    return 0


def cmd_audit(args) -> int:
    env = _make_env(_load_source(args), args,
                    keep_dir=getattr(args, "keep_dir", None))
    try:
        env.validate("root")
    except (RuntimeError, OSError) as exc:
        print(f"oracle unavailable: {exc}", file=sys.stderr)
        return 1
    run_strategy(env, "beam", top_k=args.top_k, seed=args.seed)
    result = {
        "mode": "audit",
        "surrogate": audit_surrogate(env),
        "ablation": ablation_mcts_vs_beam(
            lambda src: VirtualCompiler(
                src, oracle_budget=args.budget, compiler=args.compiler),
            env.branches["root"].source_text, top_k=args.top_k,
            seed=args.seed),
        "oracle_used": env.oracle.used,
    }
    _emit(args, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="virtual-compiler",
                                description="Branchable surrogate over "
                                            "compilation (see docs/VISION.md).")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--source", help="C source file (default: demo)")
        sp.add_argument("--budget", type=int, default=8)
        sp.add_argument("--top-k", type=int, default=3)
        sp.add_argument("--runs", type=int, default=3)
        sp.add_argument("--compiler", default="gcc")
        sp.add_argument("--objective", choices=("runtime", "size"),
                        default="runtime")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--emit", help="write JSON result to this path")
        sp.add_argument("--evidence", help="append evidence JSONL here")
        sp.add_argument("--keep-dir",
                        help="keep built binaries in this directory")

    s = sub.add_parser("screen", help="budgeted virtual screening")
    s.add_argument("--strategy",
                   choices=("stock", "random", "surrogate", "beam", "mcts",
                            "swarm"),
                   default="beam")
    common(s)

    s = sub.add_parser("swarm", help="agent swarm over the branch DAG")
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--rule",
                   choices=("uncertainty", "expected-improvement",
                            "ucb", "info-gain"),
                   default="uncertainty")
    common(s)

    s = sub.add_parser("campaign", help="closed-loop experimental engine")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--fanout", type=int, default=8)
    s.add_argument("--rule",
                   choices=("uncertainty", "expected-improvement",
                            "ucb", "info-gain"),
                   default="expected-improvement")
    common(s)

    s = sub.add_parser("benchmark", help="killer benchmark across workloads")
    s.add_argument("--workloads", help="directory of *.c held-out programs")
    s.add_argument("--arms", nargs="*", default=list(ARMS))
    s.add_argument("--seeds", type=int, default=1,
                   help="repeat every (program, arm) cell (G6)")
    s.add_argument("--verify-winners", action="store_true",
                   help="paired + differential re-check of winners (G3)")
    common(s)

    s = sub.add_parser("audit", help="surrogate degradation studies (T5)")
    common(s)

    s = sub.add_parser("levels", help="grade the system L1-L5 (VISION N7)")
    s.add_argument("--dataset", required=True,
                   help="collected dataset JSONL (see collect)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--emit", help="write JSON result to this path")

    s = sub.add_parser("collect",
                       help="measure workloads x configs to a dataset "
                            "for LOPO transfer tests (G8)")
    s.add_argument("--workloads", help="directory of *.c programs")
    s.add_argument("--full-space", action="store_true",
                   help="include per-flag tune configs (slower)")
    s.add_argument("--xforms", action="store_true",
                   help="include source-transform variants at O0/O2/O3")
    s.add_argument("--lopo", action="store_true",
                   help="run leave-one-program-out test after collecting")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--compiler", default="gcc")
    s.add_argument("--source", help="single C file if --workloads missing")
    s.add_argument("--emit", required=True, help="output dataset path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "version", False):
        print(Surrogate.SCHEMA)
        return 0
    if args.cmd is None:
        # bare invocation stays a demo: one budgeted beam screen
        args.cmd = "screen"
        args.strategy = "beam"
        for attr, default in (("source", None), ("budget", 8),
                              ("top_k", 3), ("runs", 3),
                              ("compiler", "gcc"),
                              ("objective", "runtime"), ("seed", 0),
                              ("emit", None), ("evidence", None)):
            if not hasattr(args, attr):
                setattr(args, attr, default)
        return cmd_screen(args)
    return {"screen": cmd_screen, "swarm": cmd_swarm,
            "campaign": cmd_campaign, "benchmark": cmd_benchmark,
            "audit": cmd_audit, "collect": cmd_collect,
            "levels": cmd_levels}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
