"""Unified CLI: one entry point for the whole Virtual Computer."""
from __future__ import annotations

import argparse
import json

from .computer import VirtualComputer, layer_available
from .levels import certify_unified
from .provenance import UnifiedManifest, append_manifest


def cmd_info(_args) -> int:
    for layer in ("compiler", "kernel", "microarchitecture"):
        ok, reason = layer_available(layer)
        print(f"{layer:18s} {'OK' if ok else 'MISSING'}  {reason if not ok else ''}")
    return 0


def _emit_manifest(args, vc, report, cert) -> None:
    man = UnifiedManifest(
        config={"seeds": args.seeds, "ns": args.ns, "controls": args.controls,
                "gains": args.gains, "kernel_steps": args.kernel_steps,
                "gcc": args.gcc},
        layers=["compiler", "kernel", "microarchitecture"],
        oracle_costs=dict(report.get("budget", {})),
        results={"certified": cert["certified_name"],
                 "blockers": cert["blockers"],
                 "micro_verdict": report["vertical"]["micro"]["stats"]["verdict"],
                 "xsurrogate": report["vertical"]["xsurrogate"],
                 "gcc": report["vertical"].get("gcc", {"status": "not-run"})})
    rec = append_manifest(args.emit, man)
    print(f"manifest -> {args.emit}  hash={rec['hash']} parent={rec['parent']}")


def cmd_vertical(args) -> int:
    vc = VirtualComputer()
    vc.branch("o3-tight", flags="O3", sysctl="tight", control="latency")
    report = vc.run_pipeline(micro_budget=args.micro_budget,
                             compiler_budget=args.compiler_budget,
                             seeds=args.seeds, ns=args.ns,
                             controls=args.controls,
                             kernel_steps=args.kernel_steps,
                             gains=args.gains, gcc=args.gcc)
    v = report["vertical"]
    ms = v["micro"]["stats"]
    print(f"micro vmicro ({ms['n']}): delta_O0-O3 median={ms['median']:.2f} "
          f"CI=[{ms['ci_low']:.2f},{ms['ci_high']:.2f}] verdict={ms['verdict']}")
    print(f"  by_n={v['micro']['median_by_n']} by_control={v['micro']['median_by_control']}")
    vs = v["vmarch"]["stats"]
    print(f"micro vmarch ({vs['n']}): delta median={vs['median']:.2f} "
          f"CI=[{vs['ci_low']:.2f},{vs['ci_high']:.2f}] verdict={vs['verdict']} "
          f"sign-agreement={v['vmarch']['sign_agreement_rate']:.2f}")
    for w in ("O0", "O3"):
        s = v["kernel"]["stats"][w]
        print(f"kernel[{w}]: delta_loose-tight median={s['median']:.3f} "
              f"CI=[{s['ci_low']:.3f},{s['ci_high']:.3f}] verdict={s['verdict']}")
    print(f"sensitivity: {v['kernel']['sensitivity_verdicts']}")
    xs = v["xsurrogate"]
    print(f"xsurrogate LOO skill: micro={xs['micro_loo']['skill']:.3f} "
          f"kernel={xs['kernel_loo']['skill']:.3f}")
    if "gcc" in v:
        g = v["gcc"]
        if g.get("status") == "measured":
            print(f"gcc live: -O0 {g['median_O0_ms']:.2f}ms vs -O3 {g['median_O3_ms']:.2f}ms "
                  f"delta={g['delta_ms']:.2f} CI=[{g['ci_low']:.2f},{g['ci_high']:.2f}] "
                  f"verdict={g['verdict']} stable={g['stable']} "
                  f"output_match={g['output_match']} equiv={g['differential_equivalent']}")
        else:
            print(f"gcc leg: {g.get('status')}: {g.get('reason', g.get('error'))}")
    print(f"ensemble on slice: mse={v['ensemble']['mse']:.5f} "
          f"n={v['ensemble']['n']}")
    print(f"gates: {v['gates']}")
    if v["gate_problems"]:
        print(f"gate problems: {v['gate_problems']}")
    print(f"budget: {report['budget']}")
    cert = certify_unified(v)
    print(f"unified grade: {cert['certified_name']}")
    for b in cert["blockers"]:
        print(f"  blocker: {b}")
    if args.emit:
        _emit_manifest(args, vc, report, cert)
    return 0 if all(v["gates"].values()) else 1


def cmd_pipeline(args) -> int:
    # legacy alias: full pipeline (probes + vertical) with honest summary
    return cmd_vertical(args)


def cmd_certify(_args) -> int:
    cert = certify_unified(None)
    print(json.dumps(cert, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="virtual-computer",
                                 description="Unified Virtual Computer (compiler+kernel+microarch)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_info = sub.add_parser("info", help="probe layer availability")
    p_info.set_defaults(fn=cmd_info)
    for name, help_text in (
            ("vertical", "run causal vertical slice + honest certification"),
            ("pipeline", "legacy alias of vertical"),
            ("certify", "grade from committed layer evidence (no live run)")):
        p = sub.add_parser(name, help=help_text)
        if name == "certify":
            p.set_defaults(fn=cmd_certify)
            continue
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
        p.add_argument("--ns", type=int, nargs="+", default=[4, 8, 12])
        p.add_argument("--controls", nargs="+", default=["balanced", "streaming"])
        p.add_argument("--gains", type=float, nargs="+", default=[0.5, 1.0, 2.0])
        p.add_argument("--kernel-steps", type=int, default=4)
        p.add_argument("--gcc", action="store_true",
                       help="also run live-gcc -O0/-O3 leg (needs toolchain)")
        p.add_argument("--micro-budget", type=int, default=2)
        p.add_argument("--compiler-budget", type=int, default=2)
        p.add_argument("--emit", default="reports/unified_manifest.jsonl")
        p.set_defaults(fn=cmd_vertical if name == "vertical" else cmd_pipeline)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
