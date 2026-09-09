"""End-to-end Virtual Computer demonstration (deep v2).

Runs the factorial causal vertical slice — C source -> flag presets ->
N-parameterized lowering -> paired exact-CPU execution (in-order +
OoO) -> telemetry-coupled paired kernel comparison with sensitivity
sweep -> learned-ensemble scoring -> cross-layer surrogate skill — then
grades the honest unified L1-L5 level and appends a bound manifest.
"""
from virtual_computer.computer import VirtualComputer
from virtual_computer.levels import certify_unified
from virtual_computer.provenance import UnifiedManifest, append_manifest


def main() -> None:
    print("=" * 64)
    print("VIRTUAL COMPUTER — compiler -> microarch -> kernel (v2 factorial)")
    print("=" * 64)
    vc = VirtualComputer()
    vc.branch("o3-tight", flags="O3", sysctl="tight", control="latency")

    print("\n[measure] running factorial vertical slice (all-exact oracles) ...")
    report = vc.run_pipeline()
    v = report["vertical"]
    mo = v["micro"]["mem_ops"]
    print(f"  lowering: {v['versions']['lowering']} {v['versions']['lowering_digest']} "
          f"mem_ops={{{', '.join(f'N={n}:O0={mo[n]['O0']}/O3={mo[n]['O3']}' for n in sorted(mo))}}}")
    ms = v["micro"]["stats"]
    print(f"  micro vmicro ({ms['n']} units): delta_O0-O3 median={ms['median']:.2f} "
          f"CI=[{ms['ci_low']:.2f},{ms['ci_high']:.2f}] verdict={ms['verdict']}")
    print(f"    by_n={ {k: round(x, 2) for k, x in v['micro']['median_by_n'].items()} } "
          f"by_control={ {k: round(x, 2) for k, x in v['micro']['median_by_control'].items()} }")
    vs = v["vmarch"]["stats"]
    print(f"  micro vmarch ({vs['n']} units): delta median={vs['median']:.2f} "
          f"CI=[{vs['ci_low']:.2f},{vs['ci_high']:.2f}] verdict={vs['verdict']} "
          f"sign-agreement={v['vmarch']['sign_agreement_rate']:.2f}")
    for w in ("O0", "O3"):
        s = v["kernel"]["stats"][w]
        print(f"  kernel[{w}]@gain1: delta_loose-tight median={s['median']:.3f} "
              f"CI=[{s['ci_low']:.3f},{s['ci_high']:.3f}] verdict={s['verdict']}")
    print(f"  kernel sensitivity (verdict by gain): {v['kernel']['sensitivity_verdicts']}")
    e = v["ensemble"]
    print(f"  ensemble on slice: n={e['n']} mse={e['mse']:.5f} "
          f"disagreement={e['mean_disagreement']:.5f} ({e['evidence']})")
    xs = v["xsurrogate"]
    print(f"  xsurrogate LOO skill: micro={xs['micro_loo']['skill']:.3f} "
          f"(R²={xs['micro_loo']['r2']:.3f}, n={xs['micro_loo']['n']}) "
          f"kernel={xs['kernel_loo']['skill']:.3f} "
          f"(R²={xs['kernel_loo']['r2']:.3f}, n={xs['kernel_loo']['n']})")
    print(f"  gates: {v['gates']}")
    if v["gate_problems"]:
        print(f"  gate problems: {v['gate_problems']}")
    print(f"  budget (vector): {report['budget']}")

    cert = certify_unified(v)
    print(f"\n[grade] unified: {cert['certified_name']}  rule={cert['rule']}")
    for b in cert["blockers"]:
        print(f"  blocker: {b}")
    rec = append_manifest("reports/unified_manifest.jsonl", UnifiedManifest(
        config={"demo": True, **{k: v["config"][k] for k in
                                 ("seeds", "ns", "controls", "kernel_steps", "gains")}},
        layers=["compiler", "kernel", "microarchitecture"],
        oracle_costs=dict(report["budget"]),
        results={"certified": cert["certified_name"],
                 "blockers": cert["blockers"], "vertical": v}))
    print(f"[provenance] hash={rec['hash']} parent={rec['parent']}")


if __name__ == "__main__":
    main()
