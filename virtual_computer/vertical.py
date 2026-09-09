"""Vertical slice v2: factorial causal path compiler->microarch->kernel.

v1 measured one configuration (N=8, balanced, gain 1.0) with degenerate
variance. v2 measures the full factorial:

  seeds S x sizes N x controls C (micro, vmicro in-order)
  seeds S x sizes N (micro, vmarch OoO — deep model, balanced)
  seeds S x workloads W x sizes N x coupling gains G x arms x H steps (kernel)

Every number comes from an exact oracle. New in v2:
- N-parameterized lowering (genuine timing variance; per-seed address
  offsets were measured to have zero effect and removed).
- vmarch OoO leg with cross-oracle (in-order vs OoO) sign agreement.
- coupling-gain sensitivity sweep {0.5, 1.0, 2.0} with verdict table.
- kernel transitions collected for learned-ensemble scoring
  (prediction-vs-oracle on the slice regime).
- cross-layer ridge surrogate scored by LOO skill (xsurrogate module).

Oracle units (reported as a vector, never summed):
compiler stub view / vmicro CPU run / vmarch OoO run / exact-sim step.
"""
from __future__ import annotations

from typing import Any, Sequence

from . import lowering
from .lowering import LOWERING_VERSION
from .trust import (check_cpu_conservation, check_deterministic,
                    check_functional_equivalence, check_kernel_invariants,
                    digest, paired_stats)
from .xsurrogate import kernel_features, loo_skill, micro_features

COUPLING_VERSION = "coupling-v2"
COUPLING_FORMULA = "D0 = clip(gain * mem_ops_retired / 32, 0.05, 1.0)"
COUPLING_REF = 32.0

SYSCTL_TIGHT_US = 2000
SYSCTL_LOOSE_US = 12000

DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5)
DEFAULT_NS = (4, 8, 12)
DEFAULT_CONTROLS = ("balanced", "streaming")
DEFAULT_GAINS = (0.5, 1.0, 2.0)


def _clip(v: float, lo: float = 0.05, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def demand_anchor(mem_ops: int, gain: float = 1.0) -> float:
    return _clip(gain * mem_ops / COUPLING_REF)


def _assemble_all(ns: Sequence[int]) -> dict[int, dict[str, Any]]:
    from vmicro.assembler import assemble

    out = {}
    for n in ns:
        per = {}
        for preset in lowering.PRESETS:
            prog = assemble(lowering.lower(preset, n))
            per[preset] = {"program": prog, "hist": lowering.op_histogram(prog)}
        out[n] = per
    return out


def _run_vmicro(program, mem: dict[int, int], control: str) -> tuple[dict, Any]:
    from vmicro.machine import ACTION_LIBRARY, CPU

    cpu = CPU(program, mem_init=dict(mem), control=ACTION_LIBRARY[control])
    return cpu.run(), cpu


def _run_vmarch(program, mem: dict[int, int]) -> tuple[dict, Any]:
    from vmarch.config import MicroarchConfig
    from vmarch.oracle import OracleSim

    arch_prog = lowering.translate_to_vmarch(program)
    sim = OracleSim(arch_prog, cfg=MicroarchConfig(), control="balanced",
                    mem_init=dict(mem))
    return sim.run(), sim.core


def run_vertical_slice(seeds: Sequence[int] = DEFAULT_SEEDS,
                       ns: Sequence[int] = DEFAULT_NS,
                       controls: Sequence[str] = DEFAULT_CONTROLS,
                       kernel_steps: int = 4,
                       gains: Sequence[float] = DEFAULT_GAINS,
                       ens_members: int = 2, ens_episodes: int = 4,
                       ens_steps: int = 8, ens_epochs: int = 10
                       ) -> dict[str, Any]:
    import random as _random

    from learned_kernel.policy.schemas import PolicyAction, SchedulerAction
    from learned_kernel.simulator.env import KernelSimulator, WorkloadProfile
    from virtual_compiler.environment import VirtualCompiler
    from virtual_compiler.oracle import StubOracle

    seeds, ns, controls, gains = list(seeds), list(ns), list(controls), list(gains)
    asm = _assemble_all(ns)
    prog_len = {n: {p: len(asm[n][p]["program"]) for p in lowering.PRESETS} for n in ns}
    mem_ops = {n: {p: asm[n][p]["hist"].get("mem", 0) for p in lowering.PRESETS} for n in ns}

    gate_problems: list[str] = []
    micro_oracle_runs = 0
    vmarch_runs = 0

    # structural claim of the lowering map, checked per N not assumed
    for n in ns:
        if not mem_ops[n]["O3"] < mem_ops[n]["O0"]:
            gate_problems.append(f"N={n} structural fail: {mem_ops[n]}")

    # -- compiler leg: preset branches + budgeted stub views (STUB evidence)
    oracle = StubOracle(runtimes={"-O0": 30.0, "-O3": 18.0}, budget=4)
    env = VirtualCompiler(source_text=lowering.c_source(ns[1] if len(ns) > 1 else ns[0]),
                          oracle=oracle)
    env.branch("o0")
    env.perturb("flags", "O0", branch="o0")
    env.branch("o3")
    env.perturb("flags", "O3", branch="o3")
    rec0 = env.validate("o0")
    rec3 = env.validate("o3")
    compiler_views = oracle.used

    # -- micro leg (vmicro): seeds x N x controls, paired per unit
    micro_units: list[dict] = []
    for s in seeds:
        for n in ns:
            mem, expected = lowering.vector_memory(s, n)
            for control in controls:
                cpu_res: dict[str, dict] = {}
                for preset in lowering.PRESETS:
                    res, cpu = _run_vmicro(asm[n][preset]["program"], mem, control)
                    micro_oracle_runs += 1
                    ok, det = check_cpu_conservation(res, prog_len[n][preset])
                    if not ok:
                        gate_problems.append(f"vmicro s={s} n={n} {control}/{preset}: {det['problems']}")
                    cpu_res[preset] = {
                        "cycles": res["cycles"], "instrs": res["instrs"],
                        "energy": res["energy"], "objective": res["objective"],
                        "status": res["completion_status"],
                        "acc": cpu.regs[3], "mem_out": cpu.mem[lowering.OUT_ADDR]}
                ok, _ = check_functional_equivalence(cpu_res["O0"], cpu_res["O3"], expected)
                if not ok:
                    gate_problems.append(f"s={s} n={n} {control} equivalence fail")
                micro_units.append({
                    "seed": s, "n": n, "control": control, "expected": expected,
                    "O0_obj": cpu_res["O0"]["objective"],
                    "O3_obj": cpu_res["O3"]["objective"],
                    "delta": cpu_res["O0"]["objective"] - cpu_res["O3"]["objective"]})

    # -- vmarch leg (OoO, balanced): seeds x N, paired + cross-oracle agreement
    vmarch_units: list[dict] = []
    agreements = 0
    for s in seeds:
        for n in ns:
            mem, expected = lowering.vector_memory(s, n)
            arch_res: dict[str, dict] = {}
            for preset in lowering.PRESETS:
                res, core = _run_vmarch(asm[n][preset]["program"], mem)
                vmarch_runs += 1
                ok, det = check_cpu_conservation(res, prog_len[n][preset])
                if not ok:
                    gate_problems.append(f"vmarch s={s} n={n}/{preset}: {det['problems']}")
                arch_res[preset] = {
                    "cycles": res["cycles"], "instrs": res["instrs"],
                    "energy": res["energy"], "objective": res["objective"],
                    "status": res["completion_status"],
                    "acc": core.regs[3], "mem_out": core.mem[lowering.OUT_ADDR]}
            ok, _ = check_functional_equivalence(arch_res["O0"], arch_res["O3"], expected)
            if not ok:
                gate_problems.append(f"vmarch s={s} n={n} equivalence fail")
            delta = arch_res["O0"]["objective"] - arch_res["O3"]["objective"]
            vmicro_delta = next(u["delta"] for u in micro_units
                                if u["seed"] == s and u["n"] == n
                                and u["control"] == "balanced")
            agree = (delta > 0) == (vmicro_delta > 0)
            agreements += int(agree)
            vmarch_units.append({"seed": s, "n": n,
                                 "O0_obj": arch_res["O0"]["objective"],
                                 "O3_obj": arch_res["O3"]["objective"],
                                 "delta": delta,
                                 "vmicro_balanced_delta": vmicro_delta,
                                 "sign_agrees_with_vmicro": agree})

    # -- determinism gate: rerun first unit twice, digest must match
    mem0, _ = lowering.vector_memory(seeds[0], ns[0])
    det_ok = True
    for preset in lowering.PRESETS:
        r1, c1 = _run_vmicro(asm[ns[0]][preset]["program"], mem0, controls[0])
        r2, c2 = _run_vmicro(asm[ns[0]][preset]["program"], mem0, controls[0])
        micro_oracle_runs += 2
        d = lambda r, c: {"cycles": r["cycles"], "instrs": r["instrs"],
                          "objective": r["objective"], "regs": c.regs[3],
                          "mem_out": c.mem[lowering.OUT_ADDR]}
        ok, _ = check_deterministic(d(r1, c1), d(r2, c2))
        det_ok = det_ok and ok
    if not det_ok:
        gate_problems.append("determinism rerun mismatch")

    # -- kernel leg: seeds x workloads x N x gains, paired sysctl arms
    from virtual_kernel import EnsembleDynamics
    from virtual_kernel.dynamics import kir_observables

    kernel_units: list[dict] = []  # one row per (seed, workload, n, gain)
    kernel_oracle_steps = 0
    transitions: list[tuple] = []  # (kir_prev, action, kir_next) at gain==1.0
    for s in seeds:
        for workload in lowering.PRESETS:
            for n in ns:
                for gain in gains:
                    d0 = demand_anchor(mem_ops[n][workload], gain)
                    cum: dict[str, float] = {}
                    for arm, target in (("tight", SYSCTL_TIGHT_US),
                                        ("loose", SYSCTL_LOOSE_US)):
                        sim = KernelSimulator(
                            seed=1000 + s,
                            workload=WorkloadProfile(rng=_random.Random(1000 + s),
                                                     demand=d0, regime=d0))
                        lats, sws = [], []
                        prev_sw = 0
                        for _ in range(kernel_steps):
                            act = PolicyAction(
                                policy_id="v", scheduler=SchedulerAction(
                                    target_latency_us=target))
                            kir_prev = sim.current_kir()
                            kir = sim.step(act)
                            kernel_oracle_steps += 1
                            lats.append(kir.scheduler.avg_latency_ms)
                            sws.append(float(kir.scheduler.total_context_switches))
                            if gain == gains[len(gains) // 2]:
                                transitions.append((kir_prev, act, kir,
                                                    prev_sw, sim.dt_s))
                            prev_sw = kir.scheduler.total_context_switches
                        ok, _ = check_kernel_invariants(lats, sws)
                        if not ok:
                            gate_problems.append(f"kernel s={s} {workload}/n={n}/g={gain}/{arm}")
                        cum[arm] = sum(lats)
                    kernel_units.append({
                        "seed": s, "workload": workload, "n": n, "gain": gain,
                        "demand": d0, "cum_tight": cum["tight"],
                        "cum_loose": cum["loose"],
                        "delta": cum["loose"] - cum["tight"]})

    # -- learned-ensemble scoring on the slice regime (toy-scale, labeled)
    ens = EnsembleDynamics.train(n_members=ens_members, n_episodes=ens_episodes,
                                 steps_per_episode=ens_steps, epochs=ens_epochs)
    ens_data = ens_episodes * ens_steps
    kernel_oracle_steps += ens_data
    import numpy as np
    errs, disags = [], []
    for kir_prev, act, kir_next, prev_sw, dt in transitions:
        y = kir_observables(kir_next, prev_sw, dt)
        outs = np.array([m.predict_obs(kir_prev, act) for m in ens.members])
        errs.append(float(np.mean((outs.mean(axis=0) - y) ** 2)))
        disags.append(float(outs.std(axis=0).mean()))
    from virtual_kernel.metrics import calibration as _cal
    ens_score = {"n": len(errs), "mse": float(np.mean(errs)) if errs else None,
                 "mean_disagreement": float(np.mean(disags)) if disags else None,
                 "calibration": _cal(disags, errs) if errs else None,
                 "train_transitions": ens_data,
                 "evidence": "toy-scale ensemble scored on slice regime (probe)"}

    # -- statistics: pooled + stratified + sensitivity
    import statistics as _st

    micro_deltas = [u["delta"] for u in micro_units]
    micro_stats = paired_stats(micro_deltas, baseline=_st.median(
        u["O0_obj"] for u in micro_units))
    micro_by_control = {c: _st.median(u["delta"] for u in micro_units if u["control"] == c)
                        for c in controls}
    micro_by_n = {n: _st.median(u["delta"] for u in micro_units if u["n"] == n)
                  for n in ns}
    vmarch_deltas = [u["delta"] for u in vmarch_units]
    vmarch_stats = paired_stats(vmarch_deltas, baseline=_st.median(
        u["O0_obj"] for u in vmarch_units))

    kernel_primary = [u for u in kernel_units if u["gain"] == gains[len(gains) // 2]]
    kernel_stats = {}
    for w in lowering.PRESETS:
        dd = [u["delta"] for u in kernel_primary if u["workload"] == w]
        kernel_stats[w] = paired_stats(dd, baseline=_st.median(
            u["cum_loose"] for u in kernel_primary if u["workload"] == w))
    sensitivity = {w: {g: paired_stats(
        [u["delta"] for u in kernel_units if u["workload"] == w and u["gain"] == g],
        baseline=_st.median(u["cum_loose"] for u in kernel_units
                            if u["workload"] == w and u["gain"] == g))["verdict"]
        for g in gains} for w in lowering.PRESETS}

    # -- cross-layer surrogate: LOO skill on both tables
    X_micro = [micro_features(u["n"], mem_ops[u["n"]]["O0"], mem_ops[u["n"]]["O3"],
                              u["control"]) for u in micro_units]
    xs_micro = loo_skill(X_micro, micro_deltas)
    X_kern = [kernel_features(u["demand"], u["workload"], u["n"], u["gain"],
                              kernel_steps) for u in kernel_units]
    xs_kernel = loo_skill(X_kern, [u["delta"] for u in kernel_units])

    gates = {
        "lowering_structural_mem_O3_lt_O0": not any("structural fail" in p for p in gate_problems),
        "functional_equivalence_all": not any("equivalence fail" in p for p in gate_problems),
        "cpu_conservation_all_runs": not any("conservation" in p for p in gate_problems),
        "vmarch_equivalence_and_conservation": not any("vmarch" in p for p in gate_problems),
        "kernel_invariants_all_trajs": not any("kernel s=" in p for p in gate_problems),
        "determinism": bool(det_ok),
    }
    budget = {"compiler_stub_views": compiler_views,
              "micro_cpu_runs": micro_oracle_runs,
              "vmarch_runs": vmarch_runs,
              "kernel_sim_steps": kernel_oracle_steps}
    report = {
        "versions": {"lowering": LOWERING_VERSION,
                     "lowering_digest": lowering.lowering_digest(tuple(ns)),
                     "coupling": COUPLING_VERSION,
                     "coupling_formula": COUPLING_FORMULA},
        "config": {"seeds": seeds, "ns": ns, "controls": controls,
                   "kernel_steps": kernel_steps, "gains": gains,
                   "sysctl_us": {"tight": SYSCTL_TIGHT_US, "loose": SYSCTL_LOOSE_US}},
        "compiler": {"stub_runtimes_ms": {"O0": rec0.runtime_ms, "O3": rec3.runtime_ms},
                     "evidence": "STUB (deterministic map, not gcc; see gcc_leg)"},
        "micro": {"units": micro_units, "stats": micro_stats,
                  "median_by_control": micro_by_control,
                  "median_by_n": micro_by_n,
                  "mem_ops": mem_ops},
        "vmarch": {"units": vmarch_units, "stats": vmarch_stats,
                   "sign_agreement_rate": agreements / max(len(vmarch_units), 1)},
        "kernel": {"units": kernel_units, "primary_gain": gains[len(gains) // 2],
                   "stats": kernel_stats, "sensitivity_verdicts": sensitivity},
        "ensemble": ens_score,
        "xsurrogate": {"micro_loo": {k: xs_micro[k] for k in
                                     ("n", "mae_model", "mae_const", "skill", "r2")},
                       "kernel_loo": {k: xs_kernel[k] for k in
                                      ("n", "mae_model", "mae_const", "skill", "r2")},
                       "evidence": "LOO screening (suggestive, needs held-out confirmation)"},
        "gates": gates,
        "gate_problems": gate_problems,
        "budget": budget,
    }
    report["digest"] = digest({k: report[k] for k in
                               ("versions", "config", "micro", "vmarch",
                                "kernel", "gates")})
    return report
