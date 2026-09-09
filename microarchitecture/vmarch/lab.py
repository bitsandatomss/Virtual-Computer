"""VMArch lab CLI: train / search / bench / agents / conserve / info.

Examples:
  python -m vmarch.lab train --family phase --seeds 8 --epochs 20 --out artifacts/vmarch_surrogate
  python -m vmarch.lab search --family mixed --seed 0
  python -m vmarch.lab bench --family phase --budget 4 --out artifacts/vmarch_benchmark.json
  python -m vmarch.lab agents --family stream --seed 0
  python -m vmarch.lab conserve
"""
from __future__ import annotations

import argparse
import json

from vmarch.config import MicroarchConfig


def _factory(cfg):
    from vmarch.virtual import VirtualMicroarchitecture

    def make(prog=None, mem=None, reg=None, seed=None, family=None):
        if prog is None:
            from vmarch.workloads import FAMILIES
            prog, mem, reg, _ = FAMILIES[family](seed)
        vm = VirtualMicroarchitecture(prog, cfg=cfg, mem_init=mem, reg_init=reg)
        return vm, 0
    return make


def cmd_train(args) -> None:
    from vmarch.surrogate import train_ensemble
    from vmarch.workloads import FAMILIES
    cfg = MicroarchConfig()
    make = _factory(cfg)
    examples = []
    for s in range(args.seeds):
        p, m, r, _ = FAMILIES[args.family]((args.seed0 or 0) + s)
        vm, root = make(p, m, r)
        examples += vm.collect_dataset(root, intervals=args.intervals,
                                       bundles=["balanced", "streaming", "irregular",
                                                "control", "latency"], interval=16)
    ens, info = train_ensemble(examples, seeds=(11, 22, 33),
                               epochs=args.epochs, batch=64)
    ens.save(args.out)
    print(json.dumps({"examples": len(examples), "info": info,
                      "saved_to": args.out}, indent=2, default=str))


def cmd_search(args) -> None:
    from vmarch.surrogate import train_ensemble
    from vmarch.workloads import FAMILIES
    cfg = MicroarchConfig()
    make = _factory(cfg)
    p, m, r, desc = FAMILIES[args.family](args.seed)
    print(f"workload: {desc} ({len(p)} instrs)")
    examples = []
    for s in range(4):
        p2, m2, r2, _ = FAMILIES[args.family](100 + s)
        vm, root = make(p2, m2, r2)
        examples += vm.collect_dataset(root, intervals=8,
                                       bundles=["balanced", "streaming", "irregular",
                                                "control", "latency"], interval=16)
    ens, _ = train_ensemble(examples, seeds=(11, 22), epochs=args.epochs)
    from vmarch.search import beam_search_plan
    vm, root = make(p, m, r)
    out = beam_search_plan(vm, root, ens, horizon=4, interval=16, beam=4,
                           oracle_budget=args.budget)
    base_cid = vm.branch(wid=root)
    base = vm.run_world(base_cid, ["balanced"] * 4, interval=16)
    print(f"baseline objective: {base['objective']:.1f}")
    print(f"search best:        {out['best']['objective']:.1f} plan={out['best']['plan']}")
    print(f"calls: {out['surrogate_calls']} surrogate, {out['oracle_calls']} oracle")


def cmd_bench(args) -> None:
    from vmarch.bench import PRIMARY_METRIC, held_out, save_artifact
    cfg = MicroarchConfig()
    make = _factory(cfg)

    def make3(prog, mem, reg):
        vm = make(prog, mem, reg)
        return vm
    train_seeds = range(0, args.train_seeds)
    test_seeds = range(10000, 10000 + args.test_seeds)
    ood = [f.strip() for f in (args.ood or "").split(",") if f.strip()]
    res = held_out(make3, args.family, train_seeds, test_seeds, budget=args.budget,
                   train_kwargs={"epochs": args.epochs}, cfg=cfg,
                   ood_families=ood or None, train_reps=args.train_reps)
    save_artifact(args.out, {"results": {args.family: res}})
    kb = res["killer_benchmark"]
    print(f"family={args.family} parity_mae={res['parity']['mae']:.2f} "
          f"skill_mean={res['skill']['skill_vs_mean']:+.2f} r2={res['skill']['r2']:+.2f} "
          f"hall={res['hallucination']['rate']:.2f} maturity={res['maturity']}")
    if res.get("parity_spread"):
        sp = res["parity_spread"]
        print(f"  train-rep spread (n={sp['reps']}): mae_std={sp['mae_std']:.3f} "
              f"skill_mean=[{sp['skill_mean_min']:+.2f},{sp['skill_mean_max']:+.2f}]")
    for k in ("best_fixed", "expert", "exact_search", "learned_alone",
              "surrogate_beam", "surrogate_mcts"):
        d = kb[f"delta_{k}"]
        tag = "  [PRIMARY]" if f"delta_{k}" == PRIMARY_METRIC else "  "
        print(f"{tag} delta_{k}: {d['mean']:+.1f} ± {d['ci95']:.1f} (n={d['n']})")
    for of, orr in (res.get("ood") or {}).items():
        print(f"  OOD train={args.family} test={of}: "
              f"mae={orr['parity']['mae']:.2f} "
              f"skill_mean={orr['skill']['skill_vs_mean']:+.2f} "
              f"r2={orr['skill']['r2']:+.2f}")


def cmd_agents(args) -> None:
    from vmarch import agents
    from vmarch.surrogate import evaluate_parity, train_ensemble
    from vmarch.workloads import FAMILIES
    cfg = MicroarchConfig()
    make = _factory(cfg)
    p, m, r, desc = FAMILIES[args.family](args.seed)
    vm, root = make(p, m, r)
    wid = vm.branch(label="lab", wid=root)
    vm.run_world(wid, ["balanced"] * 4, interval=16)
    examples = []
    for s in range(3):
        p2, m2, r2, _ = FAMILIES[args.family](200 + s)
        v2, r2id = make(p2, m2, r2)
        examples += v2.collect_dataset(r2id, intervals=6,
                                      bundles=["balanced", "streaming", "control"],
                                      interval=16)
    ens, _ = train_ensemble(examples, seeds=(11, 22), epochs=args.epochs)
    findings = [agents.bottleneck_analyst(vm, wid),
                agents.perturber(vm, wid, ens),
                agents.researcher(vm, wid, "streaming beats balanced on stream phases",
                                  ["balanced"] * 4, ["streaming"] * 4)]
    print(agents.synthesize_report(desc, findings))


def cmd_conserve(args) -> None:
    from vmarch.bench import conservation_suite
    cfg = MicroarchConfig()
    make = _factory(cfg)

    def make3(prog, mem, reg):
        return make(prog, mem, reg)
    res = conservation_suite(make3, seeds=[0, 1, 2])
    print(json.dumps(res, indent=2))
    if not res["ok"]:
        raise SystemExit(1)


def cmd_mimic(args) -> None:
    from vmarch.mimic import run_mimic_experiment
    cfg = MicroarchConfig()
    make = _factory(cfg)

    def make3(prog, mem, reg):
        return make(prog, mem, reg)
    res = run_mimic_experiment(make3, train_family=args.train_family,
                               test_family=args.test_family, epochs=args.epochs)
    print(json.dumps(res, indent=2, default=str))
    print(f"verdict: {res['verdict']} ({res.get('verdict_detail', '')})")


def cmd_info(args) -> None:
    for name, cfg in MicroarchConfig.design_points().items():
        print(f"{name:12s} digest={cfg.digest()} rob={cfg.rob_size} "
              f"issue={cfg.issue_width} pred={cfg.predictor}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="vmarch.lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--family", default="phase")
    t.add_argument("--seeds", type=int, default=8)
    t.add_argument("--seed0", type=int, default=0)
    t.add_argument("--intervals", type=int, default=10)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--out", default="artifacts/vmarch_surrogate")
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("search")
    s.add_argument("--family", default="mixed")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--budget", type=int, default=4)
    s.set_defaults(fn=cmd_search)
    b = sub.add_parser("bench")
    b.add_argument("--family", default="phase")
    b.add_argument("--train-seeds", type=int, default=8)
    b.add_argument("--test-seeds", type=int, default=8)
    b.add_argument("--train-reps", type=int, default=3)
    b.add_argument("--ood", default="",
                   help="comma-separated OOD test families, e.g. loop_branch,stream")
    b.add_argument("--epochs", type=int, default=20)
    b.add_argument("--budget", type=int, default=4)
    b.add_argument("--out", default="artifacts/vmarch_benchmark.json")
    b.set_defaults(fn=cmd_bench)
    a = sub.add_parser("agents")
    a.add_argument("--family", default="stream")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--epochs", type=int, default=20)
    a.set_defaults(fn=cmd_agents)
    c = sub.add_parser("conserve")
    c.set_defaults(fn=cmd_conserve)
    m = sub.add_parser("mimic")
    m.add_argument("--train-family", default="stream")
    m.add_argument("--test-family", default="loop_branch")
    m.add_argument("--epochs", type=int, default=30)
    m.set_defaults(fn=cmd_mimic)
    i = sub.add_parser("info")
    i.set_defaults(fn=cmd_info)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
