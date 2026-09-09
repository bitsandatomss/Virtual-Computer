"""Research-harness tests: stub oracle, strategies, benchmark, campaign.

No gcc required — `StubOracle` stands in for the toolchain with
deterministic, flag-dependent runtimes, so the *logic* of budgeting,
search, acquisition, and reporting is tested exactly.
"""
import json

from virtual_compiler import acquisition, analysis
from virtual_compiler.agents import Critic, run_swarm
from virtual_compiler.benchmark import run_killer_benchmark
from virtual_compiler.campaign import run_campaign
from virtual_compiler.environment import VirtualCompiler
from virtual_compiler.evidence import EvidenceLog, STAGES
from virtual_compiler.features import (
    FEATURE_KEYS,
    ir_from_telemetry,
)
from virtual_compiler.oracle import OracleBudgetExhausted, StubOracle
from virtual_compiler.search import (
    STRATEGIES,
    run_strategy,
    virtual_screen,
)
from virtual_compiler.state import CompilationState

RUNTIMES = {"-O0": 100.0, "-O1": 60.0, "-O2": 40.0, "-O3": 30.0,
            "-Os": 55.0, "-O2|-flto": 35.0, "-O3|-march=native": 28.0,
            "-O3|-funroll-loops": 32.0}

SRC = "int main(void){volatile long s=0;for(long i=0;i<100;i++)s+=i;return 0;}\n"


def _stub_env(budget=10):
    oracle = StubOracle(runtimes=RUNTIMES, budget=budget,
                        evidence=EvidenceLog())
    return VirtualCompiler(SRC, oracle_budget=budget, oracle=oracle,
                           evidence=EvidenceLog())


# -- oracle -------------------------------------------------------------
def test_stub_oracle_budget_and_response():
    env = _stub_env(budget=2)
    assert env.oracle.remaining == 2
    env.validate("root")  # -O2 -> 40.0
    assert env.branches["root"].runtime_ms == 40.0
    assert env.branches["root"].validated
    assert len(env.surrogate) == 1
    env.validate("root")
    try:
        env.validate("root")
    except RuntimeError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("expected budget exhaustion")
    try:
        env.oracle.measure(env.branches["root"])
    except OracleBudgetExhausted:
        pass
    else:
        raise AssertionError("expected OracleBudgetExhausted")


def test_oracle_result_distinguishes_flags():
    env = _stub_env()
    fast = env.perturb("flags", "O3", new_branch="fast")
    slow = env.perturb("flags", "O0", branch="root", new_branch="slow")
    env.validate(fast)
    env.validate(slow)
    assert env.branches[fast].runtime_ms == 30.0
    assert env.branches[slow].runtime_ms == 100.0
    assert env.compare(fast, slow)["winner"] == "fast"


# -- evidence -----------------------------------------------------------
def test_evidence_stages():
    log = EvidenceLog()
    for stage in STAGES:
        log.emit(stage, f"test-{stage}", branch="root")
    assert len(log) == len(STAGES)
    assert [r["stage"] for r in log.records] == list(STAGES)
    assert all(r["schema"].startswith("virtual-compiler.") for r in log.records)


def test_evidence_file_roundtrip():
    import tempfile
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory(prefix="vcc-ev-") as td:
        p = _Path(td) / "ev.jsonl"
        log = EvidenceLog(p)
        log.emit("observe", "x", v=1)
        rows = EvidenceLog.read(p)
    assert len(rows) == 1 and rows[0]["event"] == "x"


# -- persistence ---------------------------------------------------------
def test_session_save_load():
    import tempfile
    from pathlib import Path as _Path
    env = _stub_env()
    env.validate("root")
    env.perturb("flags", "O3", new_branch="opt")
    with tempfile.TemporaryDirectory(prefix="vcc-sess-") as td:
        p = _Path(td) / "sess.json"
        env.save(p)
        env2 = VirtualCompiler.load(p)
    assert env2.current == env.current
    assert set(env2.branches) == set(env.branches)
    assert env2.branches["root"].runtime_ms == 40.0
    assert len(env2.surrogate) >= 1


# -- features ------------------------------------------------------------
def test_ir_fold_sums_and_maxes():
    events = [
        {"event": "function-ir", "basic_blocks": 9,
         "gimple_statements": 12, "max_loop_depth": 1},
        {"event": "function-ir", "basic_blocks": 3,
         "gimple_statements": 5, "max_loop_depth": 2},
        {"event": "pass-gate", "pass": "evrp"},
    ]
    agg = ir_from_telemetry(events)
    assert agg["basic_blocks"] == 12.0
    assert agg["gimple_statements"] == 17.0
    assert agg["max_loop_depth"] == 2.0  # max, not sum
    assert set(agg) <= set(FEATURE_KEYS)


# -- strategies -----------------------------------------------------------
def test_all_strategies_run_under_budget():
    for name in ("stock", "random", "surrogate", "beam", "mcts"):
        env = _stub_env(budget=6)
        env.validate("root")
        res = run_strategy(env, name, top_k=2, seed=1)
        assert res.strategy == name
        assert env.oracle.used <= 6
        assert res.best_branch in env.branches
    assert set(STRATEGIES) >= {"stock", "random", "surrogate", "beam", "mcts"}


def test_mcts_beats_or_matches_random_signal():
    env = _stub_env(budget=6)
    env.validate("root")
    res = run_strategy(env, "mcts", top_k=2, seed=0)
    # stub world: O3+native (28.0) exists; mcts top-2 should find ≤40
    assert res.best_runtime_ms is not None
    assert res.best_runtime_ms <= 40.0
    assert res.virtual_calls >= 2


def test_virtual_screen_respects_objective_size():
    env = _stub_env(budget=6)
    env.validate("root")
    res = virtual_screen(env, top_k=1, objective="size")
    assert res.oracle_calls == 1


# -- acquisition -----------------------------------------------------------
def test_acquisition_rules_pick_unvalidated():
    env = _stub_env(budget=6)
    env.validate("root")
    env.perturb("flags", "O3", new_branch="a")
    env.perturb("flags", "O0", branch="root", new_branch="b")
    for rule in ("uncertainty", "expected-improvement", "ucb", "info-gain"):
        pick = acquisition.select(env, rule)
        assert pick in ("a", "b"), rule
    try:
        acquisition.select(env, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected bad-rule error")
    try:
        acquisition.score_branch(env, "a", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected bad-rule error")


# -- benchmark --------------------------------------------------------------
def test_killer_benchmark_equal_budgets_and_worst_case():
    programs = {"p1": SRC, "p2": SRC + "/* x */\n"}

    def make_env(source):
        oracle = StubOracle(runtimes=RUNTIMES, budget=5,
                            evidence=EvidenceLog())
        return VirtualCompiler(source, oracle_budget=5, oracle=oracle,
                               evidence=EvidenceLog())

    report = run_killer_benchmark(make_env, programs,
                                  arms=("stock", "random", "beam", "mcts"),
                                  budget_per_program=5, top_k=2, seed=0)
    assert len(report.programs) == 2
    for prog in report.programs:
        for arm in prog.arms:
            assert arm.oracle_calls <= 5, (prog.program, arm.arm)
    assert set(report.worst_case_gain) == {"stock", "random", "beam", "mcts"}
    d = report.to_dict()
    assert d["schema"] == "virtual-compiler.benchmark.v1"
    _ = json.dumps(d)  # serializable


# -- analysis ---------------------------------------------------------------
def test_audit_and_ablation_on_stub():
    env = _stub_env(budget=8)
    env.validate("root")
    run_strategy(env, "beam", top_k=3, seed=0)
    audit = analysis.audit_surrogate(env)
    assert audit["n"] >= 2
    assert audit["verdict"] in ("trustworthy", "unreliable")
    assert 0.0 <= audit["calibration_error"] <= 1.0

    def make_env(source):
        oracle = StubOracle(runtimes=RUNTIMES, budget=6,
                            evidence=EvidenceLog())
        return VirtualCompiler(source, oracle_budget=6, oracle=oracle,
                               evidence=EvidenceLog())

    abl = analysis.ablation_mcts_vs_beam(make_env, SRC, top_k=2, seed=0)
    assert abl["beam"]["best_runtime_ms"] <= 40.0
    assert abl["mcts"]["best_runtime_ms"] <= 40.0


def test_rollout_divergence_reports_depth():
    env = _stub_env(budget=6)
    env.validate("root")
    rows = analysis.rollout_divergence(
        env, "root", [("flags", "O3"), ("policy", "no-evrp")])
    assert [r["depth"] for r in rows] == [1, 2]
    assert all("uncertainty" in r for r in rows)


# -- campaign + swarm ----------------------------------------------------------
def test_campaign_engine_loop():
    env = _stub_env(budget=8)
    env.validate("root")
    out = run_campaign(env, rounds=2, fanout=4, physics_k=1)
    assert out["total_oracle"] <= 8
    assert out["total_virtual"] >= 1
    assert len(out["rounds"]) >= 1
    assert out["incumbent"] in env.branches


def test_swarm_with_critic_and_rule():
    env = _stub_env(budget=8)
    env.validate("root")
    out = run_swarm(env, oracle_rounds=1, top_k=1, rule="ucb")
    assert out["oracle_used"] >= 1
    agents = {v.agent for v in out["log"]}
    assert {"optimizer", "analyst"} <= agents
    assert Critic().review(env) is not None


def test_state_serialization_roundtrip():
    st = CompilationState(SRC, flags=("-O3",), policy_text="disable_pass=evrp")
    st.build_ok, st.runtime_ms, st.validated = True, 30.0, True
    st2 = CompilationState.from_dict(st.to_dict())
    assert st2.state_id == st.state_id
    assert st2.validated and st2.runtime_ms == 30.0


# -- combinatorial tune actions (G4) -----------------------------------------
def test_tune_add_remove_toggle():
    env = _stub_env()
    n1 = env.perturb("tune", "+funroll-loops", new_branch="u1")
    assert env.branches[n1].flags == ("-O2", "-funroll-loops")
    n2 = env.perturb("tune", "-funroll-loops", branch=n1, new_branch="u2")
    assert env.branches[n2].flags == ("-O2", "-fno-unroll-loops")
    n3 = env.perturb("tune", "+funroll-loops", branch=n2, new_branch="u3")
    assert env.branches[n3].flags == ("-O2", "-funroll-loops")
    # idempotent: no duplicates
    n4 = env.perturb("tune", "+funroll-loops", branch=n3, new_branch="u4")
    assert env.branches[n4].flags == ("-O2", "-funroll-loops")
    try:
        env.perturb("tune", "+floop-di-loops", branch="root")
    except ValueError:
        pass
    else:
        raise AssertionError("expected unknown tune flag error")
    try:
        env.perturb("tune", "funroll-loops", branch="root")
    except ValueError:
        pass
    else:
        raise AssertionError("expected malformed tune arg error")


def test_search_space_has_depth():
    from virtual_compiler.search import SEARCH_SPACE, TUNE_ACTIONS
    assert len(TUNE_ACTIONS) == 16
    assert len(SEARCH_SPACE) > 9
    env = _stub_env(budget=6)
    env.validate("root")
    res = virtual_screen(env, top_k=1)
    # Some xform actions may be inapplicable to the stub source, so
    # virtual_calls may be < len(SEARCH_SPACE). But the space is large
    # enough that most actions succeed.
    assert res.virtual_calls >= len(TUNE_ACTIONS)  # all tune actions work
    assert res.virtual_calls <= len(SEARCH_SPACE)  # but not more than space


# -- metrology (G3) ---------------------------------------------------------------
def test_bootstrap_ci_and_gate():
    from virtual_compiler.metrology import bootstrap_median_ci, gate_verdict
    lo, hi = bootstrap_median_ci([10.0] * 9)
    assert lo == hi == 10.0
    lo, hi = bootstrap_median_ci([])
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = bootstrap_median_ci([1.0, 2.0, 3.0, 4.0, 100.0], seed=1)
    assert lo <= 3.0 <= hi  # median inside CI
    assert gate_verdict(5.0, 2.0, 8.0, margin=0.02, baseline=100.0) == "improves"
    assert gate_verdict(1.0, -2.0, 4.0, margin=0.02, baseline=100.0) == "within-noise"
    assert gate_verdict(-5.0, -8.0, -2.0, margin=0.02, baseline=100.0) == "regresses"
    from virtual_compiler.metrology import paired_compare
    try:
        paired_compare("no-such-a", "no-such-b", runs=2, warmups=0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected paired failure on bad binaries")
    try:
        paired_compare("x", "y", runs=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected runs validation")


# -- dataset + LOPO (G1/G8) ----------------------------------------------------------
def _lopo_records():
    from virtual_compiler.features import extract_features
    recs = []
    for prog in ("progA", "progB"):
        for flags, rt in ((("-O0",), 100.0), (("-O2",), 40.0),
                          (("-O3",), 30.0)):
            recs.append({
                "program": prog,
                "flags": list(flags),
                "features": extract_features(SRC, tuple(flags)),
                "outcome": {"build_ok": True, "runtime_ms": rt,
                            "binary_size": 8192},
            })
    return recs


def test_train_from_dataset_and_lopo_beats_mean():
    import tempfile
    from pathlib import Path as _Path

    from virtual_compiler.analysis import lopo
    from virtual_compiler.surrogate import Surrogate
    recs = _lopo_records()
    with tempfile.TemporaryDirectory(prefix="vcc-ds-") as td:
        import json as _json
        p = _Path(td) / "d.jsonl"
        p.write_text("\n".join(_json.dumps(r) for r in recs), encoding="utf-8")
        s = Surrogate()
        assert s.train_from_dataset(p) == 6
        assert len(s) == 6
        out = lopo(recs)
    assert out["verdict"] == "beats-mean", out
    assert out["programs"]["progA"]["pairwise_accuracy"] == 1.0


def test_lopo_insufficient_evidence():
    from virtual_compiler.analysis import lopo
    recs = _lopo_records()
    only_one = [r for r in recs if r["program"] == "progA"][:1]
    assert lopo(only_one)["verdict"] == "insufficient-evidence"
    assert lopo([])["verdict"] == "insufficient-evidence"


# -- seeds + swarm arm (G5/G6) ----------------------------------------------------------
def test_benchmark_seeds_and_swarm_arm():
    from virtual_compiler.benchmark import run_killer_benchmark
    programs = {"p1": SRC}

    def make_env(source):
        oracle = StubOracle(runtimes=RUNTIMES, budget=6,
                            evidence=EvidenceLog())
        return VirtualCompiler(source, oracle_budget=6, oracle=oracle,
                               evidence=EvidenceLog())

    report = run_killer_benchmark(
        make_env, programs, arms=("stock", "beam", "swarm"),
        budget_per_program=6, top_k=1, seed=0, seeds=2)
    labels = sorted(p.program for p in report.programs)
    assert labels == ["p1@s0", "p1@s1"], labels
    for prog in report.programs:
        for arm in prog.arms:
            assert arm.oracle_calls <= 6
            assert arm.flags is not None  # recorded winning config
    d = report.to_dict()
    assert all("verification" in p for p in d["programs"])


def test_swarm_strategy_runs():
    from virtual_compiler.search import run_strategy
    env = _stub_env(budget=8)
    env.validate("root")
    res = run_strategy(env, "swarm", top_k=1, seed=0)
    assert res.strategy == "swarm"
    assert res.best_branch in env.branches
    assert env.oracle.used <= 8


# -- simulator-side constraints (N2/N3.2) -----------------------------------------
def test_constraint_findings():
    from virtual_compiler.constraints import check_flags
    assert check_flags(("-O2",)) == []
    assert check_flags(("-O2", "-funroll-loops")) == []
    errs = check_flags(("-O2", "-funroll-loops", "-fno-unroll-loops"))
    assert any(f["level"] == "error"
               and f["rule"] == "contradictory-flags" for f in errs)
    warns = check_flags(("-O2", "-O3"))
    assert any(f["rule"] == "duplicate-opt-level" for f in warns)
    unk = check_flags(("-O2", "-ffrobulate-the-loops"))
    assert any(f["rule"] == "unknown-flag" for f in unk)
    dup = check_flags(("-O2", "-O2"))
    assert any(f["rule"] == "duplicate-flag" for f in dup)


def test_query_enabled_live_or_silent():
    from virtual_compiler.constraints import query_enabled
    res = query_enabled("gcc", "-O2")
    assert res is None or (
        isinstance(res, frozenset) and len(res) > 50
        and "-fcprop-registers" in res)
    assert query_enabled("no-such-compiler-xyz", "-O2") is None


def test_env_check_and_campaign_skip():
    from virtual_compiler.state import CompilationState
    env = _stub_env()
    assert env.check("root") == []  # stub: static only, presets are clean
    bad = CompilationState(SRC, flags=("-O2", "-funroll-loops",
                                       "-fno-unroll-loops"))
    env.branches["bad"] = bad
    assert any(f["level"] == "error" for f in env.check("bad"))
    out = run_campaign(env, rounds=1, fanout=2, physics_k=1)
    assert "skipped_constraints" in out["rounds"][0]


# -- OOD regimes (N3.3) ------------------------------------------------------------
def test_ood_fit_assess_abstain():
    from virtual_compiler import ood as _ood
    assert _ood.fit([])["threshold"] is None
    assert _ood.fit([[1.0]])["threshold"] is None
    cold = _ood.assess([0.0], _ood.fit([]))
    assert cold["regime"] == "unknown"
    assert all(cold["abstain"].values())
    rows = [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]  # gapped lattice
    model = _ood.fit(rows)
    model["rows"] = rows
    assert model["threshold"] == 1.0
    near = _ood.assess([2.1], model)
    assert near["regime"] == "familiar"
    assert not any(near["abstain"].values())
    far = _ood.assess([6.0], model)
    assert far["regime"] == "novel"
    assert far["abstain"]["magnitude"] and far["abstain"]["ranking"]
    mid = _ood.assess([3.5], model)  # dist 1.5: inside (t, 2t]
    assert mid["regime"] == "unfamiliar"
    # task-dependent: ranking tolerates unfamiliar, magnitude does not
    assert not mid["abstain"]["ranking"] and mid["abstain"]["magnitude"]


def test_ood_fit_on_surrogate_and_latent_rows():
    from virtual_compiler import ood as _ood
    env = _stub_env()
    env.validate("root")
    assert len(env.surrogate.latent_rows()) == 1
    model = _ood.fit_on_surrogate(env.surrogate)
    assert model["n"] == 1  # single row: uncalibrated, abstains
    a = _ood.assess(env.surrogate.latent_rows()[0], model)
    assert all(a["abstain"].values())


# -- uncertainty propagation (N3.4) --------------------------------------------------
def test_propagate_compounds():
    from virtual_compiler.analysis import propagate
    env = _stub_env(budget=6)
    env.validate("root")
    rows = propagate(env, "root", [("flags", "O3"), ("tune", "+funroll-loops")])
    assert [r["depth"] for r in rows] == [1, 2]
    assert rows[0]["uncertainty_propagated"] >= rows[0]["uncertainty_fresh"]
    assert rows[1]["uncertainty_propagated"] >= rows[0]["uncertainty_propagated"]


# -- L1-L5 grading (N7) ----------------------------------------------------------------
def _grade_records():
    from virtual_compiler.features import extract_features
    recs = []
    cfgs = ((("-O0",), 100.0), (("-O2",), 40.0),
            (("-O3",), 30.0), (("-Os",), 55.0))
    for prog in ("progA", "progB"):
        for flags, rt in cfgs:
            recs.append({
                "program": prog,
                "flags": list(flags),
                "features": extract_features(SRC, tuple(flags)),
                "outcome": {"build_ok": True, "runtime_ms": rt,
                            "binary_size": 8192},
            })
    return recs


def test_levels_grade_transferable():
    from virtual_compiler.levels import grade_dataset
    out = grade_dataset(_grade_records(), seed=0)
    assert out["levels"]["L1"]["pass"] is True
    assert out["levels"]["L2"]["pass"] is True
    assert out["levels"]["L3"]["pass"] is True
    assert out["levels"]["L4"]["pass"] is True
    assert out["levels"]["L5"]["pass"] is True
    assert out["grade"] == "L5"
    # deterministic across runs
    out2 = grade_dataset(_grade_records(), seed=0)
    assert out2 == out


def test_levels_refuse_single_program():
    from virtual_compiler.levels import grade_dataset
    recs = [r for r in _grade_records() if r["program"] == "progA"][:2]
    out = grade_dataset(recs, seed=0)
    assert out["grade"] in ("L0", "L1")
    assert out["levels"]["L2"]["lopo_verdict"] == "insufficient-evidence"


def test_grade_session():
    from virtual_compiler.levels import grade_session
    env = _stub_env(budget=6)
    env.validate("root")
    env.perturb("flags", "O3", new_branch="s-opt")
    env.validate("s-opt")
    out = grade_session(env)
    assert out["L1"]["pass"] is True
    assert out["L3_structural"]["branch_count"] == 2


# -- composable policies: subset interactions (§6.6) -------------------------------
def test_policy_union_and_clear():
    env = _stub_env()
    n1 = env.perturb("policy", "no-evrp", new_branch="p1")
    assert env.branches[n1].policy_text == "disable_pass=evrp"
    n2 = env.perturb("policy", "no-ccp", branch=n1, new_branch="p2")
    text = env.branches[n2].policy_text
    assert "disable_pass=evrp" in text and "disable_pass=ccp" in text
    # idempotent: no duplicate rules
    n3 = env.perturb("policy", "no-ccp", branch=n2, new_branch="p3")
    assert env.branches[n3].policy_text.count("disable_pass=ccp") == 1
    # rollout builds the full subset lattice path
    end = env.rollout([("policy", "no-cddce")], branch="p2")
    assert "disable_pass=cddce" in env.branches[end].policy_text
    n4 = env.perturb("policy", "none", branch=end, new_branch="p4")
    assert env.branches[n4].policy_text is None
    try:
        env.perturb("policy", "# just a comment", branch="root")
    except ValueError:
        pass
    else:
        raise AssertionError("expected empty-policy error")


# -- normalized native target (§6.4) --------------------------------------------------
def test_predict_relative():
    from virtual_compiler.state import CompilationState
    from virtual_compiler.surrogate import Surrogate as _S
    surr = _S()
    st = CompilationState(SRC)
    st.build_ok, st.runtime_ms, st.binary_size, st.validated = (
        True, 40.0, 8192, True)
    surr.observe(st)
    q = CompilationState(SRC)
    r = surr.predict_relative(q, 50.0)
    assert r["basis"] == "surrogate"
    assert abs(r["delta"] - 0.2) < 1e-9
    assert surr.predict_relative(q, None)["basis"] == "no-baseline"
    assert surr.predict_relative(q, 0.0)["basis"] == "no-baseline"
    assert _S().predict_relative(q, 50.0)["basis"] == "no-estimate"


# -- PolyBench-grade guards (§6.3) ---------------------------------------------------------
def test_guarded_median():
    from virtual_compiler.metrology import guarded_median
    g = guarded_median([10.0, 10.0, 10.0, 10.0, 100.0])
    assert g["median"] == 10.0 and g["stable"] is True  # extreme dropped
    g = guarded_median([1.0, 2.0, 3.0, 4.0, 5.0])
    assert g["median"] == 3.0 and g["stable"] is False  # exceeds guard
    g = guarded_median([7.0, 9.0])
    assert g["stable"] is False and g["reason"] == "insufficient-samples"
    g = guarded_median([5.0, 5.0, 5.0])
    assert g["stable"] is True
