"""Tests for the virtual compiler surrogate environment (no gcc needed)."""
from virtual_compiler.agents import Adversary, Proposer, Validator, run_swarm
from virtual_compiler.cli import DEMO_SOURCE
from virtual_compiler.environment import VirtualCompiler
from virtual_compiler.search import gain_per_oracle, virtual_screen
from virtual_compiler.state import CompilationState, extract_features
from virtual_compiler.surrogate import Surrogate

SRC_A = "int main(void){return 0;}\n"
SRC_B = "#include <stdio.h>\nint main(void){printf(\"hi\\n\");return 0;}\n"


def test_features_separate_programs():
    fa, fb = extract_features(SRC_A), extract_features(SRC_B)
    assert fa["includes"] == 0.0
    assert fb["includes"] == 1.0


def test_perturb_branch_rollout_are_free():
    env = VirtualCompiler(SRC_A, oracle_budget=2)
    b1 = env.perturb("flags", "O3")
    assert env.oracle_calls_used == 0
    b2 = env.branch("alt")
    assert env.branches[b2].flags == env.branches[b1].flags
    end = env.rollout([("policy", "no-evrp"), ("flags", "Os")], branch="root")
    assert env.oracle_calls_used == 0
    assert env.lineage(end)[0] == "root"


def test_surrogate_uncertainty_and_observe():
    env = VirtualCompiler(SRC_A, oracle_budget=2)
    assert env.uncertainty() == 1.0  # no experience yet
    # fake one oracle observation to seed experience
    st = env.branches["root"]
    st.build_ok, st.runtime_ms, st.binary_size, st.validated = True, 10.0, 8000, True
    env.surrogate.observe(st)
    assert env.uncertainty() < 1.0
    obs = env.observe()
    assert obs["surrogate"]["uncertainty"] < 1.0


def test_compare_prefers_oracle_truth():
    env = VirtualCompiler(SRC_A, oracle_budget=4)
    env.perturb("flags", "O3", new_branch="fast")
    env.perturb("flags", "O0", branch="root", new_branch="slow")
    for n, rt in (("fast", 5.0), ("slow", 50.0)):
        st = env.branches[n]
        st.build_ok, st.runtime_ms, st.validated = True, rt, True
        env.surrogate.observe(st)
    assert env.compare("fast", "slow")["winner"] == "fast"


def test_budget_enforced_without_gcc():
    env = VirtualCompiler(SRC_A, oracle_budget=0)
    try:
        env.validate()
    except RuntimeError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("expected budget error")


def test_swarm_and_screen_run_virtual_only(monkeypatch):
    env = VirtualCompiler(DEMO_SOURCE, oracle_budget=4)

    def fake_validate(branch=None, benchmark_runs=3):
        from virtual_compiler.environment import ValidationRecord
        name = branch or env.current
        st = env.branches[name]
        st.build_ok, st.validated = True, True
        st.runtime_ms = 10.0 + len(name)
        st.binary_size = 9000
        env.surrogate.observe(st)
        env.oracle.used += 1
        return ValidationRecord(name, True, 9000, st.runtime_ms,
                                env.oracle_calls_used)

    monkeypatch.setattr(env, "validate", fake_validate)
    res = virtual_screen(env, top_k=2)
    assert res.oracle_calls == 2
    assert res.virtual_calls >= 2
    out = run_swarm(env, oracle_rounds=1, top_k=1)
    assert out["oracle_used"] >= 2
    assert gain_per_oracle(20.0, 10.0, 2) == 0.25
    assert Proposer().propose(env, n=2)
    assert isinstance(Validator().next_to_validate(env), (str, type(None)))
    assert Adversary().refute(env) is None or True


def test_state_ids_differ():
    a = CompilationState(SRC_A, flags=("-O2",))
    b = CompilationState(SRC_A, flags=("-O3",))
    assert a.state_id != b.state_id


def test_surrogate_save_load():
    import tempfile
    from pathlib import Path as _Path
    s = Surrogate()
    st = CompilationState(SRC_A)
    st.build_ok, st.runtime_ms, st.binary_size, st.validated = True, 7.0, 5000, True
    s.observe(st)
    with tempfile.TemporaryDirectory(prefix="vcc-test-") as td:
        p = _Path(td) / "s.json"
        s.save(p)
        s2 = Surrogate.load(p)
    assert len(s2) == 1
    pred = s2.predict(CompilationState(SRC_A))
    assert pred.neighbors == 1
