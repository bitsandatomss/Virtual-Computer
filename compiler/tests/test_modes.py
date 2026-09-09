import shutil

import pytest

from self_compiler.ai_engine import AIEngine
from self_compiler.diagnostics import diagnostic_from_output
from self_compiler.modes import RepairLoop
from self_compiler.verifier import BenchmarkResult


pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is required")


class FixedAI:
    def __init__(self, candidate):
        self.candidate = candidate

    def propose_patch(self, source, diagnostics, mode):
        return self.candidate


class MutatingAI(FixedAI):
    def __init__(self, candidate, source_path):
        super().__init__(candidate)
        self.source_path = source_path

    def propose_patch(self, source, diagnostics, mode):
        self.source_path.write_text("external edit\n", encoding="utf-8")
        return super().propose_patch(source, diagnostics, mode)


class ExplodingAI:
    def propose_patch(self, source, diagnostics, mode):
        raise AssertionError("AI must not run when GCC supplied a valid fix-it")


class CapturingAI(FixedAI):
    def __init__(self, candidate):
        super().__init__(candidate)
        self.diagnostics = None

    def propose_patch(self, source, diagnostics, mode):
        self.diagnostics = diagnostics
        return super().propose_patch(source, diagnostics, mode)


def test_one_attempt_can_repair_and_publish_without_cwd_artifact(tmp_path, monkeypatch):
    source = tmp_path / "broken.c"
    source.write_text(
        "int main(void) {\n    int x = 1\n    return x - 1;\n}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    loop = RepairLoop(str(source), max_attempts=1, ai=AIEngine(offline=True))
    assert loop.run("repair") is True
    assert "int x = 1;" in source.read_text(encoding="utf-8")
    assert not (tmp_path / "a.out").exists()
    assert not list(tmp_path.glob(".*.candidate.*"))


def test_gcc_fixit_repairs_before_ai_is_consulted(tmp_path):
    source = tmp_path / "member.c"
    source.write_text(
        "struct S { int colour; };\n"
        "int main(void) { struct S value = {0}; return value.color; }\n",
        encoding="utf-8",
    )
    loop = RepairLoop(str(source), max_attempts=1, ai=ExplodingAI())
    assert loop.run("repair") is True
    assert "value.colour" in source.read_text(encoding="utf-8")


def test_failed_candidates_leave_source_unchanged(tmp_path):
    source = tmp_path / "broken.c"
    original = "int main(void) { this is invalid }\n"
    source.write_text(original, encoding="utf-8")

    loop = RepairLoop(str(source), max_attempts=1, ai=FixedAI("still invalid"))
    assert loop.run("repair") is False
    assert source.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".*.candidate.*"))


def test_concurrent_source_edit_is_not_overwritten(tmp_path):
    source = tmp_path / "broken.c"
    original = "int main(void) { return nope; }\n"
    candidate = "int main(void) { return 0; }\n"
    source.write_text(original, encoding="utf-8")
    loop = RepairLoop(str(source), max_attempts=1, ai=MutatingAI(candidate, source))
    assert loop.run("repair") is False
    assert source.read_text(encoding="utf-8") == "external edit\n"


def test_post_promotion_gate_failure_rolls_back(tmp_path, monkeypatch):
    source = tmp_path / "broken.c"
    original = "int main(void) { return nope; }\n"
    candidate = "int main(void) { return 0; }\n"
    source.write_text(original, encoding="utf-8")
    loop = RepairLoop(str(source), max_attempts=1, ai=FixedAI(candidate))
    calls = iter([[], [diagnostic_from_output(str(source), "final gate failed")]])
    monkeypatch.setattr(loop, "_validate_candidate", lambda *args: next(calls))
    assert loop.run("repair") is False
    assert source.read_text(encoding="utf-8") == original


def test_self_correct_requires_reproducer(tmp_path):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 1; }\n", encoding="utf-8")
    loop = RepairLoop(str(source), ai=FixedAI(""))
    assert loop.run("self-correct") is False


@pytest.mark.skipif(shutil.which("gdb") is None, reason="gdb is required")
def test_self_correct_receives_structured_gdb_stack(tmp_path):
    source = tmp_path / "crash.c"
    source.write_text(
        "static int read_value(const int *p) { return *p; }\n"
        "static int compute(void) { return read_value((void *)0); }\n"
        "int main(void) { return compute(); }\n",
        encoding="utf-8",
    )
    candidate = "int main(void) { return 0; }\n"
    ai = CapturingAI(candidate)
    loop = RepairLoop(
        str(source),
        compiler_args=["-g", "-O0"],
        max_attempts=1,
        debugger="gdb",
        ai=ai,
    )
    assert loop.run("self-correct") is True
    assert ai.diagnostics is not None
    evidence = "\n".join(diagnostic.message for diagnostic in ai.diagnostics)
    assert "SIGSEGV" in evidence
    assert "read_value" in evidence
    assert "compute" in evidence


def test_secure_requires_regression_command(tmp_path):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    loop = RepairLoop(str(source), ai=FixedAI(""))
    assert loop.run("secure") is False


def test_secure_repairs_finding_and_preserves_output(tmp_path):
    source = tmp_path / "unsafe.c"
    original = """#include <stdio.h>
#include <string.h>
int main(void) {
    char buffer[32];
    gets(buffer);
    printf("value=%s\\n", buffer);
    return 0;
}
"""
    candidate = original.replace(
        "    gets(buffer);",
        "    if (!fgets(buffer, sizeof buffer, stdin)) return 1;\n    buffer[strcspn(buffer, \"\\r\\n\")] = '\\0';",
    )
    source.write_text(original, encoding="utf-8")
    loop = RepairLoop(
        str(source),
        ai=FixedAI(candidate),
        max_attempts=1,
        test_cmd="echo hello|{binary}",
    )
    baseline = loop._compile(str(source), tmp_path / "baseline.exe")
    if not baseline.ok:
        pytest.skip("host C library no longer links gets")
    analysis = loop._analyze(str(source), tmp_path / "analysis.exe")
    if not analysis.diagnostics:
        pytest.skip("host analyzers do not flag gets")

    assert loop.run("secure") is True
    assert "fgets" in source.read_text(encoding="utf-8")


def test_optimize_requires_benchmark(tmp_path):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    loop = RepairLoop(str(source), ai=FixedAI(""))
    assert loop.run("optimize") is False


def test_optimize_accepts_only_measured_improvement(tmp_path, monkeypatch):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    candidate = "/* faster */\nint main(void) { return 0; }\n"
    measurements = iter(
        [
            BenchmarkResult(True, 100.0, (100.0,), "", 0),
            BenchmarkResult(True, 80.0, (80.0,), "", 0),
            BenchmarkResult(True, 80.0, (80.0,), "", 0),
        ]
    )
    loop = RepairLoop(
        str(source),
        ai=FixedAI(candidate),
        max_attempts=1,
        benchmark_cmd="{binary}",
        benchmark_runs=3,
    )
    monkeypatch.setattr(loop, "_benchmark", lambda *args: next(measurements))
    assert loop.run("optimize") is True
    assert source.read_text(encoding="utf-8") == candidate


def test_noop_fixit_falls_through_to_ai_same_attempt(tmp_path):
    source = tmp_path / "broken.c"
    original = "int main(void) { return nope; }\n"
    candidate = "int main(void) { return 0; }\n"
    source.write_text(original, encoding="utf-8")

    calls = {"ai": 0}

    class RecordingAI(FixedAI):
        def propose_patch(self, source, diagnostics, mode):
            calls["ai"] += 1
            return super().propose_patch(source, diagnostics, mode)

    loop = RepairLoop(str(source), max_attempts=1, ai=RecordingAI(candidate))
    # A fix-it that replaces a byte range with identical content produces
    # no new revision; the loop must consult AI within the same attempt.
    from self_compiler.diagnostics import FixIt, SourcePoint

    original_bytes = source.read_text(encoding="utf-8").encode("utf-8")
    target = original_bytes.index(b"nope") + 1  # GCC byte columns are 1-based
    point = SourcePoint(str(source), 1, target, target)
    end_point = SourcePoint(
        str(source), 1, target + len(b"nope"), target + len(b"nope")
    )
    loop._next_candidate(
        1,
        current=original,
        feedback=[
            diagnostic_from_output(str(source), "synthetic").__class__(
                file=str(source),
                line=1,
                col=target,
                severity="error",
                message="noop",
                code="",
                fixits=(FixIt(point, end_point, "nope"),),
            )
        ],
        current_path=str(source),
        seen=set(),
        mode="repair",
    )
    assert calls["ai"] == 1


def test_valid_fixit_still_prevents_ai_consultation(tmp_path):
    source = tmp_path / "member.c"
    source.write_text(
        "struct S { int colour; };\n"
        "int main(void) { struct S value = {0}; return value.color; }\n",
        encoding="utf-8",
    )
    loop = RepairLoop(str(source), max_attempts=1, ai=ExplodingAI())
    assert loop.run("repair") is True
