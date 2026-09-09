from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from self_compiler import differential
from self_compiler.differential import (
    generate_cases,
    run_differential,
    write_report,
)

gcc_available = shutil.which("gcc") is not None


def test_generate_cases_is_deterministic_and_edge_first():
    first = generate_cases(8, seed=1234)
    second = generate_cases(8, seed=1234)
    other = generate_cases(8, seed=99)

    assert [case.name for case in first] == [case.name for case in second]
    assert [case.stdin for case in first] == [case.stdin for case in second]
    # Edge cases are fixed regardless of seed...
    assert first[0].stdin == other[0].stdin == b""
    assert first[1].stdin == other[1].stdin == b"\x00"
    # ...while the seeded random tail differs across seeds.
    assert any(a.stdin != b.stdin for a, b in zip(first[5:], other[5:]))

    with pytest.raises(ValueError):
        generate_cases(0, seed=1)


def test_detects_exit_code_divergence(monkeypatch, tmp_path):
    def fake_run(binary, stdin, timeout):
        if Path(binary).name == "ref":
            return 0, b"ok", ""
        return (1, b"ok", "") if stdin else (0, b"ok", "")

    monkeypatch.setattr(differential, "_run_one", fake_run)
    report = run_differential(tmp_path / "ref", tmp_path / "cand", trials=6, seed=1)
    assert report.equivalent is False
    assert {m.kind for m in report.mismatches} == {"exit-code"}
    assert all("exited" in m.detail for m in report.mismatches)


def test_detects_stdout_divergence(monkeypatch, tmp_path):
    def fake_run(binary, stdin, timeout):
        payload = b"same"
        if Path(binary).name == "cand":
            payload = b"different"
        return 0, payload, ""

    monkeypatch.setattr(differential, "_run_one", fake_run)
    report = run_differential(tmp_path / "ref", tmp_path / "cand", trials=3, seed=2)
    assert report.equivalent is False
    assert {m.kind for m in report.mismatches} == {"stdout"}


def test_timeout_classification(monkeypatch, tmp_path):
    def fake_run(binary, stdin, timeout):
        if Path(binary).name == "ref":
            return None, b"", "timeout"
        if not stdin:
            return None, b"", "timeout"
        return 0, b"x", ""

    monkeypatch.setattr(differential, "_run_one", fake_run)
    report = run_differential(tmp_path / "ref", tmp_path / "cand", trials=6, seed=3)
    timeout_cases = {m.case for m in report.mismatches if m.kind == "timeout"}
    # Empty stdin times out on both sides and must be skipped, never a mismatch.
    assert "empty" not in timeout_cases
    assert any(case != "empty" for case in timeout_cases)
    assert all(m.kind != "error" for m in report.mismatches)


def test_execution_error_is_recorded(monkeypatch, tmp_path):
    def fake_run(binary, stdin, timeout):
        if Path(binary).name == "cand" and stdin == b"\x00":
            return None, b"", f"binary not found: {binary}"
        return 0, b"", ""

    monkeypatch.setattr(differential, "_run_one", fake_run)
    report = run_differential(tmp_path / "ref", tmp_path / "cand", trials=5, seed=4)
    assert any(m.kind == "error" for m in report.mismatches)


def test_report_file_round_trip(monkeypatch, tmp_path):
    def fake_run(binary, stdin, timeout):
        return 0, b"", ""

    monkeypatch.setattr(differential, "_run_one", fake_run)
    report = run_differential(tmp_path / "ref", tmp_path / "cand", trials=2, seed=7)
    out_path = tmp_path / "diff.jsonl"
    write_report(report, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert '"equivalent": true' in text.replace(" ", " ")
    assert '"event": "decision"' in text


def _compile(source_text: str, destination: Path) -> Path:
    source_path = destination.with_suffix(".c")
    source_path.write_text(source_text, encoding="utf-8")
    completed = subprocess.run(
        ["gcc", str(source_path), "-o", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return destination


@pytest.mark.skipif(not gcc_available, reason="gcc required")
def test_identical_binaries_are_equivalent(tmp_path):
    source = (
        "#include <stdio.h>\n"
        "int main(void) {\n"
        "    int value = getchar();\n"
        '    printf("len=%d\\n", value >= 0);\n'
        "    return 0;\n"
        "}\n"
    )
    left = _compile(source, tmp_path / "left.exe")
    right = _compile(source, tmp_path / "right.exe")
    report = run_differential(left, right, trials=5, seed=11)
    assert report.equivalent is True


@pytest.mark.skipif(not gcc_available, reason="gcc required")
def test_extra_output_byte_is_caught(tmp_path):
    good = (
        '#include <stdio.h>\nint main(void) {\n    printf("ok\\n");\n    return 0;\n}\n'
    )
    bad = good.replace('printf("ok\\n");', 'printf("ok \\n");')
    reference = _compile(good, tmp_path / "ref.exe")
    candidate = _compile(bad, tmp_path / "cand.exe")
    report = run_differential(reference, candidate, trials=4, seed=12)
    assert report.equivalent is False
    assert {m.kind for m in report.mismatches} == {"stdout"}
