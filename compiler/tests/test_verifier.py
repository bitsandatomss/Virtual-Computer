import sys

from self_compiler.verifier import benchmark, verify


def test_command_template_receives_quoted_binary():
    command = '"{binary}" -c "print(42)"'
    result = verify(sys.executable, command, timeout=5)
    assert result.passed is True
    assert result.output == "42"


def test_benchmark_rejects_nondeterministic_output(monkeypatch):
    outputs = iter(["warmup", "different"])

    def fake_verify(*args, **kwargs):
        from self_compiler.verifier import VerifyResult

        return VerifyResult(True, next(outputs), 1.0, 0)

    monkeypatch.setattr("self_compiler.verifier.verify", fake_verify)
    result = benchmark("program", "{binary}", runs=1, warmups=1)
    assert result.passed is False
    assert "not deterministic" in result.reason
