import pytest
import subprocess
from self_compiler.compiler import compile_source


def test_compile_success(tmp_path):
    c_file = tmp_path / "valid.c"
    c_file.write_text("int main() { return 0; }")
    out_file = tmp_path / "a.out"

    try:
        # Assuming gcc is available, otherwise this test will fail gracefully
        result = compile_source(str(c_file), "gcc", output_path=str(out_file))
        assert result.ok is True
        assert result.returncode == 0
    except RuntimeError:
        pytest.skip("Compiler not found on host system.")


def test_rejects_caller_controlled_output_flag(tmp_path):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 0; }")
    with pytest.raises(ValueError, match="--output"):
        compile_source(
            str(source),
            flags=["-o", "elsewhere"],
            output_path=str(tmp_path / "managed"),
        )


def test_compile_timeout_is_structured(monkeypatch, tmp_path):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 0; }")

    def expire(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], stderr=b"partial")

    monkeypatch.setattr(subprocess, "run", expire)
    result = compile_source(
        str(source), output_path=str(tmp_path / "program"), timeout=0.1
    )
    assert result.ok is False
    assert result.timed_out is True
    assert result.returncode == -1
    assert result.stderr == "partial"
