import shutil

import pytest

from self_compiler.cli import main


pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is required")


def test_offline_cli_repairs_in_one_attempt(tmp_path):
    source = tmp_path / "broken.c"
    output = tmp_path / "program.exe"
    source.write_text(
        "int main(void) {\n    int value = 1\n    return value - 1;\n}\n",
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--repair",
            "--offline",
            "--max-attempts",
            "1",
            "--output",
            str(output),
            str(source),
            "--",
            "-Wall",
        ]
    )
    assert exit_code == 0
    assert output.exists()
    assert "int value = 1;" in source.read_text(encoding="utf-8")


def test_cli_rejects_compiler_output_flag(tmp_path):
    source = tmp_path / "valid.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    assert main([str(source), "--", "-o", "other"]) == 1
