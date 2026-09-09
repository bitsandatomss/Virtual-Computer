import shutil
import subprocess

import pytest

from self_compiler.debugger import _parse_stack_frames, debug_binary


MI_TRANSCRIPT = r"""
*stopped,reason="signal-received",signal-name="SIGSEGV",signal-meaning="Segmentation fault",frame={addr="0x1",func="read_value",file="demo.c",fullname="C:\\work\\demo.c",line="4"}
4^done,stack=[frame={level="0",addr="0x1",func="read_value",file="demo.c",fullname="C:\\work\\demo.c",line="4",arch="i386:x86-64"},frame={level="1",addr="0x2",func="main",file="demo.c",fullname="C:\\work\\demo.c",line="8",arch="i386:x86-64"}]
"""


def test_parses_structured_mi_stack_frames():
    frames = _parse_stack_frames(MI_TRANSCRIPT)
    assert [(frame.level, frame.function, frame.line) for frame in frames] == [
        (0, "read_value", 4),
        (1, "main", 8),
    ]
    assert frames[0].fullname == r"C:\work\demo.c"


@pytest.mark.skipif(
    shutil.which("gcc") is None or shutil.which("gdb") is None,
    reason="gcc and gdb are required",
)
def test_live_gdb_mi_captures_signal_and_call_chain(tmp_path):
    source = tmp_path / "crash.c"
    binary = tmp_path / "crash.exe"
    source.write_text(
        "static int read_value(const int *p) { return *p; }\n"
        "static int compute(void) { return read_value((void *)0); }\n"
        "int main(void) { return compute(); }\n",
        encoding="utf-8",
    )
    compiled = subprocess.run(
        ["gcc", "-g", "-O0", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr

    report = debug_binary(str(binary), timeout=15)
    assert report.crashed is True
    assert report.signal == "SIGSEGV"
    assert [frame.function for frame in report.frames[:3]] == [
        "read_value",
        "compute",
        "main",
    ]
    formatted = report.format_for_diagnostic()
    assert "SIGSEGV" in formatted
    assert "read_value" in formatted
