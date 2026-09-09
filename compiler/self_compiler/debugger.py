from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class StackFrame:
    level: int
    address: str
    function: str
    file: str
    fullname: str
    line: int


@dataclass(frozen=True)
class CrashReport:
    crashed: bool
    signal: str
    signal_meaning: str
    frames: tuple[StackFrame, ...]
    debugger: str
    error: str = ""

    def format_for_diagnostic(self) -> str:
        if self.error:
            return f"Debugger evidence unavailable: {self.error}"
        if not self.crashed:
            return "GDB did not observe a signal-received stop."
        lines = [f"GDB observed {self.signal}: {self.signal_meaning}"]
        for frame in self.frames:
            location = frame.fullname or frame.file or "<unknown>"
            if frame.line:
                location += f":{frame.line}"
            lines.append(
                f"#{frame.level} {frame.function or '<unknown>'} at {location} "
                f"[{frame.address or 'no-address'}]"
            )
        return "\n".join(lines)


def debug_binary(
    binary_path: str,
    program_args: Sequence[str] = (),
    *,
    debugger: str = "gdb",
    timeout: float = 15.0,
) -> CrashReport:
    """Run a binary under GDB/MI and return structured crash evidence."""
    command = [
        debugger,
        "--quiet",
        "--nx",
        "--interpreter=mi2",
        "--args",
        str(Path(binary_path).resolve()),
        *program_args,
    ]
    requests = (
        "1-gdb-set pagination off\n"
        "2-gdb-set confirm off\n"
        "3-exec-run\n"
        "4-stack-list-frames\n"
        "5-gdb-exit\n"
    )
    try:
        result = subprocess.run(
            command,
            input=requests,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return CrashReport(False, "", "", (), debugger, f"'{debugger}' was not found")
    except subprocess.TimeoutExpired:
        return CrashReport(
            False, "", "", (), debugger, f"GDB timed out after {timeout:g}s"
        )

    transcript = result.stdout + ("\n" + result.stderr if result.stderr else "")
    signal = _mi_field(transcript, "signal-name")
    meaning = _mi_field(transcript, "signal-meaning")
    frames = tuple(_parse_stack_frames(transcript))
    crashed = 'reason="signal-received"' in transcript
    error = ""
    if result.returncode not in {0, 1} and not crashed:
        error = f"GDB exited {result.returncode}"
    return CrashReport(crashed, signal, meaning, frames, debugger, error)


def _parse_stack_frames(transcript: str) -> list[StackFrame]:
    stack_record = ""
    for line in transcript.splitlines():
        if "^done,stack=[" in line:
            stack_record = line
    frames: list[StackFrame] = []
    for record in _braced_records(stack_record, "frame={"):
        frames.append(
            StackFrame(
                level=_int_field(record, "level"),
                address=_record_field(record, "addr"),
                function=_record_field(record, "func"),
                file=_record_field(record, "file"),
                fullname=_record_field(record, "fullname"),
                line=_int_field(record, "line"),
            )
        )
    return frames


def _braced_records(text: str, marker: str) -> list[str]:
    records: list[str] = []
    search_from = 0
    while True:
        marker_index = text.find(marker, search_from)
        if marker_index < 0:
            break
        start = marker_index + len(marker) - 1
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if quoted and char == "\\":
                escaped = True
                continue
            if char == '"':
                quoted = not quoted
                continue
            if quoted:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    records.append(text[start + 1 : index])
                    search_from = index + 1
                    break
        else:
            break
    return records


def _mi_field(text: str, name: str) -> str:
    match = re.search(rf'(?:^|,){re.escape(name)}="((?:\\.|[^"\\])*)"', text)
    return _decode_mi_string(match.group(1)) if match else ""


def _record_field(record: str, name: str) -> str:
    return _mi_field(record, name)


def _int_field(record: str, name: str) -> int:
    value = _record_field(record, name)
    try:
        return int(value)
    except ValueError:
        return 0


def _decode_mi_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r"\\", "\\").replace(r"\"", '"')
