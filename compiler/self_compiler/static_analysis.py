from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .compiler import compile_source
from .diagnostics import Diagnostic, parse_compiler_stderr


@dataclass(frozen=True)
class AnalysisResult:
    diagnostics: tuple[Diagnostic, ...]
    tools_run: tuple[str, ...]
    tools_unavailable: tuple[str, ...]

    @property
    def available(self) -> bool:
        return bool(self.tools_run)


def run_cppcheck(source_path: str, timeout: float = 30.0) -> list[Diagnostic] | None:
    executable = shutil.which("cppcheck")
    if not executable:
        return None
    command = [
        executable,
        # Security repair consumes these findings, so only bug-shaped
        # categories are enabled; style/performance findings are not
        # security evidence and would flood the repair loop with noise.
        "--enable=warning,portability",
        "--template=gcc",
        source_path,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [
            Diagnostic(
                source_path, 0, 0, "error", f"cppcheck timed out after {timeout:g}s"
            )
        ]
    return parse_compiler_stderr(result.stderr)


def run_gcc_analyzer(
    source_path: str,
    compiler: str,
    compiler_flags: Sequence[str],
    output_path: str,
    timeout: float = 30.0,
) -> list[Diagnostic] | None:
    compiler_name = Path(compiler).name.lower()
    if "gcc" not in compiler_name and "g++" not in compiler_name:
        return None

    analysis_flags = [
        *compiler_flags,
        "-fanalyzer",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Wformat=2",
    ]
    result = compile_source(
        source_path,
        compiler,
        analysis_flags,
        output_path=output_path,
        timeout=timeout,
    )
    diagnostics = parse_compiler_stderr(result.stderr)
    if result.returncode != 0 and not diagnostics:
        if "fanalyzer" in result.stderr and (
            "unrecognized" in result.stderr.lower()
            or "unknown" in result.stderr.lower()
        ):
            return None
        diagnostics.append(
            Diagnostic(
                source_path,
                0,
                0,
                "error",
                result.stderr.strip() or f"GCC analyzer exited {result.returncode}",
            )
        )
    return diagnostics


def analyze(
    source_path: str,
    compiler: str = "gcc",
    compiler_flags: Sequence[str] = (),
    output_path: str | None = None,
    timeout: float = 30.0,
) -> AnalysisResult:
    """Run available analyzers and report tool availability explicitly."""
    diagnostics: list[Diagnostic] = []
    tools_run: list[str] = []
    unavailable: list[str] = []

    cppcheck = run_cppcheck(source_path, timeout)
    if cppcheck is None:
        unavailable.append("cppcheck")
    else:
        tools_run.append("cppcheck")
        diagnostics.extend(cppcheck)

    analyzer_output = output_path or os.devnull
    gcc = run_gcc_analyzer(
        source_path, compiler, compiler_flags, analyzer_output, timeout
    )
    if gcc is None:
        unavailable.append("gcc -fanalyzer")
    else:
        tools_run.append("gcc -fanalyzer")
        diagnostics.extend(gcc)

    unique = list(dict.fromkeys(diagnostics))
    return AnalysisResult(tuple(unique), tuple(tools_run), tuple(unavailable))
