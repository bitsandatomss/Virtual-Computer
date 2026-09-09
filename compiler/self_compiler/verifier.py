from __future__ import annotations

import os
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .debugger import CrashReport, debug_binary


@dataclass(frozen=True)
class VerifyResult:
    passed: bool
    output: str
    elapsed_ms: float
    returncode: int
    timed_out: bool = False
    crash_report: CrashReport | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    passed: bool
    median_ms: float
    samples_ms: tuple[float, ...]
    output: str
    returncode: int
    reason: str = ""

    @property
    def relative_spread(self) -> float:
        """Sample dispersion as (max-min)/median; 0.0 when not measurable."""
        if len(self.samples_ms) < 2 or self.median_ms <= 0:
            return 0.0
        return (max(self.samples_ms) - min(self.samples_ms)) / self.median_ms


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Terminate a timed-out child and its descendants.

    A shell command template spawns real grandchildren; killing only the
    shell leaves them running, and CPU burned by orphans pollutes every
    later timing measurement in the same lab run.
    """
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def verify(
    binary_path: str,
    test_cmd: str | None = None,
    timeout: float = 5.0,
    source_path: str | None = None,
    program_args: Sequence[str] = (),
    debugger: str | None = None,
) -> VerifyResult:
    """Run a candidate binary or an explicit command template.

    Custom commands may use ``{binary}`` and ``{source}``. They intentionally
    run through the system shell to support pipes/redirection; the values are
    shell-quoted before substitution. Timed-out processes are killed together
    with their descendants.
    """
    if timeout <= 0:
        raise ValueError("Verification timeout must be greater than zero.")

    binary = str(Path(binary_path).resolve())
    source = str(Path(source_path).resolve()) if source_path else ""
    environment = os.environ.copy()
    environment["SELF_COMPILER_BINARY"] = binary
    if source:
        environment["SELF_COMPILER_SOURCE"] = source

    if test_cmd:
        command: str | list[str] = _render_command(test_cmd, binary, source)
        use_shell = True
    else:
        command = [binary, *program_args]
        use_shell = False

    start = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=environment,
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        return VerifyResult(False, str(exc), 0.0, -1)

    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError):
            stdout, stderr = "", ""

    output = stdout or ""
    if stderr:
        output += ("\n" if output else "") + stderr
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if timed_out:
        return VerifyResult(
            passed=False,
            output=(f"[TIMEOUT after {timeout:g}s] " + output).strip(),
            elapsed_ms=elapsed_ms,
            returncode=-1,
            timed_out=True,
        )

    returncode = process.returncode if process.returncode is not None else -1
    crash_report = None
    if returncode != 0 and debugger and not test_cmd:
        crash_report = debug_binary(
            binary,
            program_args,
            debugger=debugger,
            timeout=max(timeout, 5.0),
        )
    return VerifyResult(
        passed=returncode == 0,
        output=output.strip(),
        elapsed_ms=elapsed_ms,
        returncode=returncode,
        crash_report=crash_report,
    )


def benchmark(
    binary_path: str,
    command: str,
    runs: int = 5,
    warmups: int = 1,
    timeout: float = 5.0,
    source_path: str | None = None,
) -> BenchmarkResult:
    if runs < 1 or warmups < 0:
        raise ValueError("Benchmark runs must be positive and warmups non-negative.")

    reference: VerifyResult | None = None
    samples: list[float] = []
    for index in range(warmups + runs):
        result = verify(binary_path, command, timeout, source_path)
        if not result.passed:
            return BenchmarkResult(
                False,
                0.0,
                tuple(samples),
                result.output,
                result.returncode,
                "benchmark command failed",
            )
        if reference is None:
            reference = result
        elif (result.returncode, result.output) != (
            reference.returncode,
            reference.output,
        ):
            return BenchmarkResult(
                False,
                0.0,
                tuple(samples),
                result.output,
                result.returncode,
                "benchmark output was not deterministic across runs",
            )
        if index >= warmups:
            samples.append(result.elapsed_ms)

    assert reference is not None
    return BenchmarkResult(
        True,
        statistics.median(samples),
        tuple(samples),
        reference.output,
        reference.returncode,
    )


def _render_command(template: str, binary: str, source: str) -> str:
    quote = (
        subprocess.list2cmdline
        if os.name == "nt"
        else lambda values: shlex.quote(values[0])
    )
    quoted_binary = quote([binary])
    quoted_source = quote([source])
    rendered = template.replace('"{binary}"', quoted_binary)
    rendered = rendered.replace('"{source}"', quoted_source)
    rendered = rendered.replace("{binary}", quoted_binary)
    rendered = rendered.replace("{source}", quoted_source)
    return rendered
