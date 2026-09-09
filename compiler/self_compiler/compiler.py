from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    command: tuple[str, ...]
    output_path: str
    elapsed_ms: float
    timed_out: bool = False


def _validate_flags(flags: Sequence[str]) -> None:
    for index, flag in enumerate(flags):
        if flag == "-o" or flag.startswith("-o") and len(flag) > 2:
            raise ValueError(
                "Compiler output is managed by GCC-AI Repair; use --output instead of -o."
            )
        if flag == "/Fe" or flag.startswith("/Fe"):
            raise ValueError(
                "Compiler output is managed by GCC-AI Repair; use --output instead of /Fe."
            )


def compile_source(
    source_path: str,
    compiler: str = "gcc",
    flags: Sequence[str] | None = None,
    *,
    output_path: str,
    timeout: float = 30.0,
) -> CompileResult:
    """Compile one source file without invoking a shell.

    The caller owns ``output_path``. RepairLoop always supplies a path in its
    private temporary directory, so speculative builds never overwrite a user
    artifact.
    """
    if timeout <= 0:
        raise ValueError("Compilation timeout must be greater than zero.")

    compiler_flags = list(flags or ())
    _validate_flags(compiler_flags)
    command = [compiler, *compiler_flags, source_path, "-o", output_path]
    environment = os.environ.copy()
    environment.setdefault("GCC_COLORS", "")
    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=environment,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Compiler '{compiler}' was not found on PATH. Install it or select --backend."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return CompileResult(
            ok=False,
            stdout=_decode_timeout_stream(exc.stdout),
            stderr=_decode_timeout_stream(exc.stderr)
            or f"Compilation timed out after {timeout:g}s.",
            returncode=-1,
            command=tuple(command),
            output_path=str(Path(output_path)),
            elapsed_ms=elapsed_ms,
            timed_out=True,
        )

    return CompileResult(
        ok=result.returncode == 0,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        command=tuple(command),
        output_path=str(Path(output_path)),
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
