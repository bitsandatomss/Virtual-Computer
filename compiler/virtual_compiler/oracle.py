"""The truth oracle: real toolchain measurements under a finite budget.

context.txt T12: "real biological measurements are the oracle… that's
exactly how a surrogate should be validated." Here the oracle is gcc +
the AI-Compiler harness: builds (`compiler.compile_source`), timed runs
(`verifier.verify/benchmark`), GIMPLE telemetry (`gcc_native` plugin
facts folded via `features.ir_from_telemetry`), and the differential
equivalence gate (`differential.run_differential`).

The oracle is *expensive and budgeted* (killer benchmark, T2); the
surrogate is cheap and unlimited. `Oracle` enforces that contract in one
place: every measurement spends budget, records evidence, and feeds the
surrogate — never the reverse.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from .state import CompilationState


@dataclass
class OracleResult:
    build_ok: bool
    binary_size: int | None
    binary_sha256: str | None
    runtime_ms: float | None
    telemetry: dict[str, float]
    timed_out: bool = False
    differential_ok: bool | None = None
    error: str = ""
    kept_path: str | None = None
    run_samples: list[float] | None = None


class OracleBudgetExhausted(RuntimeError):
    pass


class Oracle:
    """Real-measurement gateway with budget accounting."""

    def __init__(self, compiler: str = "gcc", test_cmd: str | None = None,
                 benchmark_runs: int = 3, timeout: float = 30.0,
                 budget: int = 20, capture_telemetry: bool = False,
                 evidence=None, keep_dir: str | Path | None = None) -> None:
        self.compiler = compiler
        self.test_cmd = test_cmd
        self.benchmark_runs = benchmark_runs
        self.timeout = timeout
        self.budget = budget
        self.used = 0
        self.capture_telemetry = capture_telemetry
        self.evidence = evidence
        self.keep_dir = Path(keep_dir) if keep_dir else None

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def measure(self, state: "CompilationState") -> OracleResult:
        """Spend 1 oracle view: build (+ run/benchmark) one state."""
        from self_compiler.compiler import compile_source
        from self_compiler.verifier import benchmark, verify

        if self.used >= self.budget:
            raise OracleBudgetExhausted(
                f"oracle budget exhausted ({self.used}/{self.budget})")
        self.used += 1
        suffix = ".exe" if os.name == "nt" else ""
        telemetry: dict[str, float] = {}
        error = ""
        with tempfile.TemporaryDirectory(prefix="vcc-oracle-") as work:
            src = Path(work) / "prog.c"
            binary = Path(work) / f"prog{suffix}"
            src.write_text(state.source_text, encoding="utf-8")
            tel_path = Path(work) / "tel.jsonl" if self.capture_telemetry else None
            flags = self._build_flags(state, tel_path)
            try:
                res = compile_source(str(src), compiler=self.compiler,
                                     flags=flags, output_path=str(binary),
                                     timeout=self.timeout)
            except RuntimeError as exc:
                return OracleResult(False, None, None, None, {},
                                    error=str(exc))
            if tel_path is not None and tel_path.exists():
                telemetry = self._read_telemetry(tel_path)
            size = binary.stat().st_size if res.ok and binary.exists() else None
            sha = (hashlib.sha256(binary.read_bytes()).hexdigest()[:16]
                   if size else None)
            runtime: float | None = None
            samples: list[float] | None = None
            if res.ok:
                if self.test_cmd:
                    vres = verify(str(binary), self.test_cmd, timeout=10.0)
                    runtime = vres.elapsed_ms if vres.passed else None
                    if not vres.passed:
                        error = vres.output[-500:]
                        return OracleResult(False, size, sha, None,
                                            telemetry, error=error)
                else:
                    bres = benchmark(str(binary), "{binary}",
                                     runs=self.benchmark_runs, warmups=2,
                                     timeout=10.0)
                    if bres.passed:
                        runtime = bres.median_ms
                        samples = [float(x) for x in bres.samples_ms]
                    else:
                        error = bres.reason
            else:
                error = (res.stderr or res.stdout)[-500:]
            kept: str | None = None
            if self.keep_dir is not None and res.ok and size:
                try:
                    self.keep_dir.mkdir(parents=True, exist_ok=True)
                    dest = self.keep_dir / f"{sha}{suffix}"
                    dest.write_bytes(binary.read_bytes())
                    kept = str(dest)
                except OSError:
                    kept = None
            result = OracleResult(res.ok, size, sha, runtime, telemetry,
                                  timed_out=res.timed_out, error=error,
                                  kept_path=kept, run_samples=samples)
        if self.evidence is not None:
            self.evidence.emit(
                "validate", "oracle-measurement",
                state_id=state.state_id, build_ok=result.build_ok,
                runtime_ms=result.runtime_ms, binary_size=result.binary_size,
                budget_used=self.used, budget_total=self.budget)
        return result

    def differential_gate(self, reference: Path, candidate: Path,
                          trials: int = 8) -> bool | None:
        """Equivalence gate between two binaries; None if unavailable."""
        try:
            from self_compiler.differential import run_differential
        except ImportError:
            return None
        try:
            report = run_differential(reference, candidate, trials=trials)
        except OSError:
            return None
        return bool(report.equivalent)

    # -- internals ------------------------------------------------------
    def _build_flags(self, state: "CompilationState",
                     tel_path: Path | None) -> list[str]:
        from self_compiler.gcc_native import _default_plugin

        flags = list(state.flags)
        if tel_path is None or state.policy_text:
            return flags
        try:
            plugin = _default_plugin(self.compiler)
        except Exception:
            return flags
        if plugin is None:
            return flags
        return [f"-fplugin={plugin}",
                f"-fplugin-arg-ai_native_c-output={tel_path}", *flags]

    @staticmethod
    def _read_telemetry(tel_path: Path) -> dict[str, float]:
        from .features import ir_from_telemetry

        events = []
        try:
            with open(tel_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return {}
        try:
            return ir_from_telemetry(events)
        except Exception:
            return {}


class StubOracle:
    """Deterministic in-memory oracle for tests and offline development.

    ``response`` maps ``(flags_key, policy_key, source_hash_prefix)``…
    in practice: a pure function of the state's flag preset with
    configurable noise-free runtimes. Behaves like `Oracle` for budget
    accounting but spends no wall-clock time.
    """

    def __init__(self, runtimes: dict[str, float] | None = None,
                 budget: int = 20, evidence=None,
                 response: Callable[[object], OracleResult] | None = None
                 ) -> None:
        self.runtimes = runtimes or {}
        self.budget = budget
        self.used = 0
        self.evidence = evidence
        self.response = response

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    def measure(self, state) -> OracleResult:
        if self.used >= self.budget:
            raise OracleBudgetExhausted(
                f"oracle budget exhausted ({self.used}/{self.budget})")
        self.used += 1
        if self.response is not None:
            return self.response(state)
        key = "|".join(state.flags)
        runtime = self.runtimes.get(key, 20.0)
        return OracleResult(True, 8192, "stub", runtime, {},
                            differential_ok=True)
