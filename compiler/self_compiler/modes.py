from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from .ai_engine import AIEngine, AIResponseError, AIUnavailableError
from .compiler import CompileResult, compile_source
from .diagnostics import (
    Diagnostic,
    diagnostic_from_output,
    parse_compiler_stderr,
    parse_gcc_json,
)
from .fixits import apply_compiler_fixits
from .journal import JournalWriter
from .locate import function_text, locate_enclosing_function, splice_function
from .logger import print_diff, print_error, print_step, print_success
from .patch import PatchManager
from .static_analysis import AnalysisResult, analyze
from .verifier import BenchmarkResult, VerifyResult, benchmark, verify


_MODES = {"repair", "secure", "self-correct", "optimize"}

# Whole-unit prompts are bounded by localizing the repair to the single
# function containing every diagnostic, but only when that actually buys
# a meaningful reduction: tiny units gain nothing and a "localized" span
# covering most of the file is not localization.
LOCALIZATION_MIN_UNIT_LINES = 120
LOCALIZATION_MAX_SPAN_FRACTION = 0.6


class RepairLoop:
    def __init__(
        self,
        source_path: str,
        compiler_cmd: str = "gcc",
        compiler_args: Sequence[str] = (),
        test_cmd: str | None = None,
        max_attempts: int = 3,
        *,
        benchmark_cmd: str | None = None,
        benchmark_runs: int = 5,
        min_improvement: float = 0.05,
        timeout: float = 30.0,
        output_path: str | None = None,
        ai: AIEngine | None = None,
        debugger: str | None = None,
        program_args: Sequence[str] = (),
        journal_path: str | None = None,
    ) -> None:
        self.source_path = str(Path(source_path).resolve())
        self.compiler_cmd = compiler_cmd
        self.compiler_args = list(compiler_args)
        self.test_cmd = test_cmd
        self.max_attempts = max_attempts
        self.benchmark_cmd = benchmark_cmd
        self.benchmark_runs = benchmark_runs
        self.min_improvement = min_improvement
        self.timeout = timeout
        self.output_path = output_path
        self.ai = ai or AIEngine()
        self.debugger = debugger
        self.program_args = tuple(program_args)
        self.journal = JournalWriter(journal_path)
        self._baseline_test: VerifyResult | None = None

    def run(self, mode: str) -> bool:
        self._baseline_test = None
        self.journal.emit("run-start", mode=mode, source=self.source_path)
        error = self._configuration_error(mode)
        if error:
            print_error(error)
            self.journal.emit("run-end", status="config-error", reason=error)
            return False

        source = Path(self.source_path)
        original = source.read_text(encoding="utf-8")
        executable_suffix = ".exe" if os.name == "nt" else ""

        with (
            tempfile.TemporaryDirectory(prefix="gcc-ai-repair-") as workspace,
            PatchManager(self.source_path) as patcher,
        ):
            workdir = Path(workspace)
            baseline_binary = workdir / f"baseline{executable_suffix}"
            print_step("Compiling the unchanged baseline in an isolated workspace...")
            baseline = self._compile(self.source_path, baseline_binary)

            prepared = self._prepare_baseline(mode, baseline, baseline_binary, workdir)
            if prepared is None:
                self.journal.emit("run-end", status="error", reason="baseline")
                return False
            if prepared is True:
                self._publish_binary(baseline_binary)
                self.journal.emit("run-end", status="success", note="no-op baseline")
                return True

            feedback, baseline_benchmark = prepared
            current = original
            current_path = self.source_path
            seen = {original}

            for attempt in range(1, self.max_attempts + 1):
                print_step(
                    f"Candidate {attempt}/{self.max_attempts}: evaluating structured repair evidence..."
                )
                candidate = self._next_candidate(
                    attempt, current, feedback, current_path, seen, mode
                )
                if candidate is None:
                    self.journal.emit(
                        "run-end", status="error", reason="candidate-generation"
                    )
                    return False

                if candidate in seen:
                    print_error(
                        "The candidate made no new change; feeding that failure back to the next attempt."
                    )
                    self.journal.emit("candidate-duplicate", attempt=attempt)
                    feedback = [
                        diagnostic_from_output(
                            self.source_path,
                            "Candidate duplicated an already rejected source revision.",
                        )
                    ]
                    continue
                seen.add(candidate)
                print_diff(current, candidate)
                candidate_source = patcher.stage(candidate)
                candidate_binary = workdir / f"candidate-{attempt}{executable_suffix}"

                result = self._compile(candidate_source, candidate_binary)
                if not result.ok:
                    print_error("Candidate did not compile.")
                    self.journal.emit("compile-failed", attempt=attempt)
                    feedback = self._compile_feedback(result, candidate_source)
                    current = candidate
                    current_path = candidate_source
                    continue

                gate_feedback = self._validate_candidate(
                    mode,
                    candidate_source,
                    candidate_binary,
                    workdir,
                    baseline_benchmark,
                )
                if gate_feedback:
                    self.journal.emit(
                        "gates-failed",
                        attempt=attempt,
                        findings=len(gate_feedback),
                    )
                    feedback = gate_feedback
                    current = candidate
                    current_path = candidate_source
                    continue

                print_step(
                    "All candidate gates passed; promoting atomically and rechecking the real source path..."
                )
                if source.read_text(encoding="utf-8") != original:
                    print_error(
                        "The source changed outside GCC-AI Repair during the run; refusing to overwrite it."
                    )
                    self.journal.emit(
                        "run-end",
                        status="error",
                        reason="external-source-change",
                    )
                    return False
                patcher.promote(candidate_source)
                self.journal.emit("promotion", attempt=attempt)
                final_binary = workdir / f"final{executable_suffix}"
                final = self._compile(self.source_path, final_binary)
                if not final.ok:
                    patcher.revert()
                    print_error(
                        "Post-promotion compilation failed; the original source was restored."
                    )
                    self.journal.emit(
                        "rollback", reason="post-promotion-compile-failed"
                    )
                    self.journal.emit("run-end", status="rolled-back")
                    return False
                final_feedback = self._validate_candidate(
                    mode,
                    self.source_path,
                    final_binary,
                    workdir,
                    baseline_benchmark,
                )
                if final_feedback:
                    patcher.revert()
                    print_error(
                        "A post-promotion verification gate failed; the original source was restored."
                    )
                    self.journal.emit("rollback", reason="post-promotion-gate-failed")
                    self.journal.emit("run-end", status="rolled-back")
                    return False

                patcher.commit()
                self._publish_binary(final_binary)
                self.journal.emit("run-end", status="success", accepted_attempt=attempt)
                print_success(
                    f"Accepted candidate {attempt}; source updated only after every configured gate passed."
                )
                return True

            print_error(
                "No candidate satisfied every verification gate; the source was left unchanged."
            )
            self.journal.emit("run-end", status="exhausted")
            return False

    def _next_candidate(
        self,
        attempt: int,
        current: str,
        feedback: list[Diagnostic],
        current_path: str,
        seen: set[str],
        mode: str,
    ) -> str | None:
        """Produce the next candidate source, preferring GCC fix-its.

        A fix-it whose result duplicates an already rejected revision is
        skipped rather than wasted as an attempt: the loop falls through to
        AI within the same iteration so bounded attempt budgets are spent
        only on new information.
        """
        compiler_fix = apply_compiler_fixits(current, feedback, current_path)
        if (
            compiler_fix.applied_edits
            and compiler_fix.source != current
            and compiler_fix.source not in seen
        ):
            print_step(
                f"Applying {compiler_fix.applied_edits} GCC-authored fix-it edit(s) "
                "before consulting AI."
            )
            self.journal.emit(
                "fixit-applied",
                attempt=attempt,
                edits=compiler_fix.applied_edits,
                source="gcc",
            )
            return compiler_fix.source

        if compiler_fix.applied_edits:
            print_step(
                "A GCC fix-it reproduced an already rejected revision; consulting AI instead."
            )
        else:
            print_step(
                f"No applicable GCC fix-it; requesting a minimal {mode} patch..."
            )
        try:
            return self._ai_candidate(current, feedback, current_path, attempt, mode)
        except (AIUnavailableError, AIResponseError) as exc:
            print_error(str(exc))
            self.journal.emit("ai-error", attempt=attempt, detail=str(exc))
            return None

    def _localization_target(
        self,
        source: str,
        feedback: list[Diagnostic],
        current_path: str,
    ):
        """Return the single function block covering every located error.

        Localization applies only when all line-bearing diagnostics fall
        inside exactly one top-level function and that function is a small
        fraction of a large translation unit; anything else returns None
        and the whole-unit path runs.
        """
        unit_lines = source.count("\n") + 1
        if unit_lines < LOCALIZATION_MIN_UNIT_LINES:
            return None
        resolved = str(Path(current_path).resolve())
        error_lines = sorted(
            {
                diagnostic.line
                for diagnostic in feedback
                if diagnostic.line >= 1
                and (
                    not diagnostic.file
                    or str(Path(diagnostic.file).resolve()) == resolved
                )
            }
        )
        if not error_lines:
            return None
        target = None
        for line in error_lines:
            block = locate_enclosing_function(source, line)
            if block is None:
                return None
            if target is None:
                target = block
            elif (block.start_offset, block.close_offset) != (
                target.start_offset,
                target.close_offset,
            ):
                return None
        assert target is not None
        span_lines = target.end_line - target.start_line + 1
        if span_lines > LOCALIZATION_MAX_SPAN_FRACTION * unit_lines:
            return None
        return target

    def _ai_candidate(
        self,
        current: str,
        feedback: list[Diagnostic],
        current_path: str,
        attempt: int,
        mode: str,
    ) -> str:
        """Consult AI localized to the defect's function when possible."""
        target = self._localization_target(current, feedback, current_path)
        if target is not None:
            try:
                replacement = self.ai.propose_localized_patch(
                    function_text(current, target),
                    feedback,
                    mode,
                    target.name or "<function>",
                )
                spliced = splice_function(current, target, replacement)
                if spliced is None:
                    raise AIResponseError(
                        "The localized candidate was unbalanced or empty."
                    )
                print_step(
                    f"Localized repair to function '{target.name or '<function>'}' "
                    f"({target.end_line - target.start_line + 1} lines); "
                    "splicing validated replacement back into the unit."
                )
                self.journal.emit(
                    "ai-requested",
                    attempt=attempt,
                    model=getattr(self.ai, "model_id", ""),
                    offline=bool(getattr(self.ai, "offline", False)),
                    strategy="localized-function",
                    function=target.name,
                )
                return spliced
            except (AIUnavailableError, AIResponseError) as exc:
                print_step(
                    "Localized repair unavailable; falling back to whole-unit "
                    f"prompt. ({exc})"
                )

        candidate = self.ai.propose_patch(current, feedback, mode)
        self.journal.emit(
            "ai-requested",
            attempt=attempt,
            model=getattr(self.ai, "model_id", ""),
            offline=bool(getattr(self.ai, "offline", False)),
            strategy="whole-unit",
        )
        return candidate

    def _configuration_error(self, mode: str) -> str | None:
        if mode not in _MODES:
            return f"Unknown mode '{mode}'."
        if not Path(self.source_path).is_file():
            return f"Source file does not exist: {self.source_path}"
        if self.max_attempts < 1:
            return "--max-attempts must be at least 1."
        if self.timeout <= 0:
            return "--timeout must be greater than zero."
        if mode == "self-correct" and not self.test_cmd and not self.debugger:
            return (
                "--self-correct requires --test-cmd or --debugger gdb with a "
                "reproducible failing execution."
            )
        if mode == "secure" and not self.test_cmd:
            return "--secure requires --test-cmd to guard behavior while repairing analyzer findings."
        if mode == "optimize" and not self.benchmark_cmd:
            return "--optimize requires --benchmark-cmd; compilation alone cannot prove an optimization."
        if mode == "optimize" and self.benchmark_runs < 3:
            return "--benchmark-runs must be at least 3 to reduce timing noise."
        if not 0 < self.min_improvement < 1:
            return "--min-improvement must be a fraction strictly between 0 and 1."
        return None

    def _prepare_baseline(
        self,
        mode: str,
        baseline: CompileResult,
        baseline_binary: Path,
        workdir: Path,
    ) -> tuple[list[Diagnostic], BenchmarkResult | None] | bool | None:
        if not baseline.ok:
            if mode != "repair":
                print_error(
                    f"The baseline does not compile; run --repair before --{mode}."
                )
                return None
            print_error("Baseline compilation failed; repair evidence captured.")
            return self._compile_feedback(baseline, self.source_path), None

        if mode == "repair":
            print_success("Baseline already compiles; no repair was needed.")
            return True

        if mode == "secure":
            baseline_test = self._verify(baseline_binary, self.source_path)
            if not baseline_test.passed:
                print_error(
                    "The security baseline fails --test-cmd; establish a passing regression check first."
                )
                return None
            self._baseline_test = baseline_test
            analysis = self._analyze(self.source_path, workdir / "baseline-analysis")
            if not analysis.available:
                print_error(
                    "No supported static analyzer is available; refusing to claim a secure result."
                )
                return None
            if not analysis.diagnostics:
                print_success(f"No findings from: {', '.join(analysis.tools_run)}.")
                return True
            print_error(
                f"Static analysis produced {len(analysis.diagnostics)} finding(s)."
            )
            return list(analysis.diagnostics), None

        if mode == "self-correct":
            execution = self._verify(baseline_binary, self.source_path)
            if execution.passed:
                print_success(
                    "The supplied runtime check already passes; nothing to self-correct."
                )
                return True
            return [self._runtime_feedback(execution)], None

        if self.test_cmd:
            correctness = self._verify(baseline_binary, self.source_path)
            if not correctness.passed:
                print_error(
                    "The baseline fails --test-cmd; establish correctness before optimizing."
                )
                return None
            self._baseline_test = correctness
        measured = self._benchmark(baseline_binary, self.source_path)
        if not measured.passed:
            print_error(
                f"Baseline benchmark is invalid: {measured.reason}: {measured.output}"
            )
            return None
        print_success(f"Baseline benchmark median: {measured.median_ms:.3f} ms.")
        self._warn_if_noisy(measured, "baseline")
        objective = diagnostic_from_output(
            self.source_path,
            f"Optimization target: beat baseline median {measured.median_ms:.3f} ms by at least "
            f"{self.min_improvement:.1%} while preserving benchmark output and exit status.",
        )
        return [objective], measured

    def _validate_candidate(
        self,
        mode: str,
        source_path: str,
        binary_path: Path,
        workdir: Path,
        baseline_benchmark: BenchmarkResult | None,
    ) -> list[Diagnostic]:
        if mode == "secure":
            analysis = self._analyze(
                source_path, workdir / f"analysis-{binary_path.stem}"
            )
            if not analysis.available:
                return [
                    diagnostic_from_output(
                        source_path, "No static analyzer was available."
                    )
                ]
            if analysis.diagnostics:
                print_error(
                    f"Candidate retains {len(analysis.diagnostics)} static-analysis finding(s)."
                )
                return list(analysis.diagnostics)

        if self.test_cmd:
            execution = self._verify(binary_path, source_path)
            if not execution.passed:
                print_error(
                    "Candidate failed the configured correctness/runtime check."
                )
                return [self._runtime_feedback(execution, source_path)]
            if self._baseline_test and (execution.returncode, execution.output) != (
                self._baseline_test.returncode,
                self._baseline_test.output,
            ):
                return [
                    diagnostic_from_output(
                        source_path,
                        "Candidate changed the observable output of the regression command.",
                    )
                ]

        if mode == "optimize":
            assert baseline_benchmark is not None
            measured = self._benchmark(binary_path, source_path)
            if not measured.passed:
                return [
                    diagnostic_from_output(
                        source_path,
                        f"Candidate benchmark invalid: {measured.reason}: {measured.output}",
                    )
                ]
            if (measured.returncode, measured.output) != (
                baseline_benchmark.returncode,
                baseline_benchmark.output,
            ):
                return [
                    diagnostic_from_output(
                        source_path,
                        "Candidate changed benchmark output or exit status.",
                    )
                ]
            improvement = 1.0 - measured.median_ms / baseline_benchmark.median_ms
            print_step(
                f"Candidate benchmark median: {measured.median_ms:.3f} ms ({improvement:+.1%})."
            )
            self._warn_if_noisy(measured, "candidate")
            if improvement < self.min_improvement:
                return [
                    diagnostic_from_output(
                        source_path,
                        f"Measured improvement {improvement:.1%} is below required {self.min_improvement:.1%}.",
                    )
                ]
        return []

    def _warn_if_noisy(self, result: BenchmarkResult, label: str) -> None:
        """Timing spread close to the required improvement makes a verdict noise."""
        if result.passed and result.relative_spread > 0.20:
            print_error(
                f"{label.capitalize()} benchmark spread is {result.relative_spread:.0%} "
                "of the median; treat the measured verdict as unreliable and consider "
                "--benchmark-runs above "
                f"{self.benchmark_runs}."
            )

    def _compile(self, source_path: str, output_path: Path) -> CompileResult:
        compiler_flags = list(self.compiler_args)
        compiler_name = Path(self.compiler_cmd).name.lower()
        if ("gcc" in compiler_name or "g++" in compiler_name) and not any(
            flag.startswith("-fdiagnostics-format=") for flag in compiler_flags
        ):
            compiler_flags.append("-fdiagnostics-format=json")
        result = compile_source(
            source_path,
            self.compiler_cmd,
            compiler_flags,
            output_path=str(output_path),
            timeout=self.timeout,
        )
        if result.ok:
            print_success(f"Compilation passed in {result.elapsed_ms:.0f} ms.")
        return result

    def _analyze(self, source_path: str, output_path: Path) -> AnalysisResult:
        print_step("Running available static analyzers...")
        return analyze(
            source_path,
            self.compiler_cmd,
            self.compiler_args,
            str(output_path),
            self.timeout,
        )

    def _verify(self, binary_path: Path, source_path: str) -> VerifyResult:
        print_step("Running the configured dynamic verification gate...")
        return verify(
            str(binary_path),
            self.test_cmd,
            self.timeout,
            source_path,
            self.program_args,
            self.debugger,
        )

    def _benchmark(self, binary_path: Path, source_path: str) -> BenchmarkResult:
        assert self.benchmark_cmd is not None
        return benchmark(
            str(binary_path),
            self.benchmark_cmd,
            self.benchmark_runs,
            1,
            self.timeout,
            source_path,
        )

    def _compile_feedback(
        self, result: CompileResult, source_path: str
    ) -> list[Diagnostic]:
        parsed = parse_gcc_json(result.stderr) or parse_compiler_stderr(result.stderr)
        if parsed:
            return parsed
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"Compiler exited {result.returncode}."
        )
        return [diagnostic_from_output(source_path, message)]

    def _runtime_feedback(
        self, result: VerifyResult, source_path: str | None = None
    ) -> Diagnostic:
        context = result.output or "no output"
        if result.crash_report:
            context += "\n" + result.crash_report.format_for_diagnostic()
        return diagnostic_from_output(
            source_path or self.source_path,
            f"Runtime verification exited {result.returncode} after {result.elapsed_ms:.1f} ms: {context}",
        )

    def _publish_binary(self, binary_path: Path) -> None:
        if not self.output_path:
            return
        destination = Path(self.output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        try:
            shutil.copy2(binary_path, temporary)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print_success(f"Published verified binary: {destination}")
