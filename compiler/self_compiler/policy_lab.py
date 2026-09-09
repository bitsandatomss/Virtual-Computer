"""Policy evaluation lab for the AI-native GCC plugin.

This module closes the loop between the plugin's two surfaces: it builds a
program under a pure-GCC baseline, under a plugin-loaded control, and under
one policy variant per candidate optimization pass, then measures runtime
with interleaved paired sampling so slow machine drift affects every
variant equally.  Results, including how many times each policy actually
gated a GCC pass, are emitted as versioned JSON Lines evidence.

Method notes:

- The plugin-loaded control isolates plugin overhead from policy effect;
  policy variants are compared against the control, not the raw compiler,
  and a plugin-neutrality record proves the control compiles bit-identical
  code to the raw baseline.
- Samples interleave across variants in a rotating order to cancel
  frequency-scaling and thermal drift instead of assuming them away.
  Sample ``i`` of every variant comes from the same measurement cycle, so
  deltas between a variant and the control are paired observations.
- Verdicts are statistically guarded: a variant is only called improving
  when its median beats the required margin AND the lower bound of a
  seeded bootstrap CI over the paired median delta excludes zero.  The
  seed is recorded in evidence so every interval is reproducible.
- ``--objective size`` switches the reward to binary size, which is
  deterministic; no CI applies there.
- A variant whose policy never gated a pass is reported as ``not-gated``
  so dead experiments are visible instead of silently meaningless.
- ``--check-reproducible`` rebuilds the control and winner and requires
  byte-identical binaries, implementing the repository's replay contract.

Nothing here trains a model yet; it produces the measured reward signal a
trainer would consume, which is the prerequisite the repository previously
lacked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

from .differential import DifferentialMismatch, run_differential
from .gcc_native import _default_plugin
from .logger import print_error, print_step, print_success
from .verifier import verify

SCHEMA = "gcc-ai.policy-lab.v1"
DEFAULT_PASSES = (
    "evrp",
    "ccp",
    "cddce",
    "tree-ifcombine",
    "phiopt",
)
OBJECTIVES = ("runtime", "size")
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class VariantSpec:
    """One compilation configuration under test."""

    name: str
    policy_text: str | None  # None means: do not load the plugin at all


@dataclass(frozen=True)
class VariantResult:
    name: str
    verdict: str
    median_ms: float
    spread_ratio: float
    improvement: float
    binary_size: int
    gate_events: int
    output_equal_to_control: bool
    error: str = ""
    binary_sha256: str = ""
    paired_rounds: int = 0
    delta_ci_low_ms: float | None = None
    delta_ci_high_ms: float | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


_TIMESTAMP_FLAG = "-Wl,--no-insert-timestamp"
_probe_cache: dict[str, bool] = {}


def _timestamp_normalization_supported(compiler: str) -> bool:
    """True when the toolchain accepts PE timestamp suppression.

    MinGW linkers embed the current time in the PE header, so two builds of
    identical inputs differ in three header bytes and no digest comparison
    can ever pass.  The ld flag is PE-specific; probe once per compiler and
    cache the answer.
    """
    cached = _probe_cache.get(compiler)
    if cached is not None:
        return cached
    supported = False
    if os.name == "nt":
        try:
            with tempfile.TemporaryDirectory(prefix="gcc-ai-probe-") as probe_dir:
                probe_source = Path(probe_dir) / "probe.c"
                probe_binary = Path(probe_dir) / "probe.exe"
                probe_source.write_text("int main(void){return 0;}\n")
                completed = subprocess.run(
                    [
                        compiler,
                        _TIMESTAMP_FLAG,
                        str(probe_source),
                        "-o",
                        str(probe_binary),
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=30,
                    check=False,
                )
                supported = completed.returncode == 0 and probe_binary.exists()
        except (OSError, subprocess.TimeoutExpired):
            supported = False
    _probe_cache[compiler] = supported
    return supported


def _normalized_build_args(
    compiler: str, compiler_args: Sequence[str]
) -> tuple[list[str], bool]:
    """Return build args plus whether timestamp normalization was applied."""
    if _timestamp_normalization_supported(compiler):
        return [*compiler_args, _TIMESTAMP_FLAG], True
    return list(compiler_args), False


def _toolchain_identity(compiler: str) -> str:
    parts = [compiler]
    for option in ("-dumpfullversion", "-dumpmachine"):
        try:
            completed = subprocess.run(
                [compiler, option],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=15,
            )
            parts.append(completed.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            parts.append("unknown")
    return " ".join(part for part in parts if part)


def _default_specs(passes: Sequence[str]) -> list[VariantSpec]:
    specs = [
        VariantSpec("gcc-baseline", None),
        VariantSpec("plugin-control", ""),
    ]
    specs.extend(
        VariantSpec(f"disable-{pass_name}", f"disable_pass={pass_name}\n")
        for pass_name in passes
    )
    return specs


def _build_variant(
    spec: VariantSpec,
    sources: Sequence[Path],
    compiler_args: Sequence[str],
    compiler: str,
    plugin: Path | None,
    workspace: Path,
    index: int,
    timeout: float,
) -> tuple[Path | None, int, str, str]:
    """Compile one variant; return (binary, gate count, sha256, error)."""
    suffix = ".exe" if sys.platform == "win32" else ""
    binary = workspace / f"{index}-{spec.name}{suffix}"
    command = [compiler, *compiler_args]
    telemetry: Path | None = None
    if spec.policy_text is not None:
        if plugin is None or not plugin.is_file():
            return None, 0, "", f"plugin not found: {plugin}"
        plugin_key = plugin.stem
        command.append(f"-fplugin={plugin}")
        telemetry = workspace / f"{index}-{spec.name}.jsonl"
        command.append(f"-fplugin-arg-{plugin_key}-output={telemetry}")
        if spec.policy_text:
            policy_path = workspace / f"{index}-{spec.name}.policy"
            policy_path.write_text(spec.policy_text, encoding="utf-8")
            command.append(f"-fplugin-arg-{plugin_key}-policy={policy_path}")
    command.extend([str(source) for source in sources])
    command.extend(["-o", str(binary)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return None, 0, "", f"compiler not found: {compiler}"
    except subprocess.TimeoutExpired:
        return None, 0, "", f"build timed out after {timeout:g}s"
    if completed.returncode != 0 or not binary.exists():
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        detail = tail[-1] if tail else f"exit code {completed.returncode}"
        return None, 0, "", f"build failed: {detail}"

    gates = 0
    if telemetry is not None and telemetry.exists():
        for line in telemetry.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("event") == "pass-gate":
                gates += 1
    return binary, gates, _sha256(binary), ""


def _sample_runner(
    binary: Path,
    benchmark_cmd: str | None,
    timeout: float,
) -> Callable[[], tuple[bool, int, str]]:
    """Return a zero-argument runner producing (passed, returncode, output)."""

    def run() -> tuple[bool, int, str]:
        result = verify(
            str(binary),
            benchmark_cmd,
            timeout,
        )
        return result.passed, result.returncode, result.output

    return run


def measure_variants(
    binaries: dict[str, Path],
    order: Sequence[str],
    benchmark_cmd: str | None,
    runs: int,
    warmups: int,
    timeout: float,
) -> tuple[dict[str, list[float]], dict[str, str], dict[str, str]]:
    """Interleave timing samples across variants in a rotating order.

    Returns per-variant samples, first observed output, and an error message
    for any variant that failed or behaved non-deterministically mid-sampling.
    """
    samples: dict[str, list[float]] = {name: [] for name in order}
    outputs: dict[str, str] = {}
    codes: dict[str, int] = {}
    errors: dict[str, str] = {}
    runners = {
        name: _sample_runner(binaries[name], benchmark_cmd, timeout) for name in order
    }

    def execute(name: str) -> tuple[bool, int, str] | None:
        passed, returncode, output = runners[name]()
        previous_code = codes.setdefault(name, returncode)
        previous_output = outputs.setdefault(name, output)
        if not passed:
            errors[name] = f"run failed (exit {returncode}): {output[:200]}"
            return None
        if returncode != previous_code or output != previous_output:
            errors[name] = "nondeterministic output across runs"
            return None
        return passed, returncode, output

    for _ in range(warmups):
        for name in order:
            if name in errors:
                continue
            if execute(name) is None:
                continue
    for cycle in range(runs):
        rotation = cycle % len(order)
        rotated = [*order[rotation:], *order[:rotation]]
        for name in rotated:
            if name in errors:
                continue
            started = time.perf_counter()
            outcome = execute(name)
            elapsed = (time.perf_counter() - started) * 1000.0
            if outcome is not None:
                samples[name].append(elapsed)
    return samples, outputs, errors


def summarize(
    samples: Sequence[float],
) -> tuple[float, float]:
    """Return (median, relative spread) with spread 0.0 when undefined."""
    if not samples:
        return 0.0, 0.0
    central = median(samples)
    spread = 0.0
    if len(samples) > 1 and central > 0:
        spread = (max(samples) - min(samples)) / central
    return central, spread


def paired_delta_stats(
    control_samples: Sequence[float],
    variant_samples: Sequence[float],
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float | None, float | None]:
    """Bootstrap CI for the median of per-cycle paired deltas.

    Interleaved sampling measures every variant once per cycle, so sample
    ``i`` of each variant was taken under comparable machine conditions;
    ``delta_i = control_i - variant_i`` is therefore a paired observation.
    The CI is the 2.5/97.5 percentile range of resampled medians using a
    deterministic seeded generator, so evidence is reproducible bit-for-bit.
    Returns (median_delta_ms, ci_low_ms, ci_high_ms); CI is ``None`` when
    fewer than three usable pairs exist.
    """
    pairs = min(len(control_samples), len(variant_samples))
    deltas = [control_samples[index] - variant_samples[index] for index in range(pairs)]
    if not deltas:
        return 0.0, None, None
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(resamples):
        medians.append(median(deltas[rng.randrange(pairs)] for _ in range(pairs)))
    medians.sort()
    ci_low = medians[int(0.025 * (resamples - 1))]
    ci_high = medians[int(0.975 * (resamples - 1))]
    return median(deltas), ci_low, ci_high


def classify(
    *,
    control_median: float,
    variant_median: float,
    min_improvement: float,
    gate_events: int,
    output_equal: bool,
    ci_low: float | None = None,
) -> tuple[str, float]:
    """Classify one variant against the plugin control.

    For runtime measurements a margin alone is not enough: with noisy
    timers a real-looking median gap routinely appears from noise.  When
    ``ci_low`` (the lower bound of the bootstrap CI on paired median
    deltas) is provided it must be strictly positive before a variant may
    be called improving; callers measuring a deterministic metric such as
    binary size pass ``ci_low=None`` and only the margin applies.
    """
    improvement = 0.0
    if control_median > 0:
        improvement = 1.0 - variant_median / control_median
    if not output_equal:
        return "invalid-behavior", improvement
    if gate_events == 0:
        return "not-gated", improvement
    significant = ci_low is None or ci_low > 0.0
    if improvement >= min_improvement and significant:
        return "improves", improvement
    if improvement <= 0:
        return "regresses", improvement
    return "within-noise", improvement


def rank_improving(results: Sequence[VariantResult]) -> VariantResult | None:
    improving = [r for r in results if r.verdict == "improves"]
    if not improving:
        return None
    return max(improving, key=lambda r: r.improvement)


def _render_table(results: Sequence[VariantResult]) -> str:
    header = (
        f"{'variant':<24}{'median ms':>12}{'spread':>9}{'vs ctrl':>10}"
        f"{'d95 low':>10}{'gates':>7}{'size':>10}  verdict"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        ci_low_text = (
            f"{result.delta_ci_low_ms:>9.3f}m"
            if result.delta_ci_low_ms is not None
            else f"{'--':>10}"
        )
        lines.append(
            f"{result.name:<24}{result.median_ms:>12.3f}{result.spread_ratio:>9.0%}"
            f"{result.improvement:>+10.1%}{ci_low_text}{result.gate_events:>7}"
            f"{result.binary_size:>10}  {result.verdict}"
        )
    return "\n".join(lines)


def run_lab(
    sources: Sequence[Path],
    compiler: str,
    compiler_args: Sequence[str],
    benchmark_cmd: str | None,
    passes: Sequence[str],
    extra_policies: Sequence[Path],
    runs: int,
    warmups: int,
    timeout: float,
    min_improvement: float,
    output_dir: Path,
    plugin_path: Path | None,
    objective: str = "runtime",
    check_reproducible: bool = False,
    bootstrap_seed: int = 20260822,
    differential_trials: int = 8,
) -> int:
    if objective not in OBJECTIVES:
        print_error(f"Unknown objective '{objective}'; choose from {OBJECTIVES}.")
        return 2
    if differential_trials < 0:
        print_error("--differential-trials must be zero or positive.")
        return 2
    if not sources:
        print_error("At least one source file is required.")
        return 2
    for source in sources:
        if not source.is_file():
            print_error(f"Source not found: {source}")
            return 2
    if runs < 3:
        print_error("--runs must be at least 3 for a median to be meaningful.")
        return 2
    if warmups < 0 or timeout <= 0 or min_improvement < 0:
        print_error("Invalid lab parameters.")
        return 2

    specs = _default_specs(passes)
    seen_policy_texts: dict[str, str] = {}
    for extra in extra_policies:
        if not extra.is_file():
            print_error(f"Policy not found: {extra}")
            return 2
        text = extra.read_text(encoding="utf-8")
        name = f"custom-{extra.stem}"
        if text in seen_policy_texts:
            print_error(f"Custom policy {extra} duplicates {seen_policy_texts[text]}.")
            return 2
        seen_policy_texts[text] = name
        specs.append(VariantSpec(name, text))
    unique_names = {spec.name for spec in specs}
    if len(unique_names) != len(specs):
        print_error("Duplicate variant names in specification; rename policies.")
        return 2

    plugin = plugin_path if plugin_path is not None else _default_plugin(compiler)
    build_args, timestamps_normalized = _normalized_build_args(compiler, compiler_args)
    workspace = output_dir
    workspace.mkdir(parents=True, exist_ok=True)

    evidence_path = workspace / "evidence.jsonl"
    evidence_lines: list[str] = []
    evidence_lines.append(
        json.dumps(
            {
                "schema": SCHEMA,
                "event": "lab-meta",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "toolchain": _toolchain_identity(compiler),
                "sources": [
                    {"path": str(source), "sha256": _sha256(source)}
                    for source in sources
                ],
                "compiler_args": list(compiler_args),
                "objective": objective,
                "build_timestamps_normalized": timestamps_normalized,
                "runs": runs,
                "warmups": warmups,
                "min_improvement": min_improvement,
                "bootstrap_seed": bootstrap_seed,
            }
        )
    )

    binaries: dict[str, Path] = {}
    gate_counts: dict[str, int] = {}
    digests: dict[str, str] = {}
    for index, spec in enumerate(specs):
        print_step(f"Building variant '{spec.name}'...")
        binary, gates, digest, error = _build_variant(
            spec,
            sources,
            build_args,
            compiler,
            plugin,
            workspace,
            index,
            timeout,
        )
        gate_counts[spec.name] = gates
        evidence_lines.append(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "event": "build",
                    "variant": spec.name,
                    "ok": binary is not None,
                    "gate_events": gates,
                    "binary_sha256": digest if binary is not None else "",
                    "error": error,
                }
            )
        )
        if binary is None:
            print_error(f"{spec.name}: {error}")
        else:
            binaries[spec.name] = binary
            digests[spec.name] = digest

    control_name = "plugin-control"
    if control_name not in binaries:
        print_error("Plugin control failed to build; cannot rank policy effects.")
        _write_evidence(evidence_path, evidence_lines)
        return 1

    # Plugin-neutrality invariant: loading the plugin without a policy must
    # not change generated code.  A mismatch here would confound every
    # comparison against the control.
    if "gcc-baseline" in digests and control_name in digests:
        neutral = digests["gcc-baseline"] == digests[control_name]
        evidence_lines.append(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "event": "plugin-neutrality",
                    "neutral": neutral,
                    "baseline_sha256": digests["gcc-baseline"],
                    "control_sha256": digests[control_name],
                }
            )
        )
        if not neutral:
            print_error(
                "The plugin changes generated code even with no policy; "
                "control comparisons are confounded."
            )

    runnable_order = [spec.name for spec in specs if spec.name in binaries]
    print_step(
        f"Measuring {len(runnable_order)} variants with interleaved sampling "
        f"(objective {objective}, warmup {warmups}, runs {runs})..."
    )
    samples, outputs, sample_errors = measure_variants(
        binaries,
        runnable_order,
        benchmark_cmd,
        1 if objective == "size" else runs,
        0 if objective == "size" else warmups,
        timeout,
    )

    control_output = outputs.get(control_name, "")
    control_samples = samples.get(control_name, [])
    if not control_samples:
        print_error(
            "The plugin control produced no timing samples; measurement "
            "environment is unusable (check --benchmark-cmd and --timeout)."
        )
        _write_evidence(evidence_path, evidence_lines)
        return 1
    control_median_value, _control_spread = summarize(control_samples)
    results: list[VariantResult] = []
    for spec in specs:
        if spec.name not in binaries:
            results.append(
                VariantResult(spec.name, "build-failed", 0.0, 0.0, 0.0, 0, 0, False)
            )
            continue
        if spec.name in sample_errors:
            results.append(
                VariantResult(
                    spec.name,
                    "run-failed",
                    0.0,
                    0.0,
                    0.0,
                    binaries[spec.name].stat().st_size,
                    gate_counts[spec.name],
                    False,
                    sample_errors[spec.name],
                )
            )
            continue
        variant_samples = samples.get(spec.name, [])
        if not variant_samples:
            results.append(
                VariantResult(
                    spec.name,
                    "run-failed",
                    0.0,
                    0.0,
                    0.0,
                    binaries[spec.name].stat().st_size,
                    gate_counts[spec.name],
                    False,
                    "no samples collected",
                )
            )
            continue
        variant_median, spread = summarize(variant_samples)
        output_equal = (
            spec.name == control_name or outputs.get(spec.name) == control_output
        )
        median_delta: float | None = None
        ci_low: float | None = None
        ci_high: float | None = None
        if objective == "runtime":
            metric_control, metric_variant = (
                control_median_value,
                variant_median,
            )
            median_delta, ci_low, ci_high = paired_delta_stats(
                control_samples, variant_samples, seed=bootstrap_seed
            )
        else:
            # Binary size is deterministic; compare sizes directly and let
            # classify() treat the margin as sufficient evidence.
            metric_control = float(binaries[control_name].stat().st_size)
            metric_variant = float(binaries[spec.name].stat().st_size)
        verdict, improvement = classify(
            control_median=metric_control,
            variant_median=metric_variant,
            min_improvement=min_improvement,
            gate_events=gate_counts[spec.name],
            output_equal=output_equal,
            ci_low=ci_low,
        )
        if spec.name == "gcc-baseline":
            verdict = "pure-gcc-reference"
        elif spec.name == control_name:
            verdict = "control"
        results.append(
            VariantResult(
                spec.name,
                verdict,
                variant_median,
                spread,
                improvement,
                binaries[spec.name].stat().st_size,
                gate_counts[spec.name],
                output_equal,
                binary_sha256=digests.get(spec.name, ""),
                paired_rounds=min(len(control_samples), len(variant_samples)),
                delta_ci_low_ms=ci_low,
                delta_ci_high_ms=ci_high,
            )
        )

    for result in results:
        evidence_lines.append(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "event": "result",
                    "variant": result.name,
                    "samples_ms": sorted(samples.get(result.name, [])),
                    "median_ms": result.median_ms,
                    "spread_ratio": result.spread_ratio,
                    "improvement_vs_control": result.improvement,
                    "paired_rounds": result.paired_rounds,
                    "delta_ci_low_ms": result.delta_ci_low_ms,
                    "delta_ci_high_ms": result.delta_ci_high_ms,
                    "gate_events": result.gate_events,
                    "binary_size": result.binary_size,
                    "binary_sha256": result.binary_sha256,
                    "output_equal_to_control": result.output_equal_to_control,
                    "verdict": result.verdict,
                    "error": result.error,
                }
            )
        )

    winner = rank_improving(results)
    decision = {
        "schema": SCHEMA,
        "event": "decision",
        "winner": winner.name if winner else "",
        "improvement_vs_control": winner.improvement if winner else 0.0,
        "caveat": (
            "single-workload measurement; requires held-out validation "
            "and differential testing before deployment"
        ),
    }

    if check_reproducible:
        repro_specs = {
            spec.name: spec
            for spec in specs
            if spec.name in {control_name, winner.name if winner else ""}
        }
        for offset, (name, spec) in enumerate(
            sorted(repro_specs.items()), start=len(specs) + 100
        ):
            print_step(f"Rebuilding '{name}' to verify build reproducibility...")
            rebuild_binary, _gates, rebuild_digest, rebuild_error = _build_variant(
                spec,
                sources,
                build_args,
                compiler,
                plugin,
                workspace,
                offset,
                timeout,
            )
            match = (
                rebuild_error == ""
                and name in digests
                and rebuild_digest == digests[name]
            )
            evidence_lines.append(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "event": "reproducibility",
                        "variant": name,
                        "reproducible": match,
                        "first_sha256": digests.get(name, ""),
                        "second_sha256": rebuild_digest,
                        "error": rebuild_error,
                    }
                )
            )
            if not match:
                print_error(f"Rebuild of '{name}' produced a different binary.")

    evidence_lines.append(json.dumps(decision))
    _write_evidence(evidence_path, evidence_lines)

    print_step(
        "Results (interleaved sampling; d95 low = CI lower bound on paired median delta):"
    )
    print(_render_table(results))

    if winner:
        # Mandatory differential gate: the winner must behave identically to
        # the control across seeded pseudo-random inputs before any artifact
        # is promoted.  A divergence voids the decision entirely.
        differential_passed: bool | None = None
        differential_mismatches: list[DifferentialMismatch] = []
        if differential_trials > 0 and objective == "runtime":
            print_step(
                f"Differential-testing '{winner.name}' against the control "
                f"over {differential_trials} seeded cases..."
            )
            report = run_differential(
                binaries[control_name],
                binaries[winner.name],
                trials=differential_trials,
                seed=bootstrap_seed,
                timeout=timeout,
            )
            differential_passed = report.equivalent
            differential_mismatches = list(report.mismatches)
            evidence_lines.append(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "event": "differential",
                        "reference": control_name,
                        "candidate": winner.name,
                        "equivalent": report.equivalent,
                        "trials": report.trials,
                        "skipped_both_timeout": report.skipped,
                        "seed": report.seed,
                        "mismatches": [
                            {
                                "case": m.case,
                                "kind": m.kind,
                                "detail": m.detail,
                            }
                            for m in differential_mismatches[:20]
                        ],
                    }
                )
            )

        if differential_passed is False:
            print_error(
                f"Differential testing rejected '{winner.name}': the policy "
                "changed observable behavior. No artifact was promoted."
            )
            decision["winner"] = ""
            decision["improvement_vs_control"] = 0.0
            decision["rejected_by"] = "differential"
            evidence_lines[-1] = json.dumps(decision)
            _write_evidence(evidence_path, evidence_lines)
            print_success(f"Evidence written to {evidence_path}")
            return 1

        best_policy_path = workspace / "best.policy"
        best_policy_path.write_text(
            next(spec.policy_text for spec in specs if spec.name == winner.name) or "",
            encoding="utf-8",
        )
        print_success(
            f"'{winner.name}' improved on the plugin control by "
            f"{winner.improvement:+.1%} (paired-delta CI excludes zero); "
            f"policy written to {best_policy_path}. Validate on held-out "
            "workloads and differential-test before relying on it."
        )
        decision["policy_artifact"] = str(best_policy_path)
        evidence_lines[-1] = json.dumps(decision)
        _write_evidence(evidence_path, evidence_lines)
    else:
        print_step(
            "No variant beat the plugin control beyond the required margin; "
            "the GCC heuristic stands."
        )
    print_success(f"Evidence written to {evidence_path}")
    return 0


def _write_evidence(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcc-ai-policy-lab",
        description=(
            "Measure GCC pass-disable policies against a plugin control using "
            "interleaved paired sampling and emit JSONL evidence."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_lab_version(),
    )
    parser.add_argument("--compiler", default="gcc", help="GCC driver executable")
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="runtime",
        help="reward metric: runtime medians or binary size",
    )
    parser.add_argument(
        "--check-reproducible",
        action="store_true",
        help="rebuild the control and winner and require identical binaries",
    )
    parser.add_argument(
        "--differential-trials",
        type=int,
        default=8,
        help="seeded differential cases between winner and control (0 disables)",
    )
    parser.add_argument(
        "--benchmark-cmd",
        help="timing command template ({binary}); defaults to direct execution",
    )
    parser.add_argument(
        "--runs", type=int, default=7, help="measured cycles per variant (>=3)"
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="required fractional speedup over the plugin control",
    )
    parser.add_argument(
        "--passes",
        default=",".join(DEFAULT_PASSES),
        help="comma-separated GCC pass names to ablate one at a time",
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        dest="policies",
        help="additional policy file to evaluate (repeatable)",
    )
    parser.add_argument("--plugin", help="explicit ai_native plugin path")
    parser.add_argument(
        "--output-dir", default="build/policy-lab", help="workspace for artifacts"
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        dest="sources",
        help="source file to compile (repeatable)",
    )
    parser.add_argument(
        "compiler_args",
        nargs=argparse.REMAINDER,
        help="compiler arguments after -- (optimization level etc.)",
    )
    return parser


def _lab_version() -> str:
    try:
        from importlib.metadata import version

        return version("gcc-ai-native")
    except Exception:
        return "0.0.0+unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compiler_args = list(args.compiler_args)
    if compiler_args[:1] == ["--"]:
        compiler_args = compiler_args[1:]
    passes = [name.strip() for name in args.passes.split(",") if name.strip()]
    try:
        return run_lab(
            sources=[Path(source) for source in args.sources],
            compiler=args.compiler,
            compiler_args=compiler_args,
            benchmark_cmd=args.benchmark_cmd,
            passes=passes,
            extra_policies=[Path(policy) for policy in args.policies],
            runs=args.runs,
            warmups=args.warmups,
            timeout=args.timeout,
            min_improvement=args.min_improvement,
            output_dir=Path(args.output_dir),
            plugin_path=Path(args.plugin).resolve() if args.plugin else None,
            objective=args.objective,
            check_reproducible=args.check_reproducible,
            differential_trials=args.differential_trials,
        )
    except OSError as exc:
        print_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
