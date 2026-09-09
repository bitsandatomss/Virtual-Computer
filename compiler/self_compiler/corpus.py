"""Multi-workload corpus builder for policy research.

A single-program lab run can never satisfy the repository's held-out
workload requirement.  This tool applies the policy lab's exact
methodology—identical build steps, interleaved paired sampling,
bootstrap-guarded classification—across many programs and emits one
versioned dataset combining per-function IR features with measured
policy outcomes.  The dataset is the input contract for rule synthesis
and any future trained selector.

Dataset schema ``gcc-ai.corpus.v1``: one ``corpus-meta`` record followed
by one record per workload containing the source digest, raw per-function
telemetry captured during the control build, deterministic feature
aggregates, and every variant's outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .logger import print_error, print_step, print_success
from .policy_lab import (
    BOOTSTRAP_RESAMPLES,
    _build_variant,
    _default_specs,
    _default_plugin,
    _normalized_build_args,
    _sha256,
    _toolchain_identity,
    classify,
    measure_variants,
    paired_delta_stats,
    summarize,
)

SCHEMA = "gcc-ai.corpus.v1"
FEATURE_FIELDS = (
    "basic_blocks",
    "gimple_statements",
    "phi_nodes",
    "calls",
    "branches",
    "edges",
    "memory_reads",
    "memory_writes",
    "float_ops",
    # Structural topology summaries derived from dominance information;
    # these let synthesis reason about shape, not just size.
    "back_edges",
    "max_loop_depth",
    "dominator_height",
    "cyclomatic_complexity",
    "max_out_degree",
)


def feature_aggregates(function_events: Sequence[dict]) -> dict[str, float]:
    """Deterministic program-level aggregates of per-function IR features."""
    aggregates: dict[str, float] = {"function_count": len(function_events)}
    for field in FEATURE_FIELDS:
        values = [float(event.get(field, 0)) for event in function_events]
        aggregates[f"total_{field}"] = sum(values)
        aggregates[f"max_{field}"] = max(values) if values else 0.0
    return aggregates


def load_function_events(telemetry_path: Path | None) -> list[dict]:
    if telemetry_path is None or not telemetry_path.is_file():
        return []
    events: list[dict] = []
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "function-ir":
            events.append(event)
    return events


def _workload_name(source: Path, taken: set[str]) -> str:
    base = source.stem or "program"
    name = base
    counter = 1
    while name in taken:
        counter += 1
        name = f"{base}-{counter}"
    taken.add(name)
    return name


def build_corpus(
    sources: Sequence[Path],
    compiler: str,
    compiler_args: Sequence[str],
    benchmark_cmd: str | None,
    passes: Sequence[str],
    runs: int,
    warmups: int,
    timeout: float,
    min_improvement: float,
    output_path: Path,
    plugin_path: Path | None = None,
    seed: int = 20260822,
) -> int:
    if not sources:
        print_error("At least one --source is required.")
        return 2
    for source in sources:
        if not source.is_file():
            print_error(f"Source not found: {source}")
            return 2
    if runs < 3:
        print_error("--runs must be at least 3.")
        return 2
    if warmups < 0 or timeout <= 0 or min_improvement < 0:
        print_error("Invalid corpus parameters.")
        return 2

    plugin = plugin_path if plugin_path is not None else _default_plugin(compiler)
    if plugin is None or not Path(plugin).is_file():
        print_error(f"Plugin not found: {plugin}; run gcc-ai-build-plugin first.")
        return 2

    specs = _default_specs(passes)
    build_args, timestamps_normalized = _normalized_build_args(compiler, compiler_args)
    workspace_root = output_path.resolve().parent / "corpus-workloads"
    workspace_root.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = [
        {
            "schema": SCHEMA,
            "event": "corpus-meta",
            "toolchain": _toolchain_identity(compiler),
            "objective": "runtime",
            "compiler_args": list(compiler_args),
            "build_timestamps_normalized": timestamps_normalized,
            "runs": runs,
            "warmups": warmups,
            "min_improvement": min_improvement,
            "bootstrap_seed": seed,
            "passes": list(passes),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        }
    ]

    taken_names: set[str] = set()
    usable_workloads = 0
    for index, source in enumerate(sources):
        name = _workload_name(source, taken_names)
        workspace = workspace_root / f"{index:03d}-{name}"
        workspace.mkdir(parents=True, exist_ok=True)
        print_step(f"[{name}] building {len(specs)} variants...")
        binaries: dict[str, Path] = {}
        gate_counts: dict[str, int] = {}
        digests: dict[str, str] = {}
        control_telemetry: Path | None = None
        for variant_index, spec in enumerate(specs):
            binary, gates, digest, error = _build_variant(
                spec,
                [source],
                build_args,
                compiler,
                Path(plugin),
                workspace,
                variant_index,
                timeout,
            )
            gate_counts[spec.name] = gates
            if binary is None:
                print_error(f"[{name}] {spec.name}: {error}")
                continue
            binaries[spec.name] = binary
            digests[spec.name] = digest
            if spec.name == "plugin-control":
                control_telemetry = workspace / f"{variant_index}-{spec.name}.jsonl"

        control_name = "plugin-control"
        function_events = load_function_events(control_telemetry)
        features = feature_aggregates(function_events)
        record: dict = {
            "schema": SCHEMA,
            "event": "workload",
            "name": name,
            "source": {
                "path": str(source),
                "sha256": _sha256(source),
            },
            "functions": function_events,
            "features": features,
            "outcomes": {},
            "errors": [],
        }

        if control_name not in binaries:
            record["errors"].append("control build failed; workload unusable")
            records.append(record)
            continue
        usable_workloads += 1

        runnable_order = [spec.name for spec in specs if spec.name in binaries]
        samples, outputs, sample_errors = measure_variants(
            binaries, runnable_order, benchmark_cmd, runs, warmups, timeout
        )
        control_samples = samples.get(control_name, [])
        if not control_samples:
            record["errors"].append("control produced no samples")
            records.append(record)
            continue
        control_median_value, _spread = summarize(control_samples)
        control_output = outputs.get(control_name, "")

        for spec in specs:
            if spec.name not in binaries:
                continue
            if spec.name in sample_errors:
                record["outcomes"][spec.name] = {
                    "verdict": "run-failed",
                    "error": sample_errors[spec.name],
                }
                continue
            variant_samples = samples.get(spec.name, [])
            if not variant_samples:
                record["outcomes"][spec.name] = {
                    "verdict": "run-failed",
                    "error": "no samples collected",
                }
                continue
            variant_median, spread = summarize(variant_samples)
            median_delta, ci_low, ci_high = paired_delta_stats(
                control_samples, variant_samples, seed=seed
            )
            verdict, improvement = classify(
                control_median=control_median_value,
                variant_median=variant_median,
                min_improvement=min_improvement,
                gate_events=gate_counts[spec.name],
                output_equal=(
                    spec.name == control_name
                    or outputs.get(spec.name) == control_output
                ),
                ci_low=ci_low,
            )
            if spec.name == "gcc-baseline":
                verdict = "pure-gcc-reference"
            elif spec.name == control_name:
                verdict = "control"
            record["outcomes"][spec.name] = {
                "verdict": verdict,
                "improvement_vs_control": improvement,
                "median_ms": variant_median,
                "median_delta_ms": median_delta,
                "delta_ci_low_ms": ci_low,
                "delta_ci_high_ms": ci_high,
                "paired_rounds": min(len(control_samples), len(variant_samples)),
                "spread_ratio": spread,
                "gate_events": gate_counts[spec.name],
                "binary_size": binaries[spec.name].stat().st_size,
                "binary_sha256": digests.get(spec.name, ""),
                "output_equal_to_control": (
                    spec.name == control_name
                    or outputs.get(spec.name) == control_output
                ),
            }
        improving = sorted(
            outcome_name
            for outcome_name, outcome in record["outcomes"].items()
            if outcome.get("verdict") == "improves"
        )
        print_step(
            f"[{name}] measured: "
            + ", ".join(
                f"{outcome_name}={record['outcomes'][outcome_name]['verdict']}"
                for outcome_name in sorted(record["outcomes"])
            )
        )
        if improving:
            print_success(f"[{name}] improving variants: {', '.join(improving)}")
        records.append(record)

    output_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    if usable_workloads == 0:
        print_error("No workload produced usable measurements.")
        return 1
    print_success(
        f"Corpus written to {output_path} ({usable_workloads}/{len(sources)} "
        "usable workloads)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcc-ai-corpus",
        description=(
            "Build a versioned multi-workload dataset of IR features and "
            "policy outcomes using the policy lab methodology."
        ),
    )
    parser.add_argument("--version", action="version", version=_corpus_version())
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--benchmark-cmd", help="timing template ({binary})")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument(
        "--passes",
        default=",".join(("evrp", "ccp", "cddce")),
        help="comma-separated pass names to ablate",
    )
    parser.add_argument("--plugin", help="explicit ai_native plugin path")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--out", required=True, help="dataset output path (.jsonl)")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        dest="sources",
        help="source file (repeatable)",
    )
    parser.add_argument(
        "compiler_args",
        nargs=argparse.REMAINDER,
        help="compiler arguments after --",
    )
    return parser


def _corpus_version() -> str:
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
        return build_corpus(
            sources=[Path(source) for source in args.sources],
            compiler=args.compiler,
            compiler_args=compiler_args,
            benchmark_cmd=args.benchmark_cmd,
            passes=passes,
            runs=args.runs,
            warmups=args.warmups,
            timeout=args.timeout,
            min_improvement=args.min_improvement,
            output_path=Path(args.out),
            plugin_path=Path(args.plugin).resolve() if args.plugin else None,
            seed=args.seed,
        )
    except OSError as exc:
        print_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
