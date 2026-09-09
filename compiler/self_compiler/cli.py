from __future__ import annotations

import argparse
import sys

from .ai_engine import AIEngine
from .logger import print_banner, print_error
from .modes import RepairLoop


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("gcc-ai-native")
    except Exception:
        return "0.0.0+unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcc-ai-repair",
        description="Generate-and-validate AI repair orchestrator for one C/C++ translation unit.",
    )
    parser.add_argument("--version", action="version", version=_version())
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--repair", action="store_true", help="repair compilation failures (default)"
    )
    modes.add_argument(
        "--optimize",
        action="store_true",
        help="require measured performance improvement",
    )
    modes.add_argument(
        "--secure",
        action="store_true",
        help="repair findings from available static analyzers",
    )
    modes.add_argument(
        "--self-correct",
        action="store_true",
        help="repair a reproducible runtime failure",
    )

    parser.add_argument(
        "--backend", default="gcc", help="compiler executable (default: gcc)"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3, help="maximum candidate generations"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="per compile/test timeout in seconds",
    )
    parser.add_argument(
        "--test-cmd",
        help="correctness command; supports {binary} and {source} placeholders",
    )
    parser.add_argument(
        "--debugger",
        choices=("gdb",),
        help="capture a structured GDB/MI stack trace after direct execution fails",
    )
    parser.add_argument(
        "--program-arg",
        action="append",
        default=[],
        help="argument passed to direct binary execution (repeatable)",
    )
    parser.add_argument(
        "--benchmark-cmd",
        help="optimization benchmark command; supports {binary} and {source} placeholders",
    )
    parser.add_argument(
        "--benchmark-runs", type=int, default=5, help="measured benchmark repetitions"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="required fractional median speedup (default: 0.05)",
    )
    parser.add_argument("--model", help="Gemini model (or set GEMINI_MODEL)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the opt-in, semicolon-only repair heuristic instead of Gemini",
    )
    parser.add_argument(
        "--output", help="publish the final verified binary to this path"
    )
    parser.add_argument(
        "--journal", help="append a machine-readable JSONL audit trail to this path"
    )
    parser.add_argument("source", help="C or C++ source file")
    parser.add_argument(
        "compiler_args",
        nargs=argparse.REMAINDER,
        help="compiler arguments after -- (output flags are not accepted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    compiler_args = args.compiler_args
    if compiler_args[:1] == ["--"]:
        compiler_args = compiler_args[1:]

    mode = "repair"
    if args.optimize:
        mode = "optimize"
    elif args.secure:
        mode = "secure"
    elif args.self_correct:
        mode = "self-correct"

    print_banner(mode, args.source)
    try:
        loop = RepairLoop(
            source_path=args.source,
            compiler_cmd=args.backend,
            compiler_args=compiler_args,
            test_cmd=args.test_cmd,
            max_attempts=args.max_attempts,
            benchmark_cmd=args.benchmark_cmd,
            benchmark_runs=args.benchmark_runs,
            min_improvement=args.min_improvement,
            timeout=args.timeout,
            output_path=args.output,
            ai=AIEngine(model=args.model, offline=args.offline),
            debugger=args.debugger,
            program_args=args.program_arg,
            journal_path=args.journal,
        )
        return 0 if loop.run(mode) else 1
    except (OSError, RuntimeError, ValueError) as exc:
        print_error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
