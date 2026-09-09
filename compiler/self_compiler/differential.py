"""Seeded random-input differential testing for policy-compiled binaries.

A policy variant is only trusted when it behaves identically to the
control across inputs, not merely on the single benchmark vector used
during timing.  This module generates deterministic pseudo-random input
cases from a recorded seed, runs both binaries under each case, and
compares observable behavior (exit status and standard output).  Because
the generator is seeded and the cases are recorded in evidence, any
reported equivalence or divergence can be replayed exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "gcc-ai.differential.v1"
DEFAULT_SEED = 20260822


@dataclass(frozen=True)
class DifferentialCase:
    name: str
    stdin: bytes


@dataclass(frozen=True)
class DifferentialMismatch:
    case: str
    kind: str  # exit-code | stdout | timeout | error
    detail: str


@dataclass(frozen=True)
class DifferentialReport:
    equivalent: bool
    trials: int
    skipped: int
    mismatches: tuple[DifferentialMismatch, ...]
    seed: int


def generate_cases(trials: int, seed: int) -> list[DifferentialCase]:
    """Build a deterministic case list: fixed edge cases plus seeded blobs."""
    if trials < 1:
        raise ValueError("Differential testing requires at least one trial.")
    rng = random.Random(seed)
    cases = [
        DifferentialCase("empty", b""),
        DifferentialCase("single-null", b"\x00"),
        DifferentialCase("newline", b"\n"),
        DifferentialCase(
            "ascii-lines",
            "".join(f"line {index}\n" for index in range(32)).encode("ascii"),
        ),
        DifferentialCase("all-zero-4k", b"\x00" * 4096),
    ]
    sizes = (1, 7, 64, 255, 1024, 8192)
    index = 0
    while len(cases) < trials:
        size = sizes[index % len(sizes)]
        blob = bytes(rng.randrange(256) for _ in range(size))
        cases.append(DifferentialCase(f"random-{index}", blob))
        index += 1
    return cases[:trials]


def _run_one(
    binary: Path, stdin: bytes, timeout: float
) -> tuple[int | None, bytes, str]:
    """Execute one side; return (returncode-or-None, stdout, error message)."""
    try:
        completed = subprocess.run(
            [str(binary)],
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return None, b"", f"binary not found: {binary}"
    except subprocess.TimeoutExpired:
        return None, b"", f"timeout after {timeout:g}s"
    return completed.returncode, completed.stdout, ""


def run_differential(
    reference: Path,
    candidate: Path,
    trials: int = 12,
    seed: int = DEFAULT_SEED,
    timeout: float = 10.0,
) -> DifferentialReport:
    """Compare two binaries over deterministic cases; never raises on mismatch."""
    mismatches: list[DifferentialMismatch] = []
    skipped = 0
    cases = generate_cases(trials, seed)
    for case in cases:
        ref_code, ref_out, ref_error = _run_one(reference, case.stdin, timeout)
        cand_code, cand_out, cand_error = _run_one(candidate, case.stdin, timeout)
        if ref_error == "timeout" or cand_error == "timeout":
            if ref_error == "timeout" and cand_error == "timeout":
                skipped += 1  # both sides exceeded budget; nothing to compare
                continue
            slow_side = "reference" if ref_error == "timeout" else "candidate"
            mismatches.append(
                DifferentialMismatch(
                    case.name,
                    "timeout",
                    f"{slow_side} exceeded {timeout:g}s",
                )
            )
            continue
        if ref_error or cand_error:
            mismatches.append(
                DifferentialMismatch(
                    case.name,
                    "error",
                    ref_error or cand_error,
                )
            )
            continue
        if ref_code != cand_code:
            mismatches.append(
                DifferentialMismatch(
                    case.name,
                    "exit-code",
                    f"reference exited {ref_code}, candidate exited {cand_code}",
                )
            )
            continue
        if ref_out != cand_out:
            mismatches.append(
                DifferentialMismatch(
                    case.name,
                    "stdout",
                    f"outputs differ at byte "
                    f"{next((i for i, (a, b) in enumerate(zip(ref_out, cand_out)) if a != b), min(len(ref_out), len(cand_out)))}",
                )
            )
    return DifferentialReport(
        not mismatches, len(cases), skipped, tuple(mismatches), seed
    )


def write_report(report: DifferentialReport, path: Path) -> None:
    lines = [
        json.dumps(
            {
                "schema": SCHEMA,
                "event": "differential-meta",
                "seed": report.seed,
                "trials": report.trials,
            }
        )
    ]
    for mismatch in report.mismatches:
        lines.append(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "event": "mismatch",
                    "case": mismatch.case,
                    "kind": mismatch.kind,
                    "detail": mismatch.detail,
                }
            )
        )
    lines.append(
        json.dumps(
            {
                "schema": SCHEMA,
                "event": "decision",
                "equivalent": report.equivalent,
                "trials": report.trials,
                "skipped_both_timeout": report.skipped,
                "mismatch_count": len(report.mismatches),
            }
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcc-ai-differential",
        description=(
            "Differential-test two binaries over seeded pseudo-random inputs; "
            "exit code 0 means behaviorally equivalent."
        ),
    )
    parser.add_argument("--reference", required=True, help="baseline binary")
    parser.add_argument("--candidate", required=True, help="policy variant binary")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--report", help="optional JSONL evidence path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = Path(args.reference)
    candidate = Path(args.candidate)
    missing = [str(path) for path in (reference, candidate) if not path.is_file()]
    if missing:
        print(f"gcc-ai-differential: binary not found: {missing[0]}", file=sys.stderr)
        return 2
    try:
        report = run_differential(
            reference, candidate, args.trials, args.seed, args.timeout
        )
    except ValueError as exc:
        print(f"gcc-ai-differential: {exc}", file=sys.stderr)
        return 2
    if args.report:
        write_report(report, Path(args.report))
    if report.equivalent:
        print(
            f"EQUIVALENT across {report.trials} cases "
            f"(seed {report.seed}, {report.skipped} both-timeout skips)"
        )
        return 0
    print(f"DIVERGENT across {report.trials} cases (seed {report.seed}):")
    for mismatch in report.mismatches[:10]:
        print(f"  [{mismatch.kind}] {mismatch.case}: {mismatch.detail}")
    if len(report.mismatches) > 10:
        print(f"  ... and {len(report.mismatches) - 10} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
