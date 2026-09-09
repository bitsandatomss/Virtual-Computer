from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .diagnostics import Diagnostic, FixIt, SourcePoint


@dataclass(frozen=True)
class FixItResult:
    source: str
    applied_edits: int
    diagnostics_used: int


def apply_compiler_fixits(
    source: str,
    diagnostics: list[Diagnostic],
    expected_file: str,
) -> FixItResult:
    """Apply compiler-authored edits atomically using GCC byte coordinates.

    All edits within one diagnostic are applied together or the whole
    diagnostic is skipped (a partial diagnostic edit could be wrong).
    Across diagnostics, independent groups are applied greedily: one
    conflicting diagnostic no longer discards unrelated compiler fixes.
    Every result is still revalidated by compilation and tests downstream.
    """
    encoded = source.encode("utf-8")
    expected = Path(expected_file).resolve()
    groups: list[list[tuple[int, int, bytes]]] = []

    for diagnostic in diagnostics:
        if diagnostic.severity not in {"error", "fatal error"} or not diagnostic.fixits:
            continue
        group: list[tuple[int, int, bytes]] = []
        for fixit in diagnostic.fixits:
            edit = _resolve_edit(encoded, fixit, expected)
            if edit is None:
                group = []
                break
            group.append(edit)
        if group:
            groups.append(group)

    if not groups:
        return FixItResult(source, 0, 0)

    accepted: list[tuple[int, int, bytes]] = []
    diagnostics_used = 0
    for group in sorted(groups, key=lambda g: min(edit[0] for edit in g)):
        ordered = sorted(group, key=lambda edit: (edit[0], edit[1]))
        if any(
            previous[1] > current[0] for previous, current in zip(ordered, ordered[1:])
        ):
            continue  # self-overlapping group is malformed; skip it entirely
        candidate = [*accepted, *ordered]
        candidate.sort(key=lambda edit: (edit[0], edit[1]))
        if any(
            previous[1] > current[0]
            for previous, current in zip(candidate, candidate[1:])
        ):
            continue  # conflicts with an already accepted diagnostic
        accepted = candidate
        diagnostics_used += 1

    if not accepted:
        return FixItResult(source, 0, 0)

    result = encoded
    for start, end, replacement in reversed(accepted):
        result = result[:start] + replacement + result[end:]
    try:
        decoded = result.decode("utf-8")
    except UnicodeDecodeError:
        return FixItResult(source, 0, 0)
    return FixItResult(decoded, len(accepted), diagnostics_used)


def _resolve_edit(
    source: bytes,
    fixit: FixIt,
    expected_file: Path,
) -> tuple[int, int, bytes] | None:
    if not _same_file(fixit.start.file, expected_file):
        return None
    if not _same_file(fixit.next.file, expected_file):
        return None
    start = _byte_offset(source, fixit.start)
    end = _byte_offset(source, fixit.next)
    if start is None or end is None or end < start:
        return None
    return start, end, fixit.replacement.encode("utf-8")


def _same_file(candidate: str, expected: Path) -> bool:
    if candidate.startswith("<") and candidate.endswith(">"):
        return False
    try:
        return Path(candidate).resolve() == expected
    except OSError:
        return False


def _byte_offset(source: bytes, point: SourcePoint) -> int | None:
    if point.line < 1 or point.byte_column < 1:
        return None
    starts = [0]
    starts.extend(index + 1 for index, byte in enumerate(source) if byte == 0x0A)
    if point.line > len(starts):
        return None
    offset = starts[point.line - 1] + point.byte_column - 1
    if offset > len(source):
        return None
    next_line_start = starts[point.line] if point.line < len(starts) else len(source)
    if offset > next_line_start:
        return None
    return offset
