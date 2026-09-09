from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_WITH_COLUMN = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<severity>fatal error|error|warning|note|remark):\s*(?P<message>.*)$"
)
_WITHOUT_COLUMN = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):\s*"
    r"(?P<severity>fatal error|error|warning|note|remark):\s*(?P<message>.*)$"
)
_OPTION = re.compile(r"\s+(?P<code>\[-[^\]]+\])$")


@dataclass(frozen=True)
class SourcePoint:
    file: str
    line: int
    column: int
    byte_column: int


@dataclass(frozen=True)
class FixIt:
    start: SourcePoint
    next: SourcePoint
    replacement: str


@dataclass(frozen=True)
class Diagnostic:
    file: str
    line: int
    col: int
    severity: str
    message: str
    code: str = ""
    fixits: tuple[FixIt, ...] = ()


def parse_compiler_stderr(stderr: str) -> list[Diagnostic]:
    """Parse GCC/Clang text diagnostics, including Windows drive paths."""
    diagnostics: list[Diagnostic] = []

    for raw_line in stderr.splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line).strip()
        match = _WITH_COLUMN.match(line) or _WITHOUT_COLUMN.match(line)
        if not match:
            continue

        message = match.group("message").strip()
        option = _OPTION.search(message)
        code = option.group("code") if option else ""
        diagnostics.append(
            Diagnostic(
                file=match.group("file"),
                line=int(match.group("line")),
                col=int(match.groupdict().get("col") or 0),
                severity=match.group("severity"),
                message=message,
                code=code,
            )
        )

    return diagnostics


def diagnostic_from_output(file: str, message: str) -> Diagnostic:
    return Diagnostic(file=file, line=0, col=0, severity="error", message=message)


def parse_gcc_json(stderr: str) -> list[Diagnostic]:
    """Parse GCC's ``-fdiagnostics-format=json`` output.

    GCC columns are one-based. Fix-it ranges are half-open and use byte columns,
    which are retained separately from display columns for exact application.
    """
    try:
        payload = json.loads(stderr.strip())
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, list):
        return []

    diagnostics: list[Diagnostic] = []
    for item in payload:
        if isinstance(item, dict):
            diagnostics.extend(_diagnostics_from_json_item(item))
    return diagnostics


def _diagnostics_from_json_item(item: dict[str, Any]) -> list[Diagnostic]:
    locations = item.get("locations")
    caret: dict[str, Any] = {}
    if isinstance(locations, list) and locations and isinstance(locations[0], dict):
        candidate = locations[0].get("caret")
        if isinstance(candidate, dict):
            caret = candidate

    fixits: list[FixIt] = []
    raw_fixits = item.get("fixits")
    if isinstance(raw_fixits, list):
        for raw_fixit in raw_fixits:
            if not isinstance(raw_fixit, dict):
                continue
            start = _source_point(raw_fixit.get("start"))
            next_point = _source_point(raw_fixit.get("next"))
            replacement = raw_fixit.get("string")
            if start and next_point and isinstance(replacement, str):
                fixits.append(FixIt(start, next_point, replacement))

    option = item.get("option")
    diagnostic = Diagnostic(
        file=str(caret.get("file", "")),
        line=_positive_int(caret.get("line")),
        col=_positive_int(caret.get("column") or caret.get("display-column")),
        severity=str(item.get("kind", "error")),
        message=str(item.get("message", "")),
        code=str(option) if option else "",
        fixits=tuple(fixits),
    )
    result = [diagnostic]
    children = item.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                result.extend(_diagnostics_from_json_item(child))
    return result


def _source_point(value: object) -> SourcePoint | None:
    if not isinstance(value, dict):
        return None
    file = value.get("file")
    line = _positive_int(value.get("line"))
    column = _positive_int(value.get("column") or value.get("display-column"))
    byte_column = _positive_int(value.get("byte-column") or column)
    if not isinstance(file, str) or not file or line < 1 or byte_column < 1:
        return None
    return SourcePoint(file, line, column, byte_column)


def _positive_int(value: object) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return result if result >= 0 else 0
