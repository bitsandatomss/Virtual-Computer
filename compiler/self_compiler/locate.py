"""Diagnostic-driven source localization for bounded repair prompts.

Feeding an entire production translation unit to a language model wastes
context, invites truncated rewrites, and multiplies the surface area of
any hallucination.  This module localizes compiler diagnostics to their
enclosing top-level function so the generative engine sees only the code
it may change; the driver splices the returned function back and the
existing compile/test gauntlet validates the result exactly as it would
a whole-file candidate.

Localization is lexical, not semantic: it recognizes brace-delimited
top-level blocks whose headers end in a parameter list, which covers the
C family well enough to bound prompt size.  When localization is unsure
it says so—the caller falls back to the whole-unit path rather than
guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceBlock:
    """One top-level brace-delimited block with a parsed header name."""

    name: str
    kind: str  # "function" | "other"
    start_offset: int  # first character of the declaration header
    open_offset: int  # offset of the opening '{'
    close_offset: int  # offset of the matching '}'
    start_line: int  # 1-based
    end_line: int  # 1-based, inclusive


def _mask_non_code(source: str) -> str:
    """Blank out comments and string/char literals, preserving length.

    Newlines are preserved so byte offsets and line numbers computed on
    the mask match the original text exactly.
    """
    chars = list(source)
    index = 0
    length = len(source)

    def blank(start: int, stop: int) -> None:
        for position in range(start, min(stop, length)):
            if chars[position] != "\n":
                chars[position] = " "

    while index < length:
        char = source[index]
        two = source[index : index + 2]
        if two == "//":
            stop = source.find("\n", index)
            stop = length if stop < 0 else stop
            blank(index, stop)
            index = stop
        elif two == "/*":
            stop = source.find("*/", index + 2)
            if stop < 0:
                blank(index + 2, length)
                break
            blank(index, stop + 2)
            index = stop + 2
        elif char in {'"', "'"}:
            quote = char
            position = index + 1
            while position < length:
                inner = source[position]
                if inner == "\\":
                    position += 2
                    continue
                if inner == quote or inner == "\n":
                    break
                position += 1
            # Blank between the quotes, keeping the quotes themselves so
            # brace depth is unaffected either way.
            blank(index + 1, min(position, length))
            index = position + 1
        else:
            index += 1
    return "".join(chars)


_HEADER_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def function_spans(source: str) -> list[SourceBlock]:
    """Return top-level function-like blocks in source order."""
    masked = _mask_non_code(source)
    spans: list[SourceBlock] = []
    depth = 0
    declaration_start = 0

    def line_of(offset: int) -> int:
        return masked.count("\n", 0, offset) + 1

    for index, char in enumerate(masked):
        if char == "{":
            if depth == 0:
                header = masked[declaration_start:index]
                stripped = header.rstrip()
                # The recorded span must begin at the declaration's first
                # non-blank character, not at leftover whitespace from the
                # previous top-level construct.
                leading_blank = len(header) - len(header.lstrip())
                body_start = declaration_start + leading_blank
                # Skip balanced attribute/specifier parens between the
                # parameter list's ')' and the body's '{'.
                probe = len(stripped)
                paren_depth = 0
                saw_close = False
                while probe > 0:
                    current = stripped[probe - 1]
                    if current == ")":
                        paren_depth += 1
                        saw_close = True
                    elif current == "(":
                        paren_depth -= 1
                        if paren_depth == 0:
                            break
                    elif paren_depth == 0 and not current.isspace():
                        break
                    probe -= 1
                is_function = saw_close and paren_depth == 0 and probe > 0
                name = ""
                if is_function:
                    # Window includes the parameter list's own '(' so plain
                    # `name(args) {` headers match, while attribute noise
                    # earlier in the header loses to the nearest name.
                    search_window = stripped[:probe]
                    names = _HEADER_NAME.findall(search_window)
                    name = names[-1] if names else ""
                kind = "function" if is_function else "other"
                spans.append(
                    SourceBlock(
                        name=name,
                        kind=kind,
                        start_offset=body_start,
                        open_offset=index,
                        close_offset=-1,
                        start_line=line_of(body_start),
                        end_line=-1,
                    )
                )
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and spans and spans[-1].close_offset < 0:
                previous = spans[-1]
                spans[-1] = SourceBlock(
                    name=previous.name,
                    kind=previous.kind,
                    start_offset=previous.start_offset,
                    open_offset=previous.open_offset,
                    close_offset=index,
                    start_line=previous.start_line,
                    end_line=line_of(index),
                )
                declaration_start = index + 1
        elif char == ";" and depth == 0:
            declaration_start = index + 1
    return [span for span in spans if span.kind == "function"]


def locate_enclosing_function(source: str, line: int) -> SourceBlock | None:
    """Find the function whose lines contain ``line`` (1-based)."""
    if line < 1:
        return None
    for span in function_spans(source):
        if span.start_line <= line <= span.end_line:
            return span
    return None


def balance_score_ok(replacement: str) -> bool:
    """Cheap sanity gate: braces outside strings/comments must balance."""
    masked = _mask_non_code(replacement)
    depth = 0
    for char in masked:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def function_text(source: str, block: SourceBlock) -> str:
    """Exact source text of a located block."""
    return source[block.start_offset : block.close_offset + 1]


def splice_function(source: str, block: SourceBlock, replacement: str) -> str | None:
    """Replace one located block, or return None on an unbalanced candidate.

    The splice itself is byte-exact: everything before the declaration and
    after the closing brace is untouched, so a validated replacement cannot
    corrupt unrelated code.
    """
    cleaned = replacement.strip("\n")
    if not cleaned.strip() or not balance_score_ok(cleaned):
        return None
    return source[: block.start_offset] + cleaned + source[block.close_offset + 1 :]
