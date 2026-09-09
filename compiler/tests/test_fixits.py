from self_compiler.diagnostics import Diagnostic, FixIt, SourcePoint
from self_compiler.fixits import apply_compiler_fixits


def _point(path, line, byte_column):
    return SourcePoint(str(path), line, byte_column, byte_column)


def test_applies_half_open_gcc_byte_range(tmp_path):
    source_path = tmp_path / "demo.c"
    source = "int café = value.color;\n"
    start = source.encode("utf-8").index(b"color") + 1
    diagnostic = Diagnostic(
        str(source_path),
        1,
        start,
        "error",
        "did you mean colour?",
        fixits=(
            FixIt(
                _point(source_path, 1, start),
                _point(source_path, 1, start + len(b"color")),
                "colour",
            ),
        ),
    )
    result = apply_compiler_fixits(source, [diagnostic], str(source_path))
    assert result.source == "int café = value.colour;\n"
    assert result.applied_edits == 1


def test_applies_insertion_at_empty_range(tmp_path):
    source_path = tmp_path / "demo.c"
    source = "int value = 1\n"
    column = len("int value = 1") + 1
    point = _point(source_path, 1, column)
    diagnostic = Diagnostic(
        str(source_path),
        1,
        column,
        "error",
        "expected semicolon",
        fixits=(FixIt(point, point, ";"),),
    )
    assert apply_compiler_fixits(source, [diagnostic], str(source_path)).source == (
        "int value = 1;\n"
    )


def test_rejects_fixit_for_another_file(tmp_path):
    source_path = tmp_path / "demo.c"
    other = tmp_path / "header.h"
    point = _point(other, 1, 1)
    diagnostic = Diagnostic(
        str(other), 1, 1, "error", "header edit", fixits=(FixIt(point, point, "x"),)
    )
    result = apply_compiler_fixits("source\n", [diagnostic], str(source_path))
    assert result.applied_edits == 0
    assert result.source == "source\n"


def test_independent_diagnostics_apply_when_another_conflicts():
    """Greedy cross-diagnostic application: the earliest group wins an
    overlap, while diagnostics that do not touch the contested range still
    apply instead of being discarded wholesale."""
    source_path_str = "demo.c"
    source = "int value = 1\nint other = 2\n"
    first = _point(source_path_str, 1, len("int value = 1") + 1)
    independent = Diagnostic(
        source_path_str,
        1,
        1,
        "error",
        "expected ';'",
        fixits=(FixIt(first, first, ";"),),
    )
    rewrite_start = _point(source_path_str, 1, len("int value") + 1)
    # End column extends one byte further so the range [9, 14) swallows the
    # newline and truly overlaps the insertion at offset 13; ending exactly
    # at 13 would merely be adjacent, which is not a conflict.
    rewrite_end = _point(source_path_str, 1, len("int value = 1") + 2)
    rewrite = Diagnostic(
        source_path_str,
        1,
        1,
        "error",
        "overlapping rewrite",
        fixits=(FixIt(rewrite_start, rewrite_end, "XX"),),
    )
    second = _point(source_path_str, 2, len("int other = 2") + 1)
    unrelated = Diagnostic(
        source_path_str,
        2,
        1,
        "error",
        "expected ';'",
        fixits=(FixIt(second, second, ";"),),
    )
    result = apply_compiler_fixits(
        source,
        [rewrite, independent, unrelated],
        source_path_str,
    )
    assert result.source == "int valueXXint other = 2;\n"
    assert result.applied_edits == 2
