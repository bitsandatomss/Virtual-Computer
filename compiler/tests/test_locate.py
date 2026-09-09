from __future__ import annotations

import json

from self_compiler.ai_engine import AIEngine, AIUnavailableError
from self_compiler.diagnostics import Diagnostic
from self_compiler.locate import (
    balance_score_ok,
    function_spans,
    function_text,
    locate_enclosing_function,
    splice_function,
)
from self_compiler.modes import LOCALIZATION_MIN_UNIT_LINES, RepairLoop

SAMPLE = """#include <stdio.h>
struct S { int a; };
int g = 3;
static int helper(int x) { return x * 2; }
int broken(int n) {
    const char* s = "} not a brace {";
    /* comment with } braces { */
    return helper(n) + (n > 0 ? 1 : 0);
}
int main(void) { return broken(1); }
"""


def test_function_spans_skip_structs_and_literal_braces():
    spans = function_spans(SAMPLE)
    assert [(s.name, s.start_line, s.end_line) for s in spans] == [
        ("helper", 4, 4),
        ("broken", 5, 9),
        ("main", 10, 10),
    ]


def test_locate_enclosing_function_and_misses():
    assert locate_enclosing_function(SAMPLE, 7).name == "broken"
    assert locate_enclosing_function(SAMPLE, 4).name == "helper"
    assert locate_enclosing_function(SAMPLE, 2) is None
    assert locate_enclosing_function(SAMPLE, 0) is None


def test_function_text_is_exact():
    broken = locate_enclosing_function(SAMPLE, 7)
    text = function_text(SAMPLE, broken)
    assert text.startswith("int broken(int n)")
    assert text.endswith("}")
    assert '"} not a brace {"' in text


def test_splice_rejects_unbalanced_replacement():
    block = locate_enclosing_function(SAMPLE, 6)
    assert splice_function(SAMPLE, block, "int broken(int n){ return 0; }") is not None
    spliced = splice_function(SAMPLE, block, "int broken(int n){ return 0; ")
    assert spliced is None


def test_balance_score_ignores_strings_and_comments():
    assert balance_score_ok('void f(void){ puts("}"); /* { */ }')
    assert not balance_score_ok("void f(void){")
    assert not balance_score_ok("}")


def _large_unit(defect_function: str) -> str:
    parts = ["#include <stdio.h>\n"]
    # One single-line filler each, comfortably past the localization floor.
    count = LOCALIZATION_MIN_UNIT_LINES + 10
    parts.extend(
        f"int filler_{index}(int x) {{ return x + {index}; }}\n"
        for index in range(count)
    )
    parts.append(defect_function)
    parts.append("int main(void) { return broken(0) + filler_0(0); }\n")
    return "".join(parts)


class LocalizedStubAI:
    """Records what it is asked to repair and answers per contract."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.replacement = "int broken(int n) {\n    return 0;\n}"

    def propose_localized_patch(self, block_text, diagnostics, mode, name):
        self.prompts.append(block_text)
        return self.replacement

    def propose_patch(self, source, diagnostics, mode):  # pragma: no cover
        raise AssertionError("whole-unit path must not run when localizing")


def test_repair_loop_localizes_large_unit_prompts(tmp_path):
    # A semantic defect GCC has no fix-it for, so the loop must consult AI,
    # and the only escape is the localized path.
    defect = "int broken(int n) {\n    return n + undeclared_identifier;\n}\n"
    source_path = tmp_path / "big.c"
    source_path.write_text(_large_unit(defect), encoding="utf-8")

    stub = LocalizedStubAI()
    stub.replacement = "int broken(int n) {\n    return n;\n}"
    loop = RepairLoop(
        str(source_path),
        max_attempts=1,
        ai=stub,
        journal_path=str(tmp_path / "journal.jsonl"),
    )
    assert loop.run("repair") is True

    # The model only ever saw the broken function, never the whole unit.
    assert len(stub.prompts) == 1
    assert stub.prompts[0].startswith("int broken(int n)")
    assert "filler_1" not in stub.prompts[0]
    assert source_path.read_text(encoding="utf-8").count("return n;") >= 1

    events = [
        json.loads(line)
        for line in (tmp_path / "journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    requested = next(e for e in events if e["event"] == "ai-requested")
    assert requested["strategy"] == "localized-function"
    assert requested["function"] == "broken"


def test_small_units_keep_whole_unit_prompts(tmp_path):
    source_path = tmp_path / "small.c"
    source_path.write_text(
        "static int helper(int x) { return x * 2; }\n"
        "int main(void) {\n"
        "    int v = helper(2)\n"
        "    return v - 4;\n"
        "}\n",
        encoding="utf-8",
    )
    seen: list[str] = []

    class WholeAI:
        def propose_patch(self, source, diagnostics, mode):
            seen.append(source)
            return source.replace("helper(2)", "helper(2);")

    loop = RepairLoop(str(source_path), max_attempts=1, ai=WholeAI())
    assert loop.run("repair") is True
    assert len(seen) == 1
    # The whole-unit path saw the original text, not the repaired file.
    assert "int v = helper(2)\n" in seen[0]
    assert "int v = helper(2);" in source_path.read_text(encoding="utf-8")


def test_localized_offline_engine_is_rejected(tmp_path):
    engine = AIEngine(offline=True)
    try:
        engine.propose_localized_patch("int f(void){return 0;}", [], "repair", "f")
    except AIUnavailableError as exc:
        assert "offline" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("offline engines must refuse localized proposals")


def test_diagnostic_line_filtering_uses_current_file(tmp_path):
    source = "int f(void) { return 0; }\n"
    feedback = [
        Diagnostic(str(tmp_path / "elsewhere.c"), 5, 1, "error", "other file"),
        Diagnostic(str(tmp_path / "here.c"), 1, 1, "error", "same file"),
    ]
    loop = RepairLoop(str(tmp_path / "here.c"), ai=None)
    target = loop._localization_target(source, feedback, str(tmp_path / "here.c"))
    # Small unit and a diagnostic in another file: no localization.
    assert target is None
