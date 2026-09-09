import json

from self_compiler.diagnostics import parse_compiler_stderr, parse_gcc_json


def test_parse_standard_error():
    stderr = "examples/missing_semicolon.c:4:5: error: expected ';' before 'printf'\n"
    diags = parse_compiler_stderr(stderr)

    assert len(diags) == 1
    d = diags[0]
    assert d.file == "examples/missing_semicolon.c"
    assert d.line == 4
    assert d.col == 5
    assert d.severity == "error"
    assert d.message == "expected ';' before 'printf'"


def test_parse_clang_error():
    stderr = "slow_search.c:15:5: warning: format specifies type 'int' but the argument has type 'char *' [-Wformat]\n"
    diags = parse_compiler_stderr(stderr)

    assert len(diags) == 1
    d = diags[0]
    assert d.severity == "warning"
    assert "-Wformat" in d.message
    assert d.code == "[-Wformat]"


def test_parse_windows_path_and_diagnostic_without_column():
    stderr = (
        "C:\\work\\demo.c:12:7: error: bad token [-Wparse]\n"
        "C:\\work\\demo.c:15: warning: suspicious code\n"
    )
    diags = parse_compiler_stderr(stderr)

    assert [(d.file, d.line, d.col) for d in diags] == [
        (r"C:\work\demo.c", 12, 7),
        (r"C:\work\demo.c", 15, 0),
    ]


def test_parse_ignores_context_lines_and_strips_ansi():
    stderr = "\x1b[31mdemo.c:2:1: fatal error: broken\x1b[0m\n  2 | nope\n"
    diags = parse_compiler_stderr(stderr)
    assert len(diags) == 1
    assert diags[0].severity == "fatal error"


def test_parse_gcc_json_retains_byte_accurate_fixit():
    payload = [
        {
            "kind": "error",
            "message": "no member named color; did you mean colour?",
            "option": "-Wmember",
            "locations": [
                {
                    "caret": {
                        "file": r"C:\work\demo.c",
                        "line": 3,
                        "column": 12,
                        "byte-column": 12,
                    }
                }
            ],
            "fixits": [
                {
                    "start": {
                        "file": r"C:\work\demo.c",
                        "line": 3,
                        "column": 12,
                        "byte-column": 12,
                    },
                    "next": {
                        "file": r"C:\work\demo.c",
                        "line": 3,
                        "column": 17,
                        "byte-column": 17,
                    },
                    "string": "colour",
                }
            ],
            "children": [],
        }
    ]
    diagnostics = parse_gcc_json(json.dumps(payload))
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "-Wmember"
    assert diagnostics[0].fixits[0].replacement == "colour"
    assert diagnostics[0].fixits[0].next.byte_column == 17


def test_parse_gcc_json_rejects_non_json_text():
    assert parse_gcc_json("ordinary compiler output") == []
