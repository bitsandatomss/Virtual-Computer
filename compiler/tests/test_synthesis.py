from __future__ import annotations

import json
from pathlib import Path


from self_compiler.synthesis import (
    Stump,
    WorkloadRow,
    _integerize,
    load_corpus,
    main,
    select_stump,
)


def _workload_record(name: str, total_statements: float, improves: bool) -> dict:
    return {
        "schema": "gcc-ai.corpus.v1",
        "event": "workload",
        "name": name,
        "source": {"path": f"{name}.c", "sha256": "0" * 64},
        "functions": [],
        "features": {
            "function_count": 2.0,
            "total_gimple_statements": total_statements,
            "max_basic_blocks": 6.0,
        },
        "outcomes": {
            "disable-ccp": {
                "verdict": "improves" if improves else "within-noise",
                "improvement_vs_control": 0.1 if improves else -0.01,
            }
        },
    }


def _write_corpus(path: Path, records: list[dict]) -> None:
    meta = {
        "schema": "gcc-ai.corpus.v1",
        "event": "corpus-meta",
        "passes": ["ccp"],
    }
    path.write_text(
        "\n".join(json.dumps(record) for record in [meta, *records]) + "\n",
        encoding="utf-8",
    )


def test_load_corpus_builds_supervised_rows(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus,
        [
            _workload_record("a", 100.0, True),
            _workload_record("b", 900.0, False),
        ],
    )
    passes, by_pass = load_corpus(corpus)
    assert passes == ["ccp"]
    rows = by_pass["ccp"]
    assert rows["a"].label == 1
    assert rows["b"].label == 0
    assert rows["a"].features["total_gimple_statements"] == 100.0


def test_select_stump_recovers_clean_threshold():
    rows = [
        WorkloadRow(f"small{index}", {"x": float(value)}, 1 if value <= 50 else 0)
        for index, value in enumerate([10, 20, 30, 40, 60, 70, 80, 90])
    ]
    stump = select_stump(rows, ["x"])
    assert stump is not None
    predictions = [stump.predict(row) for row in rows]
    labels = [row.label for row in rows]
    assert predictions == labels


def test_integerize_preserves_semantics_over_integers():
    le = _integerize(Stump("f", "<=", 37.5))
    ge = _integerize(Stump("f", ">=", 37.5))
    assert int(le.threshold) == 37 and le.op == "<="
    assert int(ge.threshold) == 38 and ge.op == ">="
    probe_values = list(range(30, 46))
    for value in probe_values:
        assert (value <= 37.5) == (value <= 37)
        assert (value >= 37.5) == (value >= 38)


def test_synthesis_emits_rule_for_learnable_pattern(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    records = [
        # Improves exactly when the program is large.
        _workload_record(f"w{index}", 200.0 + index * 100.0, index >= 6)
        for index in range(12)
    ]
    _write_corpus(corpus, records)

    policy_out = tmp_path / "learned.policy"
    report_out = tmp_path / "synthesis.jsonl"
    exit_code = main(
        [
            "--corpus",
            str(corpus),
            "--policy-out",
            str(policy_out),
            "--report",
            str(report_out),
        ]
    )
    assert exit_code == 0
    policy_text = policy_out.read_text(encoding="utf-8")
    import re

    match = re.search(r"^disable_pass=ccp if (\w+)(<=|>=)(\d+)$", policy_text.strip())
    assert match, policy_text
    feature, op, threshold = match.groups()
    assert feature == "total_gimple_statements"
    boundary = float(threshold)
    for record in records:
        value = record["features"]["total_gimple_statements"]
        predicted_improves = value >= boundary if op == ">=" else value <= boundary
        actual = record["outcomes"]["disable-ccp"]["verdict"] == "improves"
        assert predicted_improves == actual


def test_synthesis_refuses_constant_features_and_records_reasons(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    records = [
        _workload_record(f"w{index}", 500.0, index % 2 == 0) for index in range(10)
    ]
    _write_corpus(corpus, records)
    report_out = tmp_path / "synthesis.jsonl"

    exit_code = main(["--corpus", str(corpus), "--report", str(report_out)])
    assert exit_code == 1
    evaluation_records = [
        json.loads(line)
        for line in report_out.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "evaluation"
    ]
    assert len(evaluation_records) == 1
    finding = evaluation_records[0]
    assert finding["emitted"] is False
    assert any(
        "no non-degenerate" in reason or "single class" in reason
        for reason in finding["reasons"]
    )


def test_synthesis_is_deterministic_given_seed(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    noisy = [
        _workload_record(f"w{index}", float(index * 37), (index * 7 + 3) % 3 == 0)
        for index in range(16)
    ]
    _write_corpus(corpus, noisy)
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    main(["--corpus", str(corpus), "--report", str(first), "--seed", "42"])
    main(["--corpus", str(corpus), "--report", str(second), "--seed", "42"])
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_missing_corpus_is_usage_error(tmp_path, capsys):
    exit_code = main(["--corpus", str(tmp_path / "missing.jsonl")])
    assert exit_code == 2
    assert "corpus not found" in capsys.readouterr().err
