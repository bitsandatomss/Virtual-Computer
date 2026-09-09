from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from self_compiler.corpus import (
    feature_aggregates,
    load_function_events,
    main,
)


def _function_event(name: str, **fields: int) -> dict:
    event = {
        "schema": "gcc-ai.telemetry.v2",
        "event": "function-ir",
        "function": name,
    }
    event.update(fields)
    return event


def test_feature_aggregates_sum_and_max():
    events = [
        _function_event("a", basic_blocks=3, gimple_statements=10, edges=4, calls=1),
        _function_event("b", basic_blocks=7, gimple_statements=30, edges=8, calls=2),
    ]
    aggregates = feature_aggregates(events)
    assert aggregates["function_count"] == 2
    assert aggregates["total_gimple_statements"] == 40
    assert aggregates["max_basic_blocks"] == 7
    assert aggregates["total_edges"] == 12
    assert aggregates["max_calls"] == 2
    # Missing fields default to zero.
    assert aggregates["total_float_ops"] == 0
    assert aggregates["max_memory_writes"] == 0


def test_feature_aggregates_empty_program():
    from self_compiler.corpus import FEATURE_FIELDS

    aggregates = feature_aggregates([])
    expected = {"function_count": 0} | {
        key: 0.0
        for key in (
            *(f"total_{field}" for field in FEATURE_FIELDS),
            *(f"max_{field}" for field in FEATURE_FIELDS),
        )
    }
    assert aggregates == expected


def test_load_function_events_tolerates_malformed_lines(tmp_path):
    telemetry = tmp_path / "t.jsonl"
    telemetry.write_text(
        json.dumps(_function_event("main", basic_blocks=1))
        + "\nnot json at all\n"
        + json.dumps({"schema": "gcc-ai.telemetry.v2", "event": "pass-gate"})
        + "\n",
        encoding="utf-8",
    )
    events = load_function_events(telemetry)
    assert [event["function"] for event in events] == ["main"]
    assert load_function_events(tmp_path / "missing.jsonl") == []


def _write_source(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


LOOP_BODY = """#include <stdio.h>
int compute(int n) {
    int total = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if ((i ^ j) & 1) total += i; else total -= j;
    return total;
}
int main(void) { printf("%d\\n", compute(24)); return 0; }
"""

TRIVIAL_BODY = "int main(void) { return 0; }\n"


def _plugin_headers_available() -> bool:
    if not shutil.which("gcc") or not shutil.which("g++"):
        return False
    plugin_directory = Path(
        subprocess.check_output(
            ["gcc", "-print-file-name=plugin"], text=True, errors="replace"
        ).strip()
    )
    return (plugin_directory / "include" / "gcc-plugin.h").is_file()


@pytest.mark.skipif(not _plugin_headers_available(), reason="plugin headers needed")
def test_corpus_dataset_end_to_end(tmp_path):
    heavy = tmp_path / "heavy.c"
    trivial = tmp_path / "trivial.c"
    _write_source(heavy, LOOP_BODY)
    _write_source(trivial, TRIVIAL_BODY)
    dataset = tmp_path / "corpus.jsonl"

    exit_code = main(
        [
            "--compiler",
            "gcc",
            "--runs",
            "3",
            "--warmups",
            "1",
            "--passes",
            "evrp",
            "--out",
            str(dataset),
            "--source",
            str(heavy),
            "--source",
            str(trivial),
            "--",
            "-O2",
        ]
    )
    assert exit_code == 0
    records = [
        json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
    ]
    events = {record["event"] for record in records}
    assert "corpus-meta" in events
    workloads = [record for record in records if record["event"] == "workload"]
    assert len(workloads) == 2
    by_name = {record["name"]: record for record in workloads}
    assert by_name["trivial"]["features"]["function_count"] >= 1
    assert "disable-evrp" in by_name["heavy"]["outcomes"]
    assert "plugin-control" in by_name["heavy"]["outcomes"]
    control_outcome = by_name["heavy"]["outcomes"]["plugin-control"]
    assert control_outcome["verdict"] == "control"
    meta = next(record for record in records if record["event"] == "corpus-meta")
    assert meta["objective"] == "runtime"
    assert meta["bootstrap_seed"]
