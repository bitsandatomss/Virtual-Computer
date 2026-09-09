from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from self_compiler.policy_lab import (
    VariantResult,
    _default_specs,
    classify,
    main,
    rank_improving,
    run_lab,
    summarize,
)


def test_summarize_reports_median_and_spread():
    median, spread = summarize([10.0, 12.0, 11.0])
    assert median == 11.0
    assert spread == pytest.approx((12.0 - 10.0) / 11.0)


def test_summarize_handles_empty_and_single_samples():
    assert summarize([]) == (0.0, 0.0)
    assert summarize([5.0]) == (5.0, 0.0)


def test_classify_requires_improvement_gates_and_behavior():
    verdict, improvement = classify(
        control_median=100.0,
        variant_median=90.0,
        min_improvement=0.02,
        gate_events=3,
        output_equal=True,
    )
    assert verdict == "improves"
    assert improvement == pytest.approx(0.1)

    assert (
        classify(
            control_median=100.0,
            variant_median=90.0,
            min_improvement=0.02,
            gate_events=0,
            output_equal=True,
        )[0]
        == "not-gated"
    )

    assert (
        classify(
            control_median=100.0,
            variant_median=90.0,
            min_improvement=0.02,
            gate_events=3,
            output_equal=False,
        )[0]
        == "invalid-behavior"
    )

    assert (
        classify(
            control_median=100.0,
            variant_median=105.0,
            min_improvement=0.02,
            gate_events=3,
            output_equal=True,
        )[0]
        == "regresses"
    )


def test_rank_prefers_largest_verified_improvement():
    results = [
        VariantResult("a", "improves", 80.0, 0.05, 0.2, 100, 4, True),
        VariantResult("b", "improves", 85.0, 0.05, 0.15, 100, 4, True),
        VariantResult("c", "regresses", 120.0, 0.05, -0.2, 100, 4, True),
        VariantResult("d", "not-gated", 70.0, 0.05, 0.3, 100, 0, True),
    ]
    assert rank_improving(results).name == "a"
    assert rank_improving([results[2]]) is None


def test_default_specs_include_control_and_ablations():
    specs = _default_specs(["evrp"])
    names = [spec.name for spec in specs]
    assert names[0] == "gcc-baseline"
    assert names[1] == "plugin-control"
    assert specs[-1].policy_text == "disable_pass=evrp\n"


def test_lab_rejects_runs_below_three(tmp_path):
    source = tmp_path / "p.c"
    source.write_text("int main(void){return 0;}\n", encoding="utf-8")
    exit_code = run_lab(
        sources=[source],
        compiler="gcc",
        compiler_args=["-O2"],
        benchmark_cmd=None,
        passes=["evrp"],
        extra_policies=[],
        runs=2,
        warmups=1,
        timeout=30.0,
        min_improvement=0.02,
        output_dir=tmp_path / "lab",
        plugin_path=None,
    )
    assert exit_code == 2


def test_lab_cli_requires_sources(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


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
def test_lab_produces_evidence_on_real_toolchain(tmp_path):
    source = tmp_path / "work.c"
    source.write_text(
        "#include <stdio.h>\n"
        "int find(int arr[], int size, int target) {\n"
        "    for (int i = 0; i < size; i++)\n"
        "        if (arr[i] == target)\n"
        "            return i;\n"
        "    return -1;\n"
        "}\n"
        "int main(void) {\n"
        "    int data[] = {1,3,5,7,9,11,13,15,17,19,21};\n"
        '    printf("%d\\n", find(data, 11, 13));\n'
        "    return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    lab_dir = tmp_path / "lab"
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
            "--output-dir",
            str(lab_dir),
            "--source",
            str(source),
            "--",
            "-O2",
        ]
    )
    assert exit_code == 0
    records = [
        json.loads(line)
        for line in (lab_dir / "evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    events = {record.get("event") for record in records}
    assert {"lab-meta", "build", "result", "decision"} <= events
    results = [record for record in records if record["event"] == "result"]
    by_name = {record["variant"]: record for record in results}
    assert by_name["gcc-baseline"]["verdict"] == "pure-gcc-reference"
    assert by_name["plugin-control"]["verdict"] == "control"
    assert by_name["disable-evrp"]["gate_events"] >= 1
    meta = next(record for record in records if record["event"] == "lab-meta")
    assert meta["sources"][0]["sha256"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path handling check")
def test_lab_missing_source_fails_cleanly(tmp_path):
    lab_dir = tmp_path / "lab"
    exit_code = main(
        [
            "--output-dir",
            str(lab_dir),
            "--source",
            str(tmp_path / "missing.c"),
        ]
    )
    assert exit_code == 2
