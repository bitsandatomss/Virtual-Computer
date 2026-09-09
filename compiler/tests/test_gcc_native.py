from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from self_compiler.build_plugin import build_plugins
from self_compiler.ai_engine import AIEngine
from self_compiler.gcc_native import main
from self_compiler.modes import RepairLoop


def _plugin_headers_available() -> bool:
    if not shutil.which("gcc") or not shutil.which("g++"):
        return False
    plugin_directory = Path(
        subprocess.check_output(
            ["gcc", "-print-file-name=plugin"], text=True, errors="replace"
        ).strip()
    )
    return (plugin_directory / "include" / "gcc-plugin.h").is_file()


@pytest.fixture(scope="module")
def native_plugins(tmp_path_factory):
    if not _plugin_headers_available():
        pytest.skip("matching GCC plugin development headers are unavailable")
    return build_plugins(tmp_path_factory.mktemp("gcc-plugins"))


def _plugin_for(plugins, language):
    if len(plugins) == 1:
        return plugins[0]
    suffix = "ai_native_cpp.dll" if language == "cpp" else "ai_native_c.dll"
    return next(path for path in plugins if path.name == suffix)


@pytest.mark.parametrize(
    ("language", "compiler", "extension"),
    [("c", "gcc", ".c"), ("cpp", "g++", ".cpp")],
)
def test_plugin_runs_inside_real_gcc_frontend(
    native_plugins, tmp_path, language, compiler, extension
):
    source = tmp_path / f"program{extension}"
    source.write_text(
        "int square(int x) { return x * x; }\n"
        "int main(void) { return square(3) == 9 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    telemetry = tmp_path / f"{language}.jsonl"
    program = tmp_path / f"{language}.exe"
    plugin = _plugin_for(native_plugins, language)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_compiler.gcc_native",
            "--compiler",
            compiler,
            "--plugin",
            str(plugin),
            "--telemetry",
            str(telemetry),
            "--",
            "-O2",
            str(source),
            "-o",
            str(program),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert subprocess.run([program], check=False).returncode == 0
    events = [json.loads(line) for line in telemetry.read_text().splitlines()]
    function_events = [e for e in events if e.get("event") == "function-ir"]
    assert {e.get("function") for e in function_events} >= {"square", "main"}
    assert all(e["schema"] == "gcc-ai.telemetry.v3" for e in events)
    for event in function_events:
        for field in (
            "basic_blocks",
            "gimple_statements",
            "phi_nodes",
            "calls",
            "branches",
            "edges",
            "memory_reads",
            "memory_writes",
            "float_ops",
            "back_edges",
            "max_loop_depth",
            "dominator_height",
            "cyclomatic_complexity",
            "max_out_degree",
        ):
            assert field in event, f"missing telemetry field {field}"
        assert event["edges"] >= event["basic_blocks"] - 1
        assert event["cyclomatic_complexity"] >= 1


def test_structural_features_separate_loops_from_straight_line(
    native_plugins, tmp_path
):
    source = tmp_path / "shape.c"
    source.write_text(
        "int nested(int n) {\n"
        "    int total = 0;\n"
        "    for (int i = 0; i < n; i++)\n"
        "        for (int j = 0; j < i; j++)\n"
        "            total += (i ^ j) & 1 ? i : -j;\n"
        "    return total;\n"
        "}\n"
        "int straight(int a) { return a + 1; }\n"
        "int main(void) { return nested(4) == 6 && straight(2) == 3 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    telemetry = tmp_path / "shape.jsonl"
    plugin = _plugin_for(native_plugins, "c")

    result = subprocess.run(
        [
            "gcc",
            "-O2",
            f"-fplugin={plugin}",
            f"-fplugin-arg-{plugin.stem}-output={telemetry}",
            str(source),
            "-o",
            str(tmp_path / "shape.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    events = {
        json.loads(line)["function"]: json.loads(line)
        for line in telemetry.read_text().splitlines()
        if '"function-ir"' in line
    }
    nested = events["nested"]
    flat = events["straight"]
    assert nested["back_edges"] >= 2
    assert nested["max_loop_depth"] >= 2
    assert nested["cyclomatic_complexity"] > flat["cyclomatic_complexity"]
    assert flat["back_edges"] == 0
    assert flat["max_loop_depth"] == 0


def test_cfg_export_emits_connectivity(native_plugins, tmp_path):
    source = tmp_path / "cfg.c"
    source.write_text(
        "int f(int n) {\n"
        "    int total = 0;\n"
        "    for (int i = 0; i < n; i++)\n"
        "        total += i;\n"
        "    return total;\n"
        "}\n"
        "int main(void) { return f(3) == 3 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    telemetry = tmp_path / "cfg.jsonl"
    plugin = _plugin_for(native_plugins, "c")

    result = subprocess.run(
        [
            "gcc",
            "-O2",
            f"-fplugin={plugin}",
            "-fplugin-arg-ai_native_c-cfg=export",
            f"-fplugin-arg-ai_native_c-output={telemetry}",
            str(source),
            "-o",
            str(tmp_path / "cfg.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    cfg_events = [
        json.loads(line)
        for line in telemetry.read_text().splitlines()
        if '"function-cfg"' in line
    ]
    by_function = {event["function"]: event for event in cfg_events}
    edges = by_function["f"]["succ"]
    # The loop body must contain a back edge (source >= dest index).
    assert any(src >= dst for src, dst, _flags in edges)


def test_pass_instance_targeting_gates_single_instance(native_plugins, tmp_path):
    source = tmp_path / "inst.c"
    source.write_text(
        "int compute(int n) {\n"
        "    int x = n * 3;\n"
        "    int y = x + n;\n"
        "    int z = y - n;\n"
        "    return z;\n"
        "}\n"
        "int main(void) { return compute(5) == 15 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    plugin = _plugin_for(native_plugins, "c")
    plugin_flag = f"-fplugin={plugin}"
    output_flag = f"-fplugin-arg-{plugin.stem}-output"

    discovery = tmp_path / "discover.jsonl"
    any_policy = tmp_path / "any.policy"
    any_policy.write_text("disable_pass=ccp\n", encoding="utf-8")
    subprocess.run(
        [
            "gcc",
            "-O2",
            plugin_flag,
            f"{output_flag}={discovery}",
            f"-fplugin-arg-{plugin.stem}-policy={any_policy}",
            str(source),
            "-o",
            str(tmp_path / "disc.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    instances = sorted(
        {
            json.loads(line)["instance"]
            for line in discovery.read_text().splitlines()
            if '"pass-gate"' in line
        }
    )
    assert instances, "expected ccp to run at -O2 for this program"

    target = instances[0]
    targeted_telemetry = tmp_path / "targeted.jsonl"
    targeted_policy = tmp_path / "targeted.policy"
    targeted_policy.write_text(f"disable_pass=ccp#{target}\n", encoding="utf-8")
    result = subprocess.run(
        [
            "gcc",
            "-O2",
            plugin_flag,
            f"{output_flag}={targeted_telemetry}",
            f"-fplugin-arg-{plugin.stem}-policy={targeted_policy}",
            str(source),
            "-o",
            str(tmp_path / "targ.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    targeted_instances = {
        json.loads(line)["instance"]
        for line in targeted_telemetry.read_text().splitlines()
        if '"pass-gate"' in line
    }
    assert targeted_instances == {target}

    miss_telemetry = tmp_path / "miss.jsonl"
    miss_policy = tmp_path / "miss.policy"
    miss_policy.write_text("disable_pass=ccp#999999\n", encoding="utf-8")
    subprocess.run(
        [
            "gcc",
            "-O2",
            plugin_flag,
            f"{output_flag}={miss_telemetry}",
            f"-fplugin-arg-{plugin.stem}-policy={miss_policy}",
            str(source),
            "-o",
            str(tmp_path / "miss.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert not [
        line
        for line in miss_telemetry.read_text().splitlines()
        if '"pass-gate"' in line
    ]


def test_conditional_policy_gates_only_matching_functions(native_plugins, tmp_path):
    source = tmp_path / "sized.c"
    source.write_text(
        "int small(void) { return 7; }\n"
        "int big(int n) {\n"
        "    int total = 0;\n"
        "    for (int i = 0; i < n; i++)\n"
        "        for (int j = 0; j < n; j++)\n"
        "            if ((i ^ j) & 1)\n"
        "                total += i * j;\n"
        "            else\n"
        "                total -= i;\n"
        "    return total;\n"
        "}\n"
        "int main(void) { return big(4) == 0 && small() == 7 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    telemetry = tmp_path / "conditional.jsonl"
    policy = tmp_path / "conditional.policy"
    policy.write_text("disable_pass=ccp if basic_blocks<4\n", encoding="utf-8")
    plugin = _plugin_for(native_plugins, "c")

    result = subprocess.run(
        [
            "gcc",
            "-O2",
            f"-fplugin={plugin}",
            f"-fplugin-arg-{plugin.stem}-output={telemetry}",
            f"-fplugin-arg-{plugin.stem}-policy={policy}",
            str(source),
            "-o",
            str(tmp_path / "sized.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in telemetry.read_text().splitlines()]
    gated_functions = {
        event.get("function")
        for event in events
        if event.get("event") == "pass-gate"
        and event.get("pass") == "ccp"
        and event.get("scope") == "rule"
    }
    assert "small" in gated_functions
    assert "big" not in gated_functions


def test_policy_controls_real_gcc_pass_gate(native_plugins, tmp_path):
    source = tmp_path / "program.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    telemetry = tmp_path / "policy.jsonl"
    policy = tmp_path / "policy.txt"
    policy.write_text("disable_pass=evrp\n", encoding="utf-8")
    plugin = _plugin_for(native_plugins, "c")

    result = subprocess.run(
        [
            "gcc",
            "-O2",
            f"-fplugin={plugin}",
            f"-fplugin-arg-{plugin.stem}-output={telemetry}",
            f"-fplugin-arg-{plugin.stem}-policy={policy}",
            str(source),
            "-o",
            str(tmp_path / "program.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert any(
        event.get("event") == "pass-gate"
        and event.get("pass") == "evrp"
        and event.get("scope") == "global"
        and event.get("decision") == "disabled-by-policy"
        for event in events
    )


def test_repair_supervisor_reenters_gcc_with_native_plugin(native_plugins, tmp_path):
    source = tmp_path / "broken.c"
    source.write_text(
        "int main(void) {\n    int value = 1\n    return value - 1;\n}\n",
        encoding="utf-8",
    )
    telemetry = tmp_path / "repair.jsonl"
    program = tmp_path / "repaired.exe"
    plugin = _plugin_for(native_plugins, "c")
    loop = RepairLoop(
        str(source),
        compiler_args=[
            f"-fplugin={plugin}",
            f"-fplugin-arg-{plugin.stem}-output={telemetry}",
        ],
        max_attempts=1,
        output_path=str(program),
        ai=AIEngine(offline=True),
    )

    assert loop.run("repair") is True
    assert subprocess.run([program], check=False).returncode == 0
    events = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert any(
        event.get("event") == "function-ir" and event.get("function") == "main"
        for event in events
    )


def test_telemetry_accumulates_across_translation_units(native_plugins, tmp_path):
    helper = tmp_path / "helper.c"
    main_source = tmp_path / "main.c"
    helper.write_text("int helper(void) { return 7; }\n", encoding="utf-8")
    main_source.write_text(
        "int helper(void);\nint main(void) { return helper() == 7 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    telemetry = tmp_path / "multi-tu.jsonl"
    plugin = _plugin_for(native_plugins, "c")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_compiler.gcc_native",
            "--plugin",
            str(plugin),
            "--telemetry",
            str(telemetry),
            "--",
            "-O2",
            str(helper),
            str(main_source),
            "-o",
            str(tmp_path / "multi-tu.exe"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    functions = {
        json.loads(line).get("function") for line in telemetry.read_text().splitlines()
    }
    assert functions >= {"helper", "main"}


def test_malformed_policy_fails_loudly(native_plugins, tmp_path):
    source = tmp_path / "program.c"
    source.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    policy = tmp_path / "broken.policy"
    policy.write_text("disable_pass=evrp if gimple_statements<\n", encoding="utf-8")
    plugin = _plugin_for(native_plugins, "c")

    result = subprocess.run(
        [
            "gcc",
            f"-fplugin={plugin}",
            f"-fplugin-arg-{plugin.stem}-policy={policy}",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "program.o"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "malformed policy line" in (result.stderr + result.stdout)


def test_launcher_rejects_missing_plugin(tmp_path, capsys):
    code = main(["--plugin", str(tmp_path / "missing.dll"), "--", "--version"])
    assert code == 2
    assert "plugin not found" in capsys.readouterr().err
