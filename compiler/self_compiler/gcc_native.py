from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("gcc-ai-native")
    except Exception:
        return "0.0.0+unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcc-ai",
        description="Load the AI-native policy plugin into a real GCC frontend.",
    )
    parser.add_argument("--version", action="version", version=_version())
    parser.add_argument("--compiler", default="gcc", help="GCC driver executable")
    parser.add_argument("--plugin", help="ai_native_c/ai_native_cpp plugin path")
    parser.add_argument("--telemetry", help="write compiler IR telemetry as JSON Lines")
    parser.add_argument("--policy", help="conservative compiler-pass policy file")
    parser.add_argument(
        "compiler_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded unchanged to GCC after --",
    )
    return parser


def _default_plugin(compiler: str) -> Path:
    if os.name == "nt":
        filename = (
            "ai_native_cpp.dll" if "++" in Path(compiler).name else "ai_native_c.dll"
        )
    else:
        filename = "ai_native.so"
    working_tree_plugin = Path.cwd() / "build" / "gcc-plugin" / filename
    package_tree_plugin = (
        Path(__file__).resolve().parents[1] / "build" / "gcc-plugin" / filename
    )
    if working_tree_plugin.is_file():
        return working_tree_plugin
    return package_tree_plugin


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forwarded = list(args.compiler_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not forwarded:
        build_parser().error("missing GCC arguments after --")

    plugin = Path(
        args.plugin
        or os.environ.get("GCC_AI_PLUGIN", "")
        or _default_plugin(args.compiler)
    ).resolve()
    if not plugin.is_file():
        print(
            f"gcc-ai: plugin not found: {plugin}\n"
            "Build it with scripts/build_gcc_plugin.ps1 or pass --plugin.",
            file=sys.stderr,
        )
        return 2

    plugin_key = plugin.stem
    command = [args.compiler, f"-fplugin={plugin}"]
    if args.telemetry:
        telemetry = Path(args.telemetry).resolve()
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        telemetry.unlink(missing_ok=True)
        command.append(f"-fplugin-arg-{plugin_key}-output={telemetry}")
    if args.policy:
        policy = Path(args.policy).resolve()
        if not policy.is_file():
            print(f"gcc-ai: policy not found: {policy}", file=sys.stderr)
            return 2
        command.append(f"-fplugin-arg-{plugin_key}-policy={policy}")
    command.extend(forwarded)

    try:
        return subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        print(f"gcc-ai: compiler not found: {args.compiler}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
