from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_plugins(
    output_directory: Path,
    *,
    gcc: str = "gcc",
    cxx: str = "g++",
) -> list[Path]:
    try:
        plugin_directory = Path(
            subprocess.check_output(
                [gcc, "-print-file-name=plugin"], text=True, errors="replace"
            ).strip()
        ).resolve()
    except FileNotFoundError as exc:
        raise RuntimeError(f"GCC driver not found: {gcc}") from exc

    include_directory = plugin_directory / "include"
    if not (include_directory / "gcc-plugin.h").is_file():
        raise RuntimeError(
            f"GCC plugin development headers were not found at {include_directory}"
        )

    source = Path(__file__).resolve().parent / "gcc_plugin" / "ai_native_plugin.cc"
    output_directory.mkdir(parents=True, exist_ok=True)
    common = [
        cxx,
        "-std=c++17",
        "-shared",
        "-fno-rtti",
        "-fno-exceptions",
        f"-I{include_directory}",
    ]

    if os.name == "nt":
        targets = [
            ("ai_native_c.dll", plugin_directory / "cc1.exe.a"),
            ("ai_native_cpp.dll", plugin_directory / "cc1plus.exe.a"),
        ]
    else:
        common.append("-fPIC")
        targets = [("ai_native.so", None)]

    outputs: list[Path] = []
    for filename, import_library in targets:
        if import_library is not None and not import_library.is_file():
            raise RuntimeError(
                f"GCC frontend import library was not found: {import_library}"
            )
        output = (output_directory / filename).resolve()
        command = [*common, "-o", str(output), str(source)]
        if import_library is not None:
            command.append(str(import_library))
        result = subprocess.run(command, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"GCC plugin build failed for {filename} with exit code {result.returncode}"
            )
        outputs.append(output)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gcc-ai-build-plugin", description="Build plugins matched to the host GCC."
    )
    parser.add_argument("--gcc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--output-dir", default="build/gcc-plugin")
    args = parser.parse_args(argv)
    try:
        outputs = build_plugins(
            Path(args.output_dir),
            gcc=args.gcc,
            cxx=args.cxx,
        )
    except RuntimeError as exc:
        print(f"gcc-ai-build-plugin: {exc}", file=sys.stderr)
        return 1
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
