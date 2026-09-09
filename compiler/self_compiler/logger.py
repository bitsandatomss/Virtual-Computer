from __future__ import annotations

import difflib

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


console = Console()


def print_banner(mode: str, source: str) -> None:
    console.print(
        Panel.fit(
            f"[bold blue]GCC-AI Repair[/bold blue] | Mode: [bold]{mode}[/bold] | Target: {source}",
            border_style="blue",
        )
    )


def print_step(msg: str) -> None:
    console.print(f"[cyan]>[/cyan] {msg}")


def print_success(msg: str) -> None:
    console.print(f"[green]OK[/green] {msg}")


def print_error(msg: str) -> None:
    console.print(f"[red]ERROR[/red] {msg}")


def print_diff(original: str, modified: str) -> None:
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="current",
            tofile="candidate",
        )
    )
    syntax = Syntax(
        diff or "(no textual change)", "diff", theme="monokai", line_numbers=False
    )
    console.print(Panel(syntax, title="Candidate diff"))
