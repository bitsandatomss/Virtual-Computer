"""Simulator-side constraints (VISION N2/N3.2).

The triad says the *simulator* supplies known structure and constraints;
the surrogate only proposes within them. Concretely for gcc:

- contradiction / unknown-flag / duplicate-`-O` checks are static and
  always available;
- flag-implication facts (is `-fX` already on at `-O2`? is `-fno-Y`
  vacuous?) are queried from the *real simulator* via
  ``gcc -Q --help=optimizers`` and never hardcoded — implication sets
  drift across GCC versions, and a stale table would be exactly the kind
  of fake knowledge this project refuses to ship.

`check_state` returns findings; `VirtualCompiler.check` exposes them;
the campaign skips error-level branches before spending oracle budget.
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .state import CompilationState

KNOWN_O_LEVELS = ("-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Og")

COMMON_EXTRA = (
    "-flto", "-march=native", "-g", "-g3", "-fno-omit-frame-pointer",
    "-fstack-protector-strong", "-D_FORTIFY_SOURCE=2",
)


def _allowed_flags() -> set[str]:
    from .environment import FLAG_ACTIONS, FLAG_TUNE_POOL
    allowed: set[str] = set()
    for preset in FLAG_ACTIONS.values():
        allowed.update(preset)
    for base in FLAG_TUNE_POOL:
        allowed.add("-" + base)
        core = base[1:] if base.startswith("f") else base
        allowed.add("-fno-" + core)
    allowed.update(KNOWN_O_LEVELS)
    allowed.update(COMMON_EXTRA)
    return allowed


def check_flags(flags: Sequence[str]) -> list[dict]:
    """Static consistency checks over one flag tuple."""
    findings: list[dict] = []
    seen_o = [f for f in flags if f in KNOWN_O_LEVELS]
    if len(seen_o) > 1:
        findings.append({"level": "warning", "rule": "duplicate-opt-level",
                         "detail": f"multiple -O levels {seen_o}; "
                                   "last one wins in gcc"})
    pos, neg = set(), set()
    for f in flags:
        if f.startswith("-fno-"):
            neg.add(f[5:])
        elif f.startswith("-f"):
            pos.add(f[2:])
    for name in sorted(pos & neg):
        findings.append({"level": "error", "rule": "contradictory-flags",
                         "detail": f"-f{name} and -fno-{name} both present; "
                                   "gcc applies last-wins, silently"})
    allowed = _allowed_flags()
    for f in flags:
        if f.startswith("-f") and f not in allowed and f not in KNOWN_O_LEVELS:
            findings.append({"level": "warning", "rule": "unknown-flag",
                             "detail": f"{f} not in the known vocabulary; "
                                       "possible typo"})
    if len(flags) != len(set(flags)):
        findings.append({"level": "warning", "rule": "duplicate-flag",
                         "detail": "repeated flag; harmless but noisy"})
    return findings


@lru_cache(maxsize=32)
def query_enabled(compiler: str = "gcc",
                  opt: str = "-O2") -> frozenset[str] | None:
    """Ask the real simulator which `-f` flags `opt` enables.

    Returns a frozenset of enabled flag names (e.g. ``-funroll-loops``)
    or None when gcc is unavailable/unparseable. Cached per pair.
    """
    try:
        proc = subprocess.run(
            [compiler, "-Q", opt, "--help=optimizers"],
            capture_output=True, text=True, timeout=30.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    enabled = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("-f") \
                and parts[1] == "[enabled]":
            enabled.add(parts[0])
    return frozenset(enabled) if enabled else None


def check_redundancy(flags: Sequence[str],
                     compiler: str = "gcc") -> list[dict]:
    """Vacuous/redundant tune detection against the simulator's own data."""
    base = next((f for f in flags if f in KNOWN_O_LEVELS), "-O2")
    enabled = query_enabled(compiler, base)
    if enabled is None:
        return []  # unqueryable simulator: stay silent, not wrong
    findings: list[dict] = []
    tune = [f for f in flags if f.startswith("-f") and f not in enabled
            and not f.startswith("-fno-")]
    _ = tune
    for f in flags:
        if f.startswith("-fno-") and ("-f" + f[5:]) not in enabled \
                and f[5:] not in ("lto",):
            # disabling something the base level never enabled
            findings.append({"level": "warning",
                             "rule": "vacuous-disable",
                             "detail": f"{f} disables a pass not enabled "
                                       f"by {base}; likely no effect"})
        if f.startswith("-f") and not f.startswith("-fno-") \
                and f in enabled and f not in KNOWN_O_LEVELS:
            # find whether it came from tune rather than the base preset
            findings.append({"level": "info", "rule": "already-enabled",
                             "detail": f"{f} already enabled by {base}; "
                                       "explicit form is redundant"})
    return findings


def check_state(state: "CompilationState",
                compiler: str = "gcc",
                query_simulator: bool = True) -> list[dict]:
    """All simulator-side checks for one compilation state."""
    out = check_flags(state.flags)
    if not query_simulator:
        return out  # stub oracle: no simulator present to query
    try:
        out.extend(check_redundancy(state.flags, compiler))
    except Exception:
        pass  # redundancy is advisory; never block on it
    return out
