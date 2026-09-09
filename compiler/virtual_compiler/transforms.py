"""Semantics-first source interventions (WHAT_IT_IS §7).

A compiler lab perturbs *code*, not just flags. Each transform maps
source → new source, or None when inapplicable (honest no-op, never a
forced edit). Two safety classes:

- ``safe`` — advisory hints the compiler may ignore; semantics preserved
  by construction (pragmas, attributes, branch hints).
- ``gated`` — may change behavior if the code violates the assumed
  precondition (e.g. ``restrict`` under aliasing); these are hypotheses
  the differential gate must clear, not facts.

First applicable site only; the site is recorded in the branch action
string so trajectories stay legible.
"""
from __future__ import annotations

import re

_FUNC_DEF = re.compile(
    r"(?m)^(?P<head>(?:static\s+)?[A-Za-z_][\w\s\*]*?\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}]*)\)\s*\{)")


def _unroll(source: str) -> str | None:
    m = re.search(r"(?m)^([ \t]*)for\s*\(", source)
    if not m:
        return None
    indent = m.group(1)
    return (source[:m.start()] + indent + "#pragma GCC unroll 4\n"
            + source[m.start():])


def _expect(source: str) -> str | None:
    m = re.search(r"if\s*\(([^();]*)\)", source)
    if not m:
        return None
    cond = m.group(1).strip()
    if not cond or "__builtin_expect" in cond:
        return None
    return (source[:m.start()] + "if (__builtin_expect(!!(" + cond
            + "), 1))" + source[m.end():])


def _attribute(source: str, attr: str) -> str | None:
    m = _FUNC_DEF.search(source)
    if not m:
        return None
    if attr in m.group("head"):
        return None
    return source[:m.start()] + attr + " " + source[m.start():]


def _always_inline(source: str) -> str | None:
    return _attribute(source, "__attribute__((always_inline))")


def _noinline(source: str) -> str | None:
    return _attribute(source, "__attribute__((noinline))")


def _restrict(source: str) -> str | None:
    m = _FUNC_DEF.search(source)
    if not m:
        return None
    params = m.group("params")
    if "restrict" in params:
        return None
    pm = re.search(r"([A-Za-z_][\w\s]*\*)\s*([A-Za-z_]\w*)", params)
    if not pm:
        return None  # no single-level pointer parameter; stay out
    new_params = (params[:pm.start()] + pm.group(1) + "restrict "
                  + pm.group(2) + params[pm.end():])
    start = m.start() + m.group("head").find(params)
    return source[:start] + new_params + source[start + len(params):]


TRANSFORMS: dict[str, dict] = {
    "unroll": {"fn": _unroll, "kind": "safe",
               "desc": "#pragma GCC unroll before first for-loop"},
    "expect": {"fn": _expect, "kind": "safe",
               "desc": "__builtin_expect on first if-condition"},
    "always_inline": {"fn": _always_inline, "kind": "safe",
                      "desc": "always_inline attribute on first function"},
    "noinline": {"fn": _noinline, "kind": "safe",
                 "desc": "noinline attribute on first function"},
    "restrict": {"fn": _restrict, "kind": "gated",
                 "desc": "restrict on first pointer parameter "
                         "(requires differential clearance)"},
}


def apply_transform(source: str, name: str) -> str | None:
    """Apply one transform; None when inapplicable."""
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform: {name!r}")
    return TRANSFORMS[name]["fn"](source)


def available(source: str) -> list[str]:
    """Names applicable to this source right now."""
    return [n for n, t in TRANSFORMS.items() if t["fn"](source) is not None]
