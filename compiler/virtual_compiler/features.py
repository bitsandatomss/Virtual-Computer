"""Feature schema for the compilation latent space.

``E(x, c)`` from context.txt T9: the embedding of observed source ``x``
plus build context ``c`` (flags/policy). Two tiers:

- *static* — always available, computed without gcc (cheap, unlimited);
- *ir* — real GIMPLE facts captured by the AI-Compiler oracle plugin
  (expensive, only on validated branches).

Both tiers share one key space so the surrogate treats missing IR keys
as 0.0 until observed. The schema string versions the contract; any
change refuses to mix with old experience (same discipline as
``gcc-ai.telemetry.v3``).
"""
from __future__ import annotations

import re

SCHEMA = "virtual-compiler.features.v1"

STATIC_KEYS: tuple[str, ...] = (
    "lines", "code_chars", "includes", "functions", "loops",
    "branches", "calls", "mem_ops", "ptr_ops",
    "opt_level", "lto", "march_native", "unroll", "policy_rules",
)

IR_KEYS: tuple[str, ...] = (
    "basic_blocks", "gimple_statements", "phi_nodes", "calls_ir",
    "branches_ir", "edges", "memory_reads", "memory_writes",
    "back_edges", "max_loop_depth", "cyclomatic_complexity",
)

KEYWORD_BITS: tuple[str, ...] = (
    "for", "while", "if", "switch", "case", "goto",
    "malloc", "calloc", "realloc", "free", "memcpy", "strcpy",
    "printf", "fprintf", "qsort", "recursion",
)

FEATURE_KEYS: tuple[str, ...] = (
    STATIC_KEYS[:9] + IR_KEYS
    + ("opt_level", "lto", "march_native", "unroll", "policy_rules")
)


def extract_features(source_text: str, flags: tuple[str, ...] = (),
                     policy_text: str | None = None) -> dict[str, float]:
    """Cheap static features computable without invoking gcc."""
    lines = source_text.splitlines()
    code = "\n".join(
        ln for ln in lines if ln.strip() and not ln.strip().startswith("#"))
    feats: dict[str, float] = {
        "lines": float(len(lines)),
        "code_chars": float(len(code)),
        "includes": float(sum(
            1 for ln in lines if ln.strip().startswith("#include"))),
        "functions": float(len(re.findall(r"\)\s*\{", source_text))),
        "loops": float(len(re.findall(r"\b(for|while)\b", code))),
        "branches": float(len(re.findall(r"\b(if|switch|case|\?)\b", code))),
        "calls": float(len(re.findall(r"[A-Za-z_]\w*\s*\(", code))),
        "mem_ops": float(len(re.findall(
            r"\b(malloc|calloc|realloc|free|memcpy|memmove|strcpy|strcat)\b",
            code))),
        "ptr_ops": float(code.count("*") + code.count("->")),
        "opt_level": 0.0,
        "lto": 0.0,
        "march_native": 0.0,
        "unroll": 0.0,
        "policy_rules": 0.0,
    }
    for flag in flags:
        if flag.startswith("-O"):
            try:
                feats["opt_level"] = float(re.sub(r"[^0-9]", "", flag) or 0)
            except ValueError:
                pass
        if flag == "-flto":
            feats["lto"] = 1.0
        if flag == "-march=native":
            feats["march_native"] = 1.0
        if "unroll" in flag:
            feats["unroll"] = 1.0
    if policy_text:
        rules = [ln for ln in policy_text.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        feats["policy_rules"] = float(len(rules))
    low = source_text.lower()
    for kw in KEYWORD_BITS:
        feats[f"kw_{kw}"] = 1.0 if kw in low else 0.0
    return feats


def feature_vector(feats: dict[str, float]) -> list[float]:
    return [float(feats.get(k, 0.0)) for k in FEATURE_KEYS]


def ir_from_telemetry(events: list[dict]) -> dict[str, float]:
    """Fold raw ``gcc-ai.telemetry.v3`` function-ir events into IR keys.

    Sums function-level facts into whole-program features so experience
    recorded with and without the plugin stays comparable.
    """
    agg: dict[str, float] = {k: 0.0 for k in IR_KEYS}
    rename = {
        "basic_blocks": "basic_blocks",
        "gimple_statements": "gimple_statements",
        "phi_nodes": "phi_nodes",
        "calls": "calls_ir",
        "branches": "branches_ir",
        "edges": "edges",
        "memory_reads": "memory_reads",
        "memory_writes": "memory_writes",
        "back_edges": "back_edges",
        "max_loop_depth": "max_loop_depth",
        "cyclomatic_complexity": "cyclomatic_complexity",
    }
    for ev in events:
        if ev.get("event") != "function-ir":
            continue
        for src, dst in rename.items():
            try:
                val = float(ev.get(src, 0.0))
            except (TypeError, ValueError):
                continue
            agg[dst] = max(agg[dst], val) if src == "max_loop_depth" else (
                agg[dst] + val)
    return agg
