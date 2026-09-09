"""Honest unified L1-L5 certification for the Virtual Computer.

Rule (consecutive-or-nothing, same as every layer): the unified grade is
the MINIMUM of the consecutive layer grades, and it is valid only while
the live integration gate (vertical slice: equivalence + determinism +
conservation + invariants) passes. A failed gate caps the grade at L0
regardless of layer numbers.

Layer grades and their evidence locus:
- microarchitecture: consecutive grade from the committed
  `microarchitecture/artifacts/vmarch_benchmark_phase.json` skill numbers
  via the layer's own `maturity_gate` L1 rule (skill>=0.10, r2>=0.10).
  Currently L0 (skill +0.06) — the binding blocker. Labeled
  committed-artifact evidence, not a live re-run.
- compiler: L1, committed-study evidence (`compiler/docs/PAPER.md`
  Study 1 + `compiler/experiments/study/killer.json`). Not re-run live
  (full study costs 783s + toolchain wall-clock).
- kernel: read from `kernel/reports/levels/results.json`
  (committed full-mode cert; re-runnable via `run_levels --full`),
  guarded by `kernel_evidence_freshness`: the artifact's config hash
  must reproduce under current code AND no grading source may be newer
  than the artifact, else the grade carries a stale-evidence blocker.

The old facade certified L3 for three ok-flags; that was grade
inflation (docs/CRITIQUE.md P0-2) and is removed. `certify_unified`
returns the grade WITH the evidence pointers so any claim is checkable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

LEVEL_NAMES = {0: "none", 1: "Exemplar", 2: "Interpolative",
               3: "Counterfactual", 4: "Mechanistic", 5: "Scientific"}


def _read_json(rel: str) -> tuple[bool, Any]:
    p = REPO_ROOT / rel
    if not p.exists():
        return False, f"missing {rel}"
    try:
        return True, json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


def grade_micro() -> dict:
    ok, payload = _read_json("microarchitecture/artifacts/vmarch_benchmark_phase.json")
    if not ok:
        return {"grade": 0, "evidence": payload, "live": False}
    try:
        skill = payload["results"]["phase"]["skill"]
        s, r2 = float(skill["skill_vs_mean"]), float(skill["r2"])
        l1 = s >= 0.10 and r2 >= 0.10
        return {"grade": 1 if l1 else 0, "live": False,
                "evidence": "microarchitecture/artifacts/vmarch_benchmark_phase.json",
                "metrics": {"skill_vs_mean": s, "r2": r2},
                "blocker": None if l1 else "L1 skill gate: need skill>=0.10 and r2>=0.10"}
    except (KeyError, TypeError, ValueError) as e:
        return {"grade": 0, "evidence": f"unparseable artifact: {e}", "live": False}


def grade_compiler() -> dict:
    ok, _ = _read_json("compiler/experiments/study/killer.json")
    paper = (REPO_ROOT / "compiler" / "docs" / "PAPER.md").exists()
    if not (ok and paper):
        return {"grade": 0, "evidence": "killer.json and/or PAPER.md missing", "live": False}
    return {"grade": 1, "live": False,
            "evidence": "compiler/docs/PAPER.md Study 1 (L1 pass, L2 fail) + "
                        "compiler/experiments/study/killer.json "
                        "(committed study; full re-run costs ~783s, not live)"}


# Sources whose bytes the kernel L2 number depends on. If any is newer
# than the committed artifact, the evidence may be stale.
_KERNEL_GRADE_SOURCES = (
    "kernel/virtual_kernel/levels.py",
    "kernel/virtual_kernel/dynamics.py",
    "kernel/virtual_kernel/ensemble.py",
    "kernel/virtual_kernel/datasets.py",
    "kernel/virtual_kernel/oracle.py",
    "kernel/virtual_kernel/config.py",
    "kernel/virtual_kernel/metrics.py",
    "kernel/benchmarks/run_levels.py",
)
_KERNEL_ARTIFACT = "kernel/reports/levels/results.json"


def _sha12(paths: list) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:12]


def kernel_evidence_freshness() -> dict:
    """Mechanical staleness check on the committed kernel grade.

    Two independent signals: (1) the artifact's recorded config hash
    reproduces under current code (catches VKConfig schema drift);
    (2) no grading source is newer than the artifact (catches
    code-after-benchmark edits). Either failing marks the evidence
    stale with a re-run pointer — mechanism, not just a label.
    """
    from virtual_kernel.config import VKConfig

    art = REPO_ROOT / _KERNEL_ARTIFACT
    srcs = [REPO_ROOT / s for s in _KERNEL_GRADE_SOURCES]
    missing = [str(s) for s in (art, *srcs) if not s.exists()]
    if missing:
        return {"fresh": False, "reason": f"missing: {missing}"}
    import json

    try:
        payload = json.loads(art.read_text(encoding="utf-8"))
        reproduced = VKConfig.from_dict(dict(payload["config"])).hash
        config_reproduces = reproduced == payload.get("config_hash")
    except Exception as e:  # noqa: BLE001 — any failure is a stale signal
        return {"fresh": False, "reason": f"config rehash failed: {type(e).__name__}: {e}"}
    art_mtime = art.stat().st_mtime
    newest = max(s.stat().st_mtime for s in srcs)
    newest_name = max(srcs, key=lambda s: s.stat().st_mtime).name
    sources_predate = newest <= art_mtime
    fresh = bool(config_reproduces and sources_predate)
    detail = {"fresh": fresh, "config_reproduces": bool(config_reproduces),
              "recorded_config_hash": payload.get("config_hash"),
              "reproduced_config_hash": reproduced,
              "sources_predate_artifact": bool(sources_predate),
              "newest_source": f"{newest_name}",
              "code_hash": _sha12([str(s) for s in srcs])}
    if not fresh:
        detail["reason"] = "re-run kernel/benchmarks/run_levels.py --full"
    return detail


def grade_kernel() -> dict:
    ok, payload = _read_json(_KERNEL_ARTIFACT)
    if not ok:
        return {"grade": 0, "evidence": payload, "live": False,
                "blocker": "kernel evidence missing"}
    try:
        freshness = kernel_evidence_freshness()
        grade = int(payload["certified_level"])
        out = {"grade": grade, "live": False,
               "evidence": f"{_KERNEL_ARTIFACT} "
                           f"(mode={payload.get('mode')}, {payload.get('elapsed_s')}s; "
                           "re-runnable: kernel/benchmarks/run_levels.py --full)",
               "freshness": freshness}
        if not freshness.get("fresh"):
            out["blocker"] = (f"kernel evidence possibly stale "
                              f"({freshness.get('reason')})")
        return out
    except (KeyError, TypeError, ValueError) as e:
        return {"grade": 0, "evidence": f"unparseable report: {e}", "live": False,
                "blocker": "kernel evidence unparseable"}


def grade_integration(vertical_report: dict | None) -> dict:
    """Live gate from the vertical slice (the only live measurement)."""
    if not vertical_report:
        return {"grade": 0, "live": True, "evidence": "no vertical report",
                "blocker": "run the vertical slice"}
    gates = vertical_report.get("gates", {})
    required = ("lowering_structural_mem_O3_lt_O0",
                "functional_equivalence_all",
                "cpu_conservation_all_runs",
                "vmarch_equivalence_and_conservation",
                "kernel_invariants_all_trajs", "determinism")
    failed = [g for g in required if not gates.get(g)]
    decided = 0
    try:
        if vertical_report["micro"]["stats"]["verdict"] != "within-noise":
            decided += 1
        decided += sum(1 for w in ("O0", "O3")
                       if vertical_report["kernel"]["stats"][w]["verdict"] != "within-noise")
    except KeyError:
        pass
    if failed:
        return {"grade": 0, "live": True, "gates": gates, "failed": failed,
                "blocker": f"integration gates failed: {failed}"}
    return {"grade": 1, "live": True, "gates": gates,
            "decided_verdicts": decided,
            "note": "I1 = coupled exemplar (measured end-to-end, no cross-layer predictor)"}


def certify_unified(vertical_report: dict | None = None) -> dict:
    micro = grade_micro()
    compiler = grade_compiler()
    kernel = grade_kernel()
    integration = grade_integration(vertical_report)
    grades = [micro["grade"], compiler["grade"], kernel["grade"], integration["grade"]]
    unified = min(grades)
    blockers = [f"{n}: {g.get('blocker')}" for n, g in
                (("micro", micro), ("compiler", compiler),
                 ("kernel", kernel), ("integration", integration))
                if g.get("blocker")]
    return {
        "certified_level": unified,
        "certified_name": ("L%d-%s" % (unified, LEVEL_NAMES[unified]) if unified
                           else "L0-none"),
        "layers": {"microarchitecture": micro, "compiler": compiler,
                   "kernel": kernel, "integration": integration},
        "rule": "min(consecutive layer grades, live integration gate)",
        "blockers": blockers,
    }


def summarize_pipeline_report(report: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible summary now delegating to certify_unified."""
    cert = certify_unified(report.get("vertical"))
    return {"certified_level": cert["certified_level"],
            "certified_name": cert["certified_name"],
            "blockers": cert["blockers"],
            "oracle_budget": report.get("budget"),
            "micro_verdict": (report.get("vertical", {}).get("micro", {})
                              .get("stats", {}).get("verdict")),
            "kernel_verdicts": {w: (report.get("vertical", {}).get("kernel", {})
                                    .get("stats", {}).get(w, {}).get("verdict"))
                                for w in ("O0", "O3")}}
