"""Metrology: paired, interleaved, CI-gated comparison (CRITIQUE G3).

Our own oracle layer (``self_compiler.policy_lab``) demands medians with
seeded bootstrap CIs over *paired* deltas from *interleaved* sampling —
because machine drift must hit every variant equally. ``validate()``'s
sequential runs=3 never met that bar. This module is the bar:

- ``paired_compare``: interleave A/B/A/B… runs of two binaries in lockstep
  cycles, bootstrap-CI the paired median delta, gate the verdict.
- ``build_and_compare``: build two flag-configs of one source, then
  paired-compare *and* differential-gate them. The benchmark's
  winner-vs-stock verification runs through here.

Timing noise is the reason MLGO attacked size before speed; since we
attack speed, every claim passes through this module or is labeled
unmeasured.
"""
from __future__ import annotations

import os
import random
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SEED = 20260905


def bootstrap_median_ci(deltas: Sequence[float],
                        resamples: int = BOOTSTRAP_RESAMPLES,
                        seed: int = DEFAULT_SEED) -> tuple[float, float]:
    """Seeded bootstrap CI for the median of paired deltas."""
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0)
    base = sorted(deltas)
    mid = len(base) // 2
    if n == 1:
        return (base[0], base[0])
    meds = []
    for _ in range(resamples):
        sample = sorted(rng.choice(base) for _ in range(n))
        meds.append(sample[len(sample) // 2] if n % 2 else
                    (sample[mid - 1] + sample[mid]) / 2.0)
    meds.sort()
    lo = meds[int(0.025 * resamples)]
    hi = meds[int(0.975 * resamples) - 1]
    return (lo, hi)


def gate_verdict(median_delta: float, ci_low: float, ci_high: float,
                 margin: float = 0.02, baseline: float = 1.0) -> str:
    """CI-gated verdict on (baseline − candidate)/baseline improvement."""
    rel = median_delta / baseline if baseline > 0 else 0.0
    rel_lo = ci_low / baseline if baseline > 0 else 0.0
    if rel > margin and rel_lo > 0:
        return "improves"
    if rel < -margin and ci_high / (baseline or 1.0) < 0:
        return "regresses"
    return "within-noise"


@dataclass
class PairedComparison:
    median_a_ms: float
    median_b_ms: float
    paired_median_delta_ms: float
    ci_low_ms: float
    ci_high_ms: float
    verdict: str  # from B's perspective vs A at `margin`
    rounds: int
    output_match: bool
    spread_a: float = 0.0
    spread_b: float = 0.0
    stable: bool | None = None


def guarded_median(samples: Sequence[float], drop: int = 1,
                   guard: float = 0.05) -> dict:
    """PolyBench-grade stability guard (`time_benchmark.sh` logic).

    Sort, drop `drop` extremes per side, require
    (max-min)/median <= guard over the core. Unstable input must
    abstain downstream, never average out.
    """
    s = sorted(samples)
    if len(s) < 3:
        med = s[len(s) // 2] if s else 0.0
        return {"median": med, "spread": 0.0, "stable": False,
                "reason": "insufficient-samples"}
    core = s[drop:-drop] if len(s) > 2 * drop else s
    med = core[len(core) // 2] if len(core) % 2 else (
        core[len(core) // 2 - 1] + core[len(core) // 2]) / 2.0
    spread = (max(core) - min(core)) / med if med > 0 else 0.0
    return {"median": med, "spread": spread, "stable": spread <= guard,
            "reason": "ok" if spread <= guard else "exceeds-guard"}


def paired_compare(binary_a: str, binary_b: str, runs: int = 7,
                   warmups: int = 2, timeout: float = 10.0,
                   margin: float = 0.02,
                   seed: int = DEFAULT_SEED) -> PairedComparison:
    """Interleaved A/B measurement with a gated verdict (B vs A)."""
    from self_compiler.verifier import verify

    if runs < 1 or warmups < 0:
        raise ValueError("runs must be positive, warmups non-negative.")
    outs: list[tuple[str, str]] = []
    deltas: list[float] = []
    a_samples: list[float] = []
    b_samples: list[float] = []
    for _ in range(warmups):
        verify(binary_a, None, timeout)
        verify(binary_b, None, timeout)
    for _ in range(runs):
        ra = verify(binary_a, None, timeout)
        rb = verify(binary_b, None, timeout)
        if not ra.passed or not rb.passed:
            raise RuntimeError(
                f"paired comparison failed: A={ra.output[-200:]!r} "
                f"B={rb.output[-200:]!r}")
        a_samples.append(ra.elapsed_ms)
        b_samples.append(rb.elapsed_ms)
        deltas.append(ra.elapsed_ms - rb.elapsed_ms)
        outs.append((ra.output, rb.output))
    med_a = statistics.median(a_samples)
    med_b = statistics.median(b_samples)
    ga = guarded_median(a_samples, drop=1 if runs >= 5 else 0)
    gb = guarded_median(b_samples, drop=1 if runs >= 5 else 0)
    srt = sorted(deltas)
    mid = len(srt) // 2
    med_delta = (srt[mid] if len(srt) % 2
                 else (srt[mid - 1] + srt[mid]) / 2.0)
    ci_lo, ci_hi = bootstrap_median_ci(deltas, seed=seed)
    return PairedComparison(
        median_a_ms=med_a, median_b_ms=med_b,
        paired_median_delta_ms=med_delta, ci_low_ms=ci_lo, ci_high_ms=ci_hi,
        verdict=gate_verdict(med_delta, ci_lo, ci_hi, margin, med_a),
        rounds=runs,
        output_match=all(a == b for a, b in outs),
        spread_a=ga["spread"], spread_b=gb["spread"],
        stable=bool(ga["stable"] and gb["stable"]))


@dataclass
class VerifiedComparison:
    equivalent: bool | None
    paired: PairedComparison | None
    builds: int = 0
    error: str = ""


def build_and_compare(source_text: str, flags_a: Sequence[str],
                      flags_b: Sequence[str], compiler: str = "gcc",
                      runs: int = 7, differential_trials: int = 8,
                      timeout: float = 30.0,
                      margin: float = 0.02) -> VerifiedComparison:
    """Build two configs, paired-compare and differential-gate them."""
    from self_compiler.compiler import compile_source

    suffix = ".exe" if os.name == "nt" else ""
    with tempfile.TemporaryDirectory(prefix="vcc-verify-") as work:
        src = Path(work) / "prog.c"
        src.write_text(source_text, encoding="utf-8")
        b_a = Path(work) / f"a{suffix}"
        b_b = Path(work) / f"b{suffix}"
        ra = compile_source(str(src), compiler=compiler, flags=list(flags_a),
                            output_path=str(b_a), timeout=timeout)
        rb = compile_source(str(src), compiler=compiler, flags=list(flags_b),
                            output_path=str(b_b), timeout=timeout)
        if not ra.ok or not rb.ok:
            return VerifiedComparison(None, None, builds=2,
                                      error="build failed")
        try:
            paired = paired_compare(str(b_a), str(b_b), runs=runs,
                                    timeout=10.0, margin=margin)
        except RuntimeError as exc:
            return VerifiedComparison(None, None, builds=2, error=str(exc))
        try:
            from self_compiler.differential import run_differential
            equiv: bool | None = bool(run_differential(
                b_a, b_b,
                trials=differential_trials).equivalent)
        except (OSError, ImportError):
            equiv = None
        return VerifiedComparison(equiv, paired, builds=2)
