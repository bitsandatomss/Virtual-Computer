"""Live-toolchain compiler leg: real gcc -O0 vs -O3 on the vertical C kernel.

Unlike the stub-budgeted preset branches, this leg produces genuine
compiler evidence through the layer's own machinery
(`metrology.build_and_compare`: 2 builds + paired lockstep comparison
with seeded bootstrap CI + gated verdict + differential equivalence).
Runnable only where a C toolchain exists (here: gcc 14.1/MSYS2);
otherwise returns a skipped record with reason — never faked.

The harness scales the vertical dot-product (N=256, REPEAT=3000) so the
-O3 vectorizer has a measurable gap over -O0 within milliseconds.
"""
from __future__ import annotations

import shutil
from typing import Any

HARNESS_C = """#include <stdio.h>
#define N 256
#ifndef REPEAT
#define REPEAT 3000
#endif
static int a[N], b[N];
int main(void) {
  for (int i = 0; i < N; i++) { a[i] = i * 3 + 1; b[i] = i * 5 + 2; }
  long long total = 0;
  for (long r = 0; r < REPEAT; r++) {
    long long s = 0;
    for (int i = 0; i < N; i++) s += (long long)a[i] * b[i];
    total += s;
  }
  printf("%lld\\n", total);
  return 0;
}
"""


def toolchain_available(compiler: str = "gcc") -> bool:
    return shutil.which(compiler) is not None


def run_gcc_leg(compiler: str = "gcc", runs: int = 7,
                differential_trials: int = 4,
                repeat: int = 3000) -> dict[str, Any]:
    if not toolchain_available(compiler):
        return {"status": "skipped",
                "reason": f"toolchain {compiler!r} not on PATH"}
    from virtual_compiler.metrology import build_and_compare

    src = HARNESS_C.replace("3000", str(repeat))
    try:
        vc = build_and_compare(src, ("-O0",), ("-O3",), compiler=compiler,
                               runs=runs, differential_trials=differential_trials)
    except Exception as e:  # toolchain failures are data, not crashes
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
    if vc.error or vc.paired is None:
        return {"status": "error", "error": vc.error or "no paired result",
                "builds": vc.builds}
    p = vc.paired
    return {"status": "measured", "builds": vc.builds,
            "median_O0_ms": p.median_a_ms, "median_O3_ms": p.median_b_ms,
            "delta_ms": p.paired_median_delta_ms,
            "ci_low": p.ci_low_ms, "ci_high": p.ci_high_ms,
            "verdict": p.verdict, "stable": p.stable,
            "output_match": p.output_match,
            "differential_equivalent": vc.equivalent,
            "evidence": "LIVE gcc (paired lockstep + bootstrap CI + diff gate)"}
