# Virtual Computer — unified executable surrogate of computation

Research-grade unification of the three builds from `context.txt`'s
**Virtual Computer** rung (`Compiler + OS + microarchitecture + I/O`,
context.txt:1365-1378) under one repo, one causal measurement chain,
and per-unit oracle budgets:

| layer | dir | exact oracle | surrogate plans |
|---|---|---|---|
| microarchitecture | `microarchitecture/` (`vmicro` in-order + `vmarch` OoO) | `OracleSim` CPU: scoreboard/ROB/caches/branch predictors/thermal governor | ridge/latent timing surrogate + beam/MCTS over control bundles |
| compiler | `compiler/` (`virtual_compiler` + `self_compiler` gcc lab) | real `gcc` builds (`Oracle`), `StubOracle` offline | runtime surrogate + beam/MCTS/swarm over flags x policy x patch |
| OS kernel | `kernel/` (`virtual_kernel` + `learned_kernel` simulator) | exact `KernelSimulator` (`TruthOracle`) | latent `E/F/D` ensemble + beam/MCTS over sysctl trajectories |

Thesis (context.txt:340-357, surrogate-vs-simulation lens): a
conventional simulator starts from explicit rules, a learned surrogate
from observations. The surrogate proposes, the simulator constrains,
reality (the exact oracle build/run) decides. Computation itself stays
exact and discrete — the surrogate never owns architectural state.

## Honest headline (measured, 2026-09-07, round 2)

**Unified grade: L0-none**, blocked by `micro: L1 skill gate
(skill +0.06 < 0.10)`. The factorial vertical slice passes all live
gates and measures real cross-layer effects over 36 micro + 18 OoO +
864 kernel exact-oracle units: O3 halves memory traffic at every N and
cuts CPU objective (vmicro median −66.95, vmarch −113.35, both
CI-gated *improves*, cross-oracle sign agreement 1.00); tight sysctl
beats loose in both workload regimes at all coupling gains; live gcc
-O3 beats -O0 (+4.12ms, CI [3.39, 6.54], direction confirmed but
`stable=false` so magnitude is noisy). First cross-layer predictor:
LOO skill 0.98 micro (near-deterministic target — machinery works,
claim bounded) and 0.50 kernel (half explained, half hidden demand).
Details: `docs/ROUND2.md`; audit: `docs/CRITIQUE.md`.

## Layout

```
context.txt               source vision (read-only, 1733 lines)
virtual_computer/         unified layer: computer.py (VirtualComputer),
                          vertical.py (factorial slice), lowering.py (v2 map),
                          xsurrogate.py (ridge + LOO skill), gcc_leg.py (live gcc),
                          trust.py (gates), levels.py (honest L1-L5),
                          provenance.py (bound manifests), cli.py, paths.py
microarchitecture/           verbatim layer copy (vmicro/vmarch/docs/tests/...)
compiler/                 verbatim layer copy (virtual_compiler/self_compiler/...)
kernel/                   verbatim layer copy (virtual_kernel/learned_kernel/...)
tests/                    cross-layer integration tests (science, not plumbing)
docs/                     CRITIQUE.md (audit) + DEEPENING.md (round 1) +
                          ROUND2.md (factorial, OoO, gcc, predictor) + UNIFICATION.md
demo_virtual_computer.py  factorial slice + honest grade + manifest
```

Layer copies are verbatim (only cache dirs dropped) so each layer's own
docs, tests, and CLIs keep working from their subdirectory. The facade
only *adds* `microarchitecture/`, `compiler/`, `kernel/` to `sys.path`
(`virtual_computer/paths.py`) and composes the three environments.

## Use

```powershell
python -m virtual_computer.cli info
python -m virtual_computer.cli vertical --emit reports/unified_manifest.jsonl
python -m virtual_computer.cli vertical --gcc   # + live-gcc -O0/-O3 leg
python -m virtual_computer.cli certify
python demo_virtual_computer.py
python -m pytest tests -q
```

Per-layer entry points still work from the repo root, e.g.:

```powershell
python -m pytest microarchitecture/tests kernel/tests compiler/tests -q
python microarchitecture/demo_vmicro.py --family mixed --seed 0
python kernel/demo_virtual_kernel.py
```

## Common operation

`observe -> learn -> represent -> branch -> intervene -> validate`,
closed into `observe -> learn -> simulate -> probe -> observe`, with
per-unit oracle budgets (compiler view vs CPU run vs sim step are
incommensurable and reported as a vector, never summed). Every virtual
claim is checked against the exact CPU / build / kernel simulator.
Unified L1-L5 = min(consecutive layer grades, live integration gate)
(`virtual_computer/levels.py`) — currently L0, blocker named above.

## Discipline

Only exact-oracle measurements count as results. Surrogate errors are
reported (predicted vs oracle side by side), oracle-call costs next to
every reward. No backend/OS/hardware behaviour is faked — I/O device
models are a named future oracle extension.
