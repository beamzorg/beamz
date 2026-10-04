# Automatic CUDA layout and shell selection — 2026-09-18

**Superseded default:** automatic mode now uses a
[geometry estimate without calibration](cuda-layout-prediction-2026-09-18.md).
The measured workflow and historical results below remain available through
explicit `BEAMZ_CUDA_AUTOTUNE=calibrate`.

Implemented automatic first-use calibration plus a persistent lookup cache. The selector measures the three cyclic storage orders and two existing CPML shell tiles. It retains canonical storage unless a candidate reduces median runtime by at least 3% and wins both timing halves. Precision, physical shape, CPML thickness, source timing and monitor sampling remain unchanged.

## Fresh automatic-branch versus origin/main comparison

Compared fetched main `c5fe0d8833ac6539241ae9d3c98445e5a7305786` against the local branch with automatic selection enabled and no manual layout/kernel overrides. Each shape ran main/automatic/automatic/main in fresh processes, with four warmups and nine timed 256-step executions per process. Values below average the two process medians. Identical inputs: FP32, lossless material, CPML12, one mode source, two compact mode monitors, three frequencies. All conversion work is timed; setup/calibration/compilation are excluded.

| Physical domain (z,y,x) | Main GCUPS | Automatic branch GCUPS | Change | Chosen axes / shell | Calibration cost |
|---|---:|---:|---:|---|---:|
| 1024×256×64 | 7.283 | 8.555 | +17.5% | 120 / 64x4 | 37.3 s |
| 128×256×512 | 8.386 | 8.772 | +4.6% | 012 / 64x4 | 33.5 s |
| 257×193×341 (smooth) | 6.965 | 8.277 | +18.8% | 012 / 64x4 | 35.5 s |

The second automatic process hit the persistent cache in all three cases; it did not repeat calibration. Initial states, coefficient arrays and compiled source arrays matched main exactly for every run. All six candidate layouts/tiles in every calibration passed bitwise complete-state validation against the branch canonical baseline.

## Validation and limits

- 139 CPU tests passed across tuning policy, cache/backend contracts, runtime contracts and simulation architecture.
- Four seeded hardware tests passed for both GPU/default and CPU setup. They cover 33 and 519 steps, including chunk boundaries and an odd tail, and check all six candidates, persistent lookup and caller-state preservation.
- A preceding same-process matrix verified selected-versus-default complete-state equality and isolated the disabled policy from the automatically selected cached plan.
- Branch-versus-main still has the earlier small CPML tolerance failures in the binary-material cases. Field/monitor comparisons pass. Automatic selection adds no numerical difference relative to the branch baseline, but this is not proof of full equivalence to main.
- The automatic scope is one RTX3090, >=8,388,608 cells, at least 32 steps, uniform 3D lossless diagonal materials, CPML12 and native-compatible slab sources/DFT monitors. Other workloads keep their existing dispatch. This is not a guarantee of beating main for arbitrary domains or schedules.
- First-use calibration costs about half a minute for these cases and is additional startup work. It does not pay back within a single short run. The exact-workload cache favors repeated simulations; changed physical requests recalibrate.
- GPU/driver/power limit, exact request and solver implementation changes invalidate cached measurements. Manual CUDA overrides take precedence. Cache writes are atomic; unwritable caches do not prevent execution.

Peak observed GPU temperature was 73 C; minimum free GPU memory was 19009 MiB. Power remained 370 W. GPU benchmark processes ran sequentially. Nothing was pushed; native CUDA kernels and FP32 defaults were unchanged by this work.

## Use

Ordinary `sim.compile(num_steps=256, backend="cuda_streamed")` selects automatically when eligible. Inspect `beamz.simulation.cuda.tuning.tuning_report(program)` for the choice, measurements, cache hit and calibration cost. `BEAMZ_CUDA_AUTOTUNE=off` disables tuning; `refresh` recalibrates after clearing the program cache or restarting. `BEAMZ_CUDA_TUNING_CACHE` overrides the persistent cache directory.

[Fresh comparison artifacts](rtx3090-2026-09-18-origin-main-auto-comparison/) contain raw timings, input fingerprints, numerical comparison details, telemetry and solver/source provenance. [Initial selector validation](rtx3090-2026-09-18-autotune/) contains the six-candidate profiles and cache/disable checks.

## Long-run completion

The selector now calibrates at most 256 steps for longer requests, matching the native chunk size, while retaining the original program horizon. This bounds calibration cost independently of total simulation duration. The 256-step table above predates this scope extension; its candidate kernels and short-run execution are unchanged. Long-run tails and source activity can still affect which candidate is fastest.

A fresh main/automatic/automatic/main comparison of 1024×256×64 for **1,025 steps** measured **7.289 -> 8.557 GCUPS (+17.4%)**, selecting 120 / 64x4. The first run calibrated 256 steps in 37.3 seconds; the next process hit the persistent cache. Both completed all 1,025 requested steps. A separate canonical-branch execution matched every final state leaf bitwise.

Main-versus-branch CPML tolerance failures grow to 773,040 entries in this longer case, with maximum absolute error divided by its leaf peak of 6.72e-05. Fields and monitors still pass the existing tolerance. The equality to the canonical branch shows these differences are not introduced by automatic layout selection. Full equivalence to main remains unresolved.

[Long-run raw evidence and final solver snapshot](rtx3090-2026-09-18-origin-main-auto-long/). Final checks: 139 CPU tests and four seeded hardware tests passed; lint and whitespace checks passed.
