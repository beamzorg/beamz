# Oriented y-face tiles for the complete CPML pair

The retained y-face specialization improves the experimental complete two-step
CPML schedule by about 11–13% on the tested flat domains. It does not beat the
ordinary schedule: same-physics cyclic storage already reaches about 8.2–8.4
logical GCUPS there, versus about 7.0 for the improved pair. Consistent >=9
GCUPS remains unachieved. FP32 and ordinary stepping remain the defaults.

## Bottleneck and retained change

Storage order 120 turns physical (z,y,x)=1024x256x64 into stored 256x64x1024.
The large absorber faces are then y faces. A kernel trace of the previous
oriented CPML pair attributes 41.6% of device-kernel time to mask 2 (y faces),
29.9% to the interior and 10.3% to yz edges. These are timings for the paired
schedule, not an attribution of the ordinary schedule's 6.8-GCUPS penalty.

Replace the oriented y-face 32x4x16/256-thread tile with 16x15x16/384 threads.
The larger face-normal extent reduces the number of tiles across each shell
and duplicated temporal halos. Tile shape alone gave only about 2% on binary
flat cases and no gain on the smooth flat control. That version used 58
registers per thread for packed FP32 material, limiting residency to two blocks.
A launch-bound minimum of three blocks reduces that kernel to 52 registers
without local-memory spills. It can now fit three blocks on SM86 with its
32,448 bytes of dynamic shared memory per block. Launch-resource occupancy
estimates are not measurements of achieved occupancy or hardware stalls.

The existing 16x8x16 interior, 15x16x16/384-thread x faces, 32x4x16 z faces,
and 16x8x4 edges/corners remain. Arithmetic, ownership, source injection,
intermediate monitor sampling and physical CPML thickness do not change.
A 32x15x16/640-thread y-face alternative reached only 6.31 GCUPS on narrow64
and was not retained; its private parser option is absent from production.

## Matched performance controls

Lossless simulations, FP32 fields/auxiliary state, exactly 12-cell CPML, one
mode source, two compact mode monitors and three frequencies. Each fresh
process performs four warmups and nine timed 256-step executions. Binary
variants are interleaved baseline/candidate/candidate/baseline; the first
narrow64 comparison also brackets the wider rejected tile. Reported values
are means of two process medians, not confidence intervals. Storage conversions
are included; compilation/setup/host validation copies are excluded.

| Physical shape (z,y,x), material | Storage axes | Previous pair | Retained pair | Change |
|---|---|---:|---:|---:|
| 1024x256x64, binary | 120 | 6.163 | 6.967 | +13.1% |
| 1024x256x65, binary | 120 | 6.296 | 7.018 | +11.5% |
| 1024x256x64, smooth | 120 | 6.359 | 7.059 | +11.0% |
| 128x256x512, binary | 012 | 7.442 | 7.662 | +2.9% |
| 257x193x341, smooth | 012 | 6.988 | 7.150 | +2.3% |

All 22 runs, including the rejected wide tile, matched final complete-state
hashes within each physical case. The irregular control has visible process
drift (candidate 7.024–7.276); its small gain needs caution. Peak temperature
was 71 C and minimum free memory 18,710 MiB. GPU power settings were unchanged.
The stronger flat-domain gains repeat across neighboring widths and materials;
this is not evidence for universal shape-independent throughput.

## Validation and artifacts

The register-bounded candidate passed 22 seeded hardware cases covering cyclic
layouts, FP32/BF16 auxiliary storage, mode/ragged-field monitors, source-only
and source-free graphs, padding, odd 33-step runs and two-step continuation.
The discarded wider candidate separately passed 16 independent ordinary-CUDA
comparisons. Compute Sanitizer reported zero memory errors and zero race
hazards for both candidate tiles across material and auxiliary-storage formats.
These sanitizer cases exercise native kernels, not the entire Python stack.

Raw samples, full-state hashes, native binary/source fingerprints and telemetry:
[initial tile options](rtx3090-2026-09-18-cpml-yface-options/),
[shape/material controls](rtx3090-2026-09-18-cpml-yface-controls/).
Private experiment scripts and detailed logs remain under `.cache/perf/`:
`run-cpml-yface-options.py`, `run-cpml-yface-controls.py`,
`cpml-yface-regcap-smoke.log`, `cpml-yface-memcheck.log`,
`cpml-yface-racecheck.log`, and `yface-traces/kernel-summary.json`.
The clean retained build removes the unused wider alternative and its parser
option. It passed the same 22 seeded checks again (73.06 seconds). Its final
unpaired narrow64 repeat measured 6.785 GCUPS and matched the original baseline
state exactly; that repeat is lower than the matched candidate mean above and
should not be hidden by the aggregate. A final kernel trace records y-face time
at 35.1% of total kernel time, with 52 registers, 32,448 dynamic shared bytes,
and 75% launch-resource occupancy (three blocks), versus the uncapped tile's
50% (two blocks). Remaining interior and yz-edge shares are 33.2% and 11.4%.
[Retained-build evidence](rtx3090-2026-09-18-cpml-yface-retained/) includes that
repeat, profile summary, test and sanitizer logs.
Only SM86 has been rebuilt and tested in this follow-up; no new multiarchitecture
release validation is claimed. Nothing was pushed.

## Next temporal experiment

The next candidate is to combine the narrow stored-y interior and its two CPML
faces into one two-step tile, keeping compact auxiliary storage only for the
physical absorber rows. This could remove duplicated interior/face halos.
Its larger shared-memory footprint and block synchronization may offset that
saving, so viability and speed remain unproven. The detailed unimplemented
proposal is `.cache/perf/full-y-tile-plan.md`. The faster ordinary cyclic layout
remains the performance reference for judging it.
