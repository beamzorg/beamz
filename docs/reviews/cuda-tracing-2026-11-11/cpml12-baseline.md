# Corrected CPML12 baseline and first launch-variant screen

2026-11-11, RTX 3090 24 GiB, **250 W limit unchanged**, CUDA 13.3, SM86,
float32 fields and CPML state, precise math. These measurements precede native
candidate installation. All fixtures use twelve CPML cells per face.

**Correctness defect found after this measurement phase and fixed during the run:** the frozen
baseline fails an existing seeded mode-source/windowed-DFT hardware test
(`test_mstep_mode_source_and_windowed_interval_dft`): thirty-five selected
hardware tests pass and one fails. Native investigation identifies a specialized
packed-core decoder that interprets rank-two constant-block material headers as
legacy byte IDs. The zero-state benchmark cases below passed their own complete
state comparisons, but do not establish general correctness of that binary.
Treat measurements in the initial-baseline sections as pre-fix diagnostic
records. The fixed-control and candidate sections later in this document
supersede them. Control and optimization candidates were rebuilt with the same
fix, passed seeded gates, and replayed the fixtures before performance
acceptance. The hardware log is
`.cache/cuda-autoresearch-2026-11-11/baseline-hardware-gates.log`.

The native component is the initial source-built ABI-30 binary:
`5f0be93afd786be9f935f6b2f8a8b2c5029b0d9c55068092f8fdd973f915cc76`.
Its frozen engine source SHA is
`2587bfa9266b09f9cba2072d309d0557fc86f634cbec0e4f839fa4df3713b73d`.
The baseline source archive and binary are retained under
`.cache/cuda-autoresearch-2026-11-11/baseline-run/baseline/`.
Native source edits made while later measurements were running did not change
the installed binary. Build provenance must use that frozen snapshot, rather
than interpreting a contemporaneous working-tree source hash as a binary hash.

## Historical workload recipe recovered

The retained workload builder is byte-identical to the October report's
builder. The 160×320×384 binary scene uses a mode source, two mode monitors,
three frequencies, CPML12, and 256 steps. Nine synchronized executable samples,
four warmups plus two warmup seconds, and three public samples were used.

| Current schedule | GCUPS | Eligible core | Active tiles |
|---|---:|---:|---:|
| Streamed | 7.001 | — | — |
| Mstep, default tile/whole column | 8.816 | 67.42% | 216 |

Mstep improves throughput by 25.9% on this current matched recipe. All thirty
returned state leaves compare equal to the saved current streamed reference.

This is **not** a strict old-source/new-source regression verdict. The archived
October environment explicitly records a **370 W** limit, with observed draws
near 368–369 W and SM clocks around 1750–1830 MHz. The current limit is 250 W.
Also, only nineteen of twenty-five coefficient fingerprints match October:
the E coefficient packing changed from small codebooks/linear packed indices
to adaptive value pools/block headers. Source waveforms match, but source slab
placement/extents and spatial profiles changed. Initial zero-state fingerprints
match. Preserve those differences instead of blaming absolute rate changes on
the temporal scheduler alone.

## Large current controls and launch settings

These use the same realistic CPML12 recipe, 256 steps, three warmups plus one
warmup second and five synchronized executable samples per fresh process.
Every launch setting passes full-state comparison, with equal returned leaves.

| Shape (z,y,x) | Streamed GCUPS | Mstep default | Mstep segment 32 | Mstep segment 64 |
|---|---:|---:|---:|---:|
| 256×256×256 | 7.372 | 7.957 | 8.416 | Not measured |
| 256×512×512 | 7.463 | 10.554 | 10.647 | 10.749 |

The large calibration shape was confirmed from archived `main-anchors` as
256×512×512, rather than an axis permutation. Default mstep improves throughput
by 41.4% over streamed on that shape. The segment-64 increment over default is
only 1.8% in this first screen; confirm in balanced fresh-process order before
promoting it. On the smaller cube, other tile/segment results were 7.926 GCUPS
for 8×16/32, 8.231 for 8×32/64, and 8.138 for 16×16/32. Those are observations,
not a universal tile policy or confidence interval.

## Corrected nanoparticle resolution sweep

680 nm cube including CPML12, 100 nm radius n=2 sphere in vacuum, 500 nm
finite-aperture illumination, six six-component DFT box faces, 30 fs and CFL
fraction 0.95. Physical PML thickness changes with resolution. Three timed
executable samples, two warmups and one public sample per fresh process.

| Resolution | Grid | Streamed seconds | Mstep seconds | Streamed GCUPS | Mstep GCUPS | Eligible core |
|---:|---|---:|---:|---:|---:|---:|
| 5 nm | 136³ | 1.6045 | 1.9607 | 5.142 | 4.208 | 35.27% |
| 4 nm | 170³ | 3.7140 | 4.0111 | 5.424 | 5.022 | 43.56% |
| 2.5 nm | 272³ | 21.7701 | 19.3962 | 6.064 | 6.806 | 64.73% |

All cases have finite, nonzero signals. All three mstep state comparisons report
zero measured maximum leaf-relative error versus streamed, including CPML, DFT
and clocks. The larger corrected case shows the expected crossover: mstep is
12.2% higher throughput at 272³, while forced mstep still loses on the two
smaller scenes. This fixture remains a nanoparticle throughput workload, not
an analytically validated Mie scattering cross section.

## Separate extra warmed profiler calls

These sums count instrumented GPU kernel event durations, excluding host gaps
and memcpy events. They are not unprofiled wall-time shares.

| Workload | Column core | Edge/PML/scratch | DFT | Source injection |
|---|---:|---:|---:|---:|
| Nanoparticle 136³, CPML12 | 34.46% | 57.39% | 6.33% | 1.67% |
| Realistic 256×512×512, CPML12 | 50.28% | 48.95% | 0.22% | Not separately identified |

The large core has 52 registers, 61,392 dynamic shared bytes, 720 threads and
a 17×33×1 block grid in profiler metadata. The theoretical occupancy estimate
is 46.875%, not a measurement of achieved occupancy. Even this large winning
fixture spends substantial profiled time in edge phases, so whole-program
optimization must address both paths.

## Artifact index

All raw artifacts reside in
`.cache/cuda-autoresearch-2026-11-11/baseline-run/`, with a compatibility symlink
from `/tmp/beamz-cuda-autoresearch-20261111-1021`:

- `baseline/provenance.json`, `source.tar.gz`, `beamz_cuda.so` freeze the initial
  source and installed component.
- `baseline/historical-{streamed,mstep}.json`, `cube256-streamed.json`,
  `large-streamed.json`, and associated full-state reference arrays.
- `variants-{cube256,large}/matrix.json` and per-setting reports/logs.
- `baseline/mie-cpml12.json` records all six fixed-duration cases.
- `profile-{mie,large}-cpml12/summary.json` and Perfetto exports.
- `baseline/historical-power-evidence.json` extracts the historical 370 W
  environment and sample telemetry from the immutable October archive.

The reusable fresh-process runner is `scripts/run_cuda_variant_matrix.py`;
profiling is `scripts/profile_cuda_workload.py`. Both guard the expected native
binary hash. Native source A/B trials use `scripts/run_cuda_build_matrix.py`,
which verifies build manifests, gates hardware/full-state checks and restores
the pre-trial installed component even on rejection.

## Fixed-control-v2 checkpoint

The fixed control was installed atomically and passes all twelve new material
layout/cache-order/traced-dependency cases plus the thirty-six selected existing
hardware gates. Its native SHA is
`792d511f87d4d5255d812d89e860faae9fc0e8a1742e6a784f6d9e044cd614fe`.
It remains installed as the restore baseline; the wrong old component is retained
only as an archived negative control.

Seven-sample fresh trials, four warmups plus two warmup seconds, measured
8.4256 GCUPS on the historical 160×320×384 fixture and 10.4374 on the large
256×512×512 fixture. Both compare equal on all thirty leaves to the existing
streamed full-state references. These trials confirm the saved references remain
valid for these physical fixtures. Their small rate changes versus the earlier
component are not a balanced performance comparison.

The preparation-order explanation is visible directly in current artifacts:
historical streamed-first preparation uses packed value pools of length 836 and
rank-two constant-block headers. Pool length describes serialized storage, not
the number of distinct materials. Fresh mstep-first preparation uses codebooks
of three/four values and legacy rank-one byte-code buffers. That is why a
fresh-process mstep benchmark could pass while a same-simulation cache-order
test failed. Future candidate gates include both preparation orders, forced
constant-block layouts, BF16 recurrence and bare-field controls.

Checkpoint artifacts are
`.cache/cuda-autoresearch-2026-11-11/fixed-control-v2-{regressions,hardware-gates}.log`
and `fixed-control-v2-baseline/{historical,large}-mstep.json`.

## First corrected native-build screen

Five frozen source snapshots were tested serially at the unchanged 250 W limit.
Each passed fifty selected hardware tests (including the new material-layout and
cache-order regressions, traced dependencies, BF16 recurrence and bare fields).
Each also matched all thirty leaves of the saved streamed references exactly on
all three realistic fixtures. Timings use seven synchronized samples after four
warmups plus two warmup seconds, in fresh processes.

| Build | 160×320×384 GCUPS | 256³ GCUPS | 256×512×512 GCUPS |
|---|---:|---:|---:|
| Fixed control | 8.4085 | 7.6264 | 10.4179 |
| Exact indexing | 8.5603 | 7.7226 | 10.5362 |
| Edge specialization | 8.1963 | 7.1980 | 10.0602 |
| Earlier prefetch wait | 8.5237 | 7.7272 | 10.1354 |
| Generated traced cell arithmetic | 8.4398 | 7.6362 | 10.4638 |

Only exact indexing is positive on every measured case, by roughly 1–2% in this
screen. That is too small to promote without balanced fresh-process confirmation.
Edge specialization loses across these cases; earlier waiting is mixed. Generated
traced arithmetic preserves parity with little measured overhead, but this does
not establish a useful speedup. Do not sum improvements from separate candidates
or turn process-internal samples into a claim about between-process confidence.

The full build manifests, hardware logs, commands, power/clock snapshots,
samples and full-state verdicts are in
`.cache/cuda-autoresearch-2026-11-11/native-ab-v2/matrix.json` and its per-build
directories. The runner restored the correct fixed-control component afterward.

## Layout/register/axis/barrier screen and broader rotation gate

The next same-method default-launch screen again passes fifty selected hardware
gates and exact physical references for every build:

| Build | 160×320×384 GCUPS | 256³ GCUPS | 256×512×512 GCUPS |
|---|---:|---:|---:|
| Fixed control repeat | 8.3850 | 7.6045 | 10.3912 |
| Compile-time material layout | 8.7639 | 8.0211 | 10.5823 |
| Register histories, default tile | 7.2787 | 7.0561 | 9.1670 |
| CPML axis masks | 7.6483 | 6.6771 | 9.6224 |
| Final-barrier removal | 8.4524 | 7.6456 | 10.4569 |

Layout specialization is the only substantial uniform positive in this screen;
balanced final-source comparisons remain required. Default register histories
lose, despite standalone generated periodic-kernel wins. Static resource data
shows 106 registers per thread for default production register histories versus
56 in the fixed control, with no spills. The register kernel uses 256 threads
and three per-thread XY slots on the default tile; shared savings alone do not
guarantee increased residency.

An attempted register 8×16 follow-up deliberately ran the entire 141-test GPU
suite first. It exposed four binary-material cyclic-storage failures and a
cache/order-sensitive shared-material test failure. Repeating the full suite
on fixed control reproduced the four cyclic failures (137 pass), establishing
a second baseline defect rather than blaming the register change. The storage
wrapper only repacked rank-one byte IDs and left rank-two spatial headers
unrotated. The Python-side fix decodes actual coefficients, rotates them and
uses a unit decay; it conservatively admits the additional dense workspace.
After that fix, the expanded 155-test baseline suite passes. Register 8×16
performance remains unmeasured at this checkpoint; no default-only rejection
is presented as an all-tile verdict.

## Full-field edge publication experiment

`control-v5` and `publication-v5` have identical frozen native source SHA
`027bfb8bff2197f0c3990b6ae8af8463a81db051b499a3ecbf55fd678d5dfe24`.
Only the compile-time publication option differs. Full suites pass (155 control,
157 publication after two additional tests were added), as do exact physical
references. These are scientifically useful negative performance results:

| Build | 160×320×384 GCUPS | 256³ GCUPS | 256×512×512 GCUPS |
|---|---:|---:|---:|
| Matched control-v5 | 8.4296 | 7.6198 | 10.4083 |
| Core publication plus full-field edges | 8.1949 | 7.2133 | 9.8788 |

Extra warmed large-case profiles explain the regression. Numbers below are
summed instrumented GPU event seconds, not unprofiled wall shares:

| Kernel group | Control | Publication |
|---|---:|---:|
| Two-step core | 0.848484 | 0.980704 |
| First-step edges / copies | 0.398448 | 0.412647 |
| Second-step edges | 0.382348 | 0.339944 |

The direct second-step addresses help, but whole-core publication costs more.
Classification explicitly recognizes the new full-edge kernel templates and
their CopyCore argument. Compact first/second assignment is used only after
verifying the even-step H/E/H/E event sequence and count. Accounting tests
reject CPU-only traces and prevent new edge kernels being hidden as “other.”

CPU-only plan reconstruction matches recorded benchmark summaries exactly.
For the large fixture, only 918,922 of 15,909,003 first-step cells are owned-core
values requested for publication. Allocated 8×16 quadrant slots cause 5,998,848
owned-core point publications: 6.53 times the useful count. About 24% of first
queue warps mix core-copy and edge-update work. The historical fixture and cube
publish 6.08 and 8.73 times the useful points, with 20.0% and 31.4% mixed warps.
This motivates testing
direct full-field edge addresses without whole-core publication, rather than
deepening the same expensive publisher.

Artifacts: `native-ab-v3v4/`, `native-{control,register}-8x16/`,
`fixed-control-v2-full-hardware.log`, `fixed-control-v2-rotation-full-hardware.log`,
`native-ab-v5/`, `profile-{control-v5,publication-v5}/`, and
`publication-plan-accounting.json` under the same local research cache.


## Full-intermediate-bank architecture and combined screen

The publication experiment above was rejected. Its measurements identified a
more useful architecture: store complete first-step intermediate fields outside
the column core, while the column kernel keeps its existing recurrence. This
removes compact-neighbor address decoding from edge phases without adding core
publication stores. Control-v6 and full-intermediates-v6 are built from the same
immutable source snapshot, differing only in the experiment macro.

Both pass all 157 hardware cases present at that checkpoint, and every realistic
fixture matches all thirty saved streamed state leaves exactly. Seven samples,
four warmups plus two warmup seconds, default tile/whole-column settings:

| Build | 160×320×384 GCUPS | 256³ GCUPS | 256×512×512 GCUPS |
|---|---:|---:|---:|
| Control-v6 | 8.4317 | 7.6077 | 10.3998 |
| Full intermediates v6 | 8.8833 | 7.9832 | 10.8431 |
| Full + layout/exact-index/barrier v7 | 9.4357 | 8.5012 | 10.9103 |

The v6 screen improves throughput by 5.4%, 4.9% and 4.3%. The combined v7
screen improves it by 11.9%, 11.7% and 4.9% against this default control. These
are initial serial screens, not final balanced or equally tuned claims. The
combined build also passes 157 hardware gates and exact full-state checks. The
new mixed-F32/BF16 CPML-history regression increases the final hardware suite to
158 cases; later candidates and final acceptance include it.

Artifacts: `.cache/cuda-autoresearch-2026-11-11/native-ab-v6/` and
`native-ab-v7-combined/`. The v6 full-bank SHA is
`8348f41e28354e0707b1aa6ff561c5ea6327a3df3e894203b4a8ded98d71608f`;
the combined-v7 SHA is
`fa12308e4d44f5c4a42f312f6c94c1b92cf05b225f2964c0cb6a1eeb7db00c8d`.

The retained worker now records material-preparation workflow and actual
coefficient layout. Final acceptance must cover both mstep-first byte-code
preparation and streamed-first block-header preparation. Preparation is outside
warm timing and included in first-result/setup timing. A minimal system-NVML
fallback supplies a 50 ms sampled **device-global** memory peak over cold/warm
execution before validation. It includes other GPU processes and driver
reservations and can miss shorter allocation peaks; it is not allocator-owned
peak memory. No package installation or GPU power change was made.

The register-liveness v8 experiment passes all 158 hardware gates but is rejected
on measured throughput. At matched 8×16 tiles and 32-cell segments on the large
fixture, control-v6 reaches 9.896 GCUPS, full-intermediates-v6 reaches 10.365,
and v8 reaches 10.192; every result matches all thirty reference leaves exactly.
The compiler's reduced register count and zero reported spills therefore do not
establish an execution-speed improvement. No register-combined v9 performance
claim is made. Raw artifacts are `native-ab-v8-smalltile/` under the same cache
root. The restored component SHA remains the verified fixed-control-v2 binary.


## Typed full-edge and generated-cell final candidate

The isolated full-edge CPML-history/PEC type specialization is positive after
combining the accepted full-bank, layout, exact-index and barrier changes. It
keeps the rejected interior bypass and axis-switch experiments disabled. Both
typed-v10 and its traced-arithmetic sibling v11 pass all 158 hardware cases and
match all thirty realistic-fixture reference leaves exactly.

| Build | 160×320×384 GCUPS | 256³ GCUPS | 256×512×512 GCUPS |
|---|---:|---:|---:|
| Typed full-edge v10 | 9.6939 | 8.7490 | 11.1577 |
| Typed full-edge + generated arithmetic v11 | 9.6564 | 8.7647 | 11.1469 |

These seven-sample fresh-process screens use the same defaults as the earlier
combined-v7 screen. The extra typed-edge change improves v7 by about 2–3% on all
three. Generated cell arithmetic changes v10 by −0.39%, +0.18% and −0.10%; a
single process per case does not establish those tiny differences as a
regression or improvement. V11 is selected for bounded sanitizer checks and
balanced acceptance because it retains generated arithmetic without a material
observed loss. Its native SHA is
`4cdde58fc618dad17b0c7f04ec6050e8ec356eacdf51d153c4b98ba714500d1a`.

A new all-macros-off `final-control-v11` is built from exactly the same frozen
source SHA as both typed variants:
`9dbc587cb4b7757f33645afc28c7b89c22ed1f0b18cd30909adc0994fa41276e`.
Final tuning is bounded to 16×32 tiles and whole/32/64-cell segments for both
builds. Balanced ABBA will use each build's best observed segment for each
fixture and cover both material preparation orders. Process medians are the
repeat units; the seven samples within one process do not constitute seven
independent process trials. Only two processes per build/order/fixture are
planned, so no confidence interval will be inferred from those repeats.

Raw screens: `native-ab-v10-typed/` and `native-ab-v11-traced/` in the cache root.

The selected v11 binary also passes the new tiny seeded CPML12 gate under
Compute Sanitizer memcheck with zero reported device errors. That test asserts
that the column core actually runs, rather than accepting an edge-only case.
It is a separate 159th hardware case added after the complete 158-case run.
Expected XLA host symbol-probe API reports are disabled for this device-access
check; the initial unfiltered log is retained. Racecheck and synccheck remain
**timeout-incomplete**, including narrowed one-core runs. The matched final
control also times out in the bounded one-core racecheck. Neither absence of a
diagnostic before timeout nor that control timeout is a sanitizer pass or proof
of global-memory race freedom. Detailed commands, limits and logs are in the
native review and `full-typed-traced-v11-*-sanitizers/` cache directories.

## Balanced, equally tuned final native acceptance

Control and selected candidate use the same frozen source SHA. Both received
the same bounded 16×32-tile whole/32/64-segment tuning screen. Historical uses
segment32 for both; cube256 uses segment64 for both; large uses whole for the
control and segment64 for the candidate. These settings were selected on the
mstep-first workflow and held constant across preparation orders. This is a
bounded setting search, not a universal autotuning optimum.

Each row below uses fresh-process ABBA order: control, candidate, candidate,
control. Each process takes seven synchronized samples after four warmups plus
two warmup seconds. The table reports the median of the two process medians,
with no confidence interval inferred. All **24** processes match all thirty
saved streamed state leaves exactly, with one unchanged timed-worker SHA.

| Material preparation | Fixture | Control GCUPS | Candidate GCUPS | Throughput gain |
|---|---|---:|---:|---:|
| Mstep first, byte IDs | 160×320×384 | 8.5130 | 9.7490 | 14.52% |
| Mstep first, byte IDs | 256³ | 8.2662 | 9.5599 | 15.65% |
| Mstep first, byte IDs | 256×512×512 | 10.4275 | 11.6004 | 11.25% |
| Streamed first, block headers | 160×320×384 | 8.7026 | 9.9682 | 14.54% |
| Streamed first, block headers | 256³ | 8.4931 | 9.7785 | 15.13% |
| Streamed first, block headers | 256×512×512 | 10.7726 | 11.9746 | 11.16% |

Actual metadata confirms shared prepared coefficient objects and distinct
rank-one byte-code versus rank-two block-header storage; preparation order is
not merely a label. The durable summary retains every process median, observed
process range, paired process ratios/range, coefficient shape/type metadata,
launch settings, source/native/worker hashes and raw artifact paths:
[`cuda-native-final-results.json`](../../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-native-final-results.json).
The exploratory setting screen is retained in
[`cuda-native-tuning-results.json`](../../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-native-tuning-results.json).

A subsequent direct harness using the frozen production-v11 core, without the
JAX/XLA graph path, checks 9,216 owned values bitwise for whole and segmented
execution while preserving guards and inputs. Its memcheck, racecheck and
synccheck all finish with zero reported errors/hazards. This supplies direct
column-core coverage; it does not convert the earlier whole-engine instrumented
timeouts into completed tests or establish global-memory race freedom. Exact
scope and logs remain in the native/tracing reviews.


## Final fixed-CPML12 nanoparticle acceptance

The final 680 nm box sweep uses the same frozen-source control and selected
traced candidate as ABBA, with 16×32 tiles and 64-cell segments for both, selected
by the bounded realistic cube screen. Each point is one fresh process with
three synchronized executable samples, two warmups and one public-call sample.
All twelve points have finite nonzero field/DFT signals. All eight mstep
comparisons report zero measured error across every returned state leaf versus
streamed. No 1 nm run or other FDTD engine is included.

| Resolution | Streamed solve s | Control mstep s | Candidate mstep s | Gain over control | Gain over streamed |
|---:|---:|---:|---:|---:|---:|
| 5 nm | 1.5880 | 1.8337 | 1.5608 | 17.49% | 1.74% |
| 4 nm | 3.7631 | 4.0719 | 3.4786 | 17.05% | 8.18% |
| 2.5 nm | 21.6765 | 19.4181 | 17.0311 | 14.02% | 27.28% |
| 2 nm | 51.0579 | 43.6121 | 38.5033 | 13.27% | 32.61% |

The 5 nm candidate/streamed difference is small enough to describe as near
parity in this single-process screen. The larger observed wins are consistent
with the final realistic-fixture ABBA results, but these nanoparticle points
do not supply independent process repeats or confidence intervals.

| Resolution | Streamed public s | Control public s | Candidate public s |
|---:|---:|---:|---:|
| 5 nm | 1.5850 | 1.8572 | 1.5857 |
| 4 nm | 3.7762 | 4.1252 | 3.5384 |
| 2.5 nm | 21.7567 | 19.5004 | 17.0463 |
| 2 nm | 51.1390 | 43.6147 | 38.5289 |

The public columns are one synchronized sample each, rather than repeated
process trials. Separate scene-to-first-result and setup/compilation metrics
remain in the JSON.

| Resolution | Streamed peak GiB | Control peak GiB | Candidate peak GiB |
|---:|---:|---:|---:|
| 5 nm | 1.667 | 1.667 | 1.667 |
| 4 nm | 2.151 | 2.151 | 2.151 |
| 2.5 nm | 5.187 | 5.187 | 5.187 |
| 2 nm | 9.194 | 9.194 | 9.194 |

Peaks are sampled device-global usage over cold/warm execution before schedule
validation; they include driver reservations and other GPU processes and can
miss brief spikes. GPU power stays at 250 W.

The chart-ready durable artifact is
[`cuda-mie-final-results.json`](../../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-mie-final-results.json). It preserves
solve/public samples, cold/setup timings, actual material representations,
signal/parity outcomes, allocation estimates, global-memory peaks, launch
settings, native/source/worker hashes and raw result paths.

## Separate final matched profiles

One extra warmed instrumented call on the large realistic fixture compares
control whole-column execution with the candidate's selected 64-cell segments.
Both profiles verify the expected H/E/H/E edge launch order: 128 column-core
launches and 256 launches for each edge stage. Full-edge specialization remains
classified as edge work rather than being hidden in the miscellaneous bucket.

| Summed GPU event durations | Control seconds | Candidate seconds |
|---|---:|---:|
| Column core | 0.846718 | 0.756559 |
| First-step edges | 0.397811 | 0.380466 |
| Second-step edges | 0.381771 | 0.326041 |
| All kernels | 1.638632 | 1.475221 |

This is a 10.0% reduction in summed instrumented kernel durations, consistent
with the approximately 11.2% unprofiled throughput gain. It compares the final
architecture plus each build's chosen launch setting; core savings cannot be
attributed solely to edge specialization. GPU-event sums exclude CPU gaps and
copies and are not unprofiled wall-time shares. Raw traces and provenance are
indexed by [`cuda-native-final-profiles.json`](../../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-native-final-profiles.json).

## Final local package installed

The all-architecture production-defaults-v13 build passes the complete **159**
hardware cases and exact all-thirty-leaf replays of the historical, cube256 and
large fixtures. Its source includes the winning defaults and an unused-variable
guard cleanup after the frozen performance candidate. The three-sample,
one-warmup replays are packaging validation, not additional performance claims.

The component is atomically installed and verified as version **0.30.0**, ABI
**30**, native SHA
`34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836`,
source SHA
`f355a62b3cab7f76a69ad955b9c08e24f04587af27bf493f5601d30e586182a9`.
It remains installed; the earlier fixed control is retained as a backup rather
than restored at the end. GPU work ended at 13:58 UTC with the unchanged 250 W
limit and no benchmark process remaining. The durable gate record is
[`cuda-production-package-validation.json`](../../../.cache/cuda-autoresearch-2026-11-11/reports/cuda-production-package-validation.json).
