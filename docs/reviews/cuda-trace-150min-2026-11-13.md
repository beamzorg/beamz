# BeamZ temporal CUDA: 2.5-hour tracing campaign, 13 November 2026

The final trace-guided composition improves the current temporal CUDA control on both complete Mie workloads. 512³: 1.031976× (3.20% more throughput), 564³: 1.032551× (3.26% more throughput). Another 2× was not achieved.

The implementation remains an opt-in frozen native research module. The installed extension and production defaults are unchanged. No RunPod resources were used. Main/default CUDA and FDTDX were not rerun in this campaign.

## Complete simulation results

| Workload | Frozen current temporal control | New composition | Paired throughput | Pair range |
| --- | ---: | ---: | ---: | ---: |
| 512³, 12,347 steps | 108.277082 s | 104.922096 s | 1.031976× | 1.030727–1.033225× |
| 564³, 13,601 steps | 152.588353 s | 147.778050 s | 1.032551× | 1.032157–1.032945× |

Each complete case uses four fresh processes in A–B–B–A order: two opposite-order pairs, one full first call and one timed full repeat per process. The displayed times are medians across the two corresponding processes; speedup is the median of paired ratios. Observed ranges are not confidence intervals. All 30 output leaves must match the independent reference bytewise on both first and repeat calls; caller inputs must remain unchanged.

The timed metric is prepared execution, including preserving device copies, fields, CPML history, source additions, chronological DFT, residual ordinary steps and clock advance. Session construction, upload and readback are outside that metric. The full first call is checked before its timed repeat.

Both domains contain the same 680 nm cube, a 100 nm radius index 2 sphere, a 500 nm source, six-face DFT monitoring and nominal 30 fs propagation. They retain 12 CPML cells per face at both resolutions. Fields and state use FP32. Local hardware is one RTX3090 at the unchanged 250 W limit, driver 610.43.03 and CUDA 13.3. Native workers require JAX absent.

## Selected changes

The publication effect pass now explicitly checks physical-allocation identities, disjoint trail-component spans, dependency order and publication regions. It rejects ordered source/DFT effects, possible overlap and cross-barrier movement. Its generated CUDA is byte-identical to the frozen grouped-publication parent; it changes no running benchmark binary.

1. **Group independent publication effects.** Six trail stores share one surface guard; final component stores share one final-version guard. Private field values, addresses, source/DFT ordering and every publication/reader-retirement barrier remain unchanged. Native field components and the trail arena are separate allocations.
2. **Lower the phase clock.** A bit-preserving inline integer move of the induction variable replaces redundant volatile shared counter reads in the full-register bodies. It bounds compiler rematerialization without changing phase numbering.
3. **Specialize source CPML facts.** CTA-uniform axis masks select the same phase pipeline with inactive CPML terms eliminated. Policy 5 stages all 36 independent coefficient profiles, 24 FP32 words each, using an existing startup barrier. Source shared storage rises from 58,680 to 62,136 bytes, within 64 KiB. H/E, term and coefficient-kind profiles remain independent.

The composition passes the strict zero-stack/local/spill check for all 66 moving kernels. It retains the existing 8×16×32 cache, 7×15×31 owned volume, 1024-thread common paths and 16-step macro schedule. Depth 4 and fallback kernels remain available. A 16-step macro does not mean each physical cell stays in one CTA for 16 updates: the shortest owned spatial extent is 7.

The passes are trace-guided CUDA lowerings around the canonical Yee cell/effect graph. They are not a newly completed general-purpose PyTorch tracer. The native math, source and monitor semantics are preserved; no JAX execution is involved.

## Instruction evidence

Unprivileged CUPTI software SASS collection was enabled with process-local CUDA library paths. All sampled first/repeat outputs are bytewise correct and collections have zero drops. Diagnostic device instructions and encodings match the selected performance kernels; timings from these patched/direct launches are never benchmark results. [CUPTI SASS Metrics API](https://docs.nvidia.com/cupti/api/group__CUPTI__SASS__METRICS__API.html)

| Sampled control launch | CTAs | Integer/address/predicate/move | Control/synchronization | Floating point | Shared memory | Global memory | Shared actual/ideal | Global requested/ideal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| noio | 178 | 58.20% | 15.27% | 13.90% | 9.00% | 3.61% | 1.000 | 1.142 |
| bulk | 368 | 46.09% | 17.89% | 20.79% | 11.53% | 3.53% | 1.000 | 1.250 |
| source | 4 | 63.55% | 17.55% | 8.07% | 5.19% | 5.40% | 1.000 | 1.121 |

These are executed warp-instruction shares from one matching launch per family, not time/stall shares or a weighted workload breakdown. Global sectors are requests, not DRAM bytes. The source sample has only four CTAs. Shared transactions match the ideal count in these samples, so shared-bank conflict tuning is not supported by this evidence. Integer/control lowering is a stronger hypothesis, but a critical-path timing claim needs additional measurements. Categories follow opcodes; moves with constant operands remain in the move group. Constant-load instructions are below 0.25% and omitted from the table.

The matching final NoIO sample executes 93,917,659 warp instructions versus 100,600,945 in the control (6.64% fewer). This is a count comparison, not a kernel-time measurement.

## Screening and rejected paths

The artifact index currently contains 46 completed comparisons, of which 44 exercise the candidate rather than the explicitly excluded fusion fallback. Prefix runs execute 256 steps and use one fresh A/B pair unless otherwise recorded. They are screening evidence only; parent ratios are never multiplied to claim a composition speedup.

| Selected 256-step 512³ screen | Throughput/control | Interpretation |
| --- | ---: | --- |
| Phase clock | 1.008693× | Preliminary prefix screen |
| Named register banks | 0.977693× | Preliminary prefix screen |
| Shared fields/smaller tile | 0.727779× | Preliminary prefix screen |
| Shared fields with private CPML history | 0.713026× | Preliminary prefix screen |
| Two-launch boundary-free path | 0.924265× | Preliminary prefix screen |
| Warp X-neighbour exchange | 0.991141× | Preliminary prefix screen |
| Predicated trail stores | 0.977780× | Preliminary prefix screen |
| Grouped publication | 1.021678× | Preliminary prefix screen |
| Checked 32-bit trail offsets | 0.999091× | Preliminary prefix screen |
| Corrected field-cohort fusion | 0.999160× | Preliminary prefix screen |
| Corrected all-policy fusion | 0.930078× | Reject; actual fusion witnessed, slower |
| Heavy-first policy order | 0.979452× | Preliminary prefix screen |
| Captured source-node priority | 1.000320× | Preliminary prefix screen |
| Selected composition | 1.028617× | Select for complete-run confirmation |
| Terminal-store epilogue composition | 1.028391× | Not selected; tied composition while adding hot-path spills |

The first box-shape version changed observation masks and failed DFT checks; its timing was rejected and the generator repaired. The oversized/deep-unroll variants that exceed spill budgets were not assigned performance results. Fusion v2 retained the active parallel-policy fallback; both such timings are explicitly excluded from fusion claims. Fusion v3 was confirmed with CUPTI: 2,048 fused launches and 787,968 CTAs on the prefix, zero dropped activity records and all 30 outputs correct.

## Validation and sanitizer limits

The selected composition completes the seven-part coupled suite: seeded cubic/rectangular schedules, both depths, 35-step wrap, 259-step Mie work, nonzero clock origin, materials and probes. The 26 CPU checks cover temporal windows, owner maps, observation geometry independence, shared-history hazard negatives, profile identity, affine wraps, same-rank physical independence, SASS-data rejection and terminal-store/source-pipeline separation. The shared-box exploratory schedule separately passes 8,863,992 state/version checks and rejects nine deliberately broken schedules; this is not a whole-engine race proof.

**sanitizers:** memcheck inconclusive timeout.

**sanitizers-direct-one:** memcheck inconclusive timeout.

**sanitizers-exact-symbol:** memcheck inconclusive timeout, synccheck inconclusive timeout, racecheck inconclusive timeout.

The broader 64-launch sanitizer attempt timed out at 55 s without a result and is not a pass. The narrower direct-launch scope uses a diagnostic host wrapper whose 66 moving device kernels match the performance binary instruction-for-instruction. The narrower attempts were configured for one matching kernel on a seeded small 35-step coupled case, but also timed out. No positive memory, synchronization or race sanitizer result was obtained in this campaign. Numerical confirmation and sanitizer scope are reported separately.

## What the papers suggest next

AN5D combines stationary register allocation, limited shared-memory buffering and resource-aware temporal blocking. Its published recognizer expects a single store statement; multi-output support is future work in that paper. Applying its scheduling ideas to six staggered fields plus CPML/source/DFT effects requires an extended dependency/effect model. [AN5D paper](https://arxiv.org/pdf/2001.01473)

EBISU selects enough parallelism to saturate the device, then spends remaining resources on deeper reuse. Its circular queues change address interpretation instead of shifting all data, and lazy streaming reduces synchronization by buffering more work. This argues for measuring resource/ILP/reuse tradeoffs rather than maximizing occupancy or temporal depth in isolation. BeamZ’s mutable CPML and chronological observations add constraints beyond the paper’s example stencils. [EBISU paper](https://arxiv.org/pdf/2305.07390)

The next substantial tracer work should make affine address recurrence, predicate uniformity, rematerialization and shared/register lifetimes explicit. A separate prototype can test whether a different tile/thread shape plus partial field storage enables more useful residency or a longer per-cell lifetime. It must show a gain with complete physics and pass resource/ordering checks. This campaign does not establish another 2× route; most simple allocation, unroll, caching and launch-scheduling changes were flat or slower.

## Reproducible artifacts

- Control module SHA256: `1a91ce344d012a9a22b38e6ebc8c3d57248fb0bbdd6d7fc2683d9ac63a13828e`.
- Selected module SHA256: `becc5f154df24dbe53b68335d9f13a682fcbb3f11742f601cf04a5229745afe7`.
- Selected consumer SHA256: `cce32cfee8c4497d05c30585672935d03515bbf6fcf2bfc179c45865a2f3b7e1`.
- [Frozen native CUDA module](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/frozen-execution/phase-grouped-sourcefacts-v1/becc5f154df24dbe53b68335d9f13a682fcbb3f11742f601cf04a5229745afe7/_cuda_native.cpython-311-x86_64-linux-gnu.so).
- [Selection manifest](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/final-artifacts/selection.json), [matching instruction-count comparison](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/final-artifacts/instruction-delta.json).
- [Source patch](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/final-artifacts/trace-publication-source.patch).
- [All completed comparisons, CSV](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/comparisons.csv), [JSON](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/comparisons.json).
- [Frozen build/resource manifest](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/phase-grouped-sourcefacts-v1/manifest.json).
- [Complete composition source](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/phase-grouped-sourcefacts-v1/source/src/asymmetric_native.cuh).
- [All66 diagnostic device-identity checks](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/sass-metrics-probe/grouped-final-all 66-identity.json).
- [CPU test record](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/unit-tests-result.json), [effect-schedule proof](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/effect-schedule-proof.json).
- [Full-run process commands](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/full-confirmation-processes.json), [final cleanup/state](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/final-artifacts/final-state.json).
- [Full comparison plot, PNG](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/final-artifacts/full-comparison.png), [SVG](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-trace-150min-2026-11-13-1620/final-artifacts/full-comparison.svg).

Use the recorded commands and frozen workers in this workspace. Recompilation creates a new binary identity and requires fresh numerical and performance gates. No commits, pushes, GitHub mutations or production extension replacement were performed by this campaign.
