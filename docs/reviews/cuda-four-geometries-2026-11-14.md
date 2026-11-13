# Shared CUDA candidate: four-geometry campaign

One-hour campaign, 2026-11-14, starting 14:09 UTC. All runs use the local RTX 3090 at a 250 W power limit, FP32 fields and exactly 12 CPML cells per face. No RunPod or ABBA runs. `origin/main` was pinned and rechecked as `387cc8fa8a48f3f1e69077bb50c299df21940916`.

## Results

| Case | Core main → candidate | Speedup | Cold total main → candidate | Host GB main → candidate | GPU GB main → candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| mie | 88.910 → 38.486 s | **2.310×** | 94.84 → 51.76 s | 4.945 → 5.373 | 8.978 → 4.909 |
| cosine | 68.018 → 36.213 s | **1.878×** | 85.98 → 51.76 s | 6.574 → 5.817 | 2.712 → 2.676 |
| ring | 45.426 → 28.292 s | **1.606×** | 56.64 → 39.73 s | 4.567 → 3.888 | 2.611 → 1.812 |
| mmi | 39.697 → 21.919 s | **1.811×** | 102.86 → 38.08 s | 7.096 → 4.442 | 2.816 → 2.336 |

Mie is 2.31× faster in the core, but its host peak is still 8.7% above main. Mie setup/decoding is 6.04 → 8.34 s. All three photonic examples have lower host and GPU peaks. Mie's candidate host peak falls from the starting shared candidate's 7.858 GB to 5.373 GB. These are material improvements, not full memory/setup parity.

Solver times are synchronized host-wall times from one warm repeat following a cold solve in each fresh process. These are complete fixed-horizon solves, including sources, CPML, raw observation publication, DFT accumulation, tails and clocks. Short screening runs are reported separately below and are not substituted for these results. Small percentage changes are directional measurements, not confidence intervals.

Cold wall time starts before Python/BeamZ imports and geometry construction and ends after result decoding, before validation export. Setup/decoding is public-call wall time minus cold engine time; graph construction performed by the engine remains in cold engine time. Candidate preparation currently includes experimental input serialization and fingerprints. Main uses its usual GPU preparation; the candidate uses CPU preparation and native host reset. In particular, MMI setup benefits from CPU source-mode preparation, independently of solver acceleration.

Memory is peak process RSS and NVML process GPU allocation during cold execution, before validation exports. Host values take the greater of sampled RSS and Linux maximum RSS at that boundary. Decimal GB are used. Validation arrays and their subsequent exports are excluded; all-run sampled peaks remain in `summary.json`. NVML includes process contexts and allocator overhead, rather than just solver payload.

The 2–3× core target is not achieved across all four geometries. The single shared candidate and preparation method are kept for every case; there are no geometry-name, cube-size or source-count dispatch rules.

## Declared workloads

| Example | Grid | Horizon | Declared variation |
| --- | --- | --- | --- |
| Mie | 384³, 680 nm cube | 9,260 steps, 30 fs | 100 nm radius, index 2 sphere; 500 nm Gaussian plane wave; all six enclosing DFT faces |
| Cosine crossing | 481×481×90, 30 nm | 17,484 steps, 1 ps | Current regular-grid comparison notebook; one source reconstruction frequency; through/cross ports and Ez field DFT |
| Ring | 344×385×96, 25 nm | 20,981 steps, 1 ps | Original 3 µm ring geometry; fixed horizon; 31 spectral samples; pulse offset 1; through/drop ports and Hz field DFT |
| 1×4 MMI | 674×267×84, 30 nm | 12,239 steps, 0.7 ps | Original slot-stack splitter geometry; five source reconstruction frequencies; 31 spectral samples; four flux outputs and Ez field DFT; output mode-projection monitors omitted |

Ring and MMI horizons are throughput/accuracy-against-main workloads, not convergence claims for ring Q or final device-design quality. Neither variant uses automatic termination. These are simulation benchmarks; post-run notebook plots are not timed. Cosine retains the notebook's pre-run construction/inspection code in both variants.

## Changes retained

1. **Common effect storage schedule.** Noncertified boundary and observation classes now use the existing paired-plane schedule alongside source classes. Their CUDA register allocation drops from 128 registers at 512 threads to 64 at 1,024 threads, with no additional global buffers. Certified bulk/NoIO/observer paths retain their established schedules.
2. **Observation effect facts.** Six component-presence bits are packed into each existing coarse observation lookup word. A component absent from that region skips the leaf lookup. Raw observation IDs, numerical values, DFT order and GPU allocation sizes are unchanged.
3. **CPML-axis lowering.** The general boundary/effect class specializes the already-proved single-axis trajectory flags inside one kernel family. This removes irrelevant CPML effects without fragmenting launch cohorts or changing physical CPML profiles.
4. **Descriptor-only preparation.** Fresh field, CPML and unused workspace zeros remain descriptors until native upload. The packet uses canonical ABI layout helpers and matches all 134 earlier inputs exactly. It does not instantiate a Python graph object per physical cell.
5. **Exact zero snapshots.** Captured positive-zero initial state and deferred scratch can be reconstructed with device clears. Nonzero data and signed zero retain the copy path. Writable host scratch backing is created only when actual ordinary execution retires used scratch. Caller-mutation/reset checks cover both positive and negative zero.
6. **Material setup allocation.** Block packing retains a view when the coefficient sample count already fits complete blocks, instead of calling `np.pad` with zero padding and copying the entire grid. Mie avoids approximately 680 MB of such potential copies. Block packing also runs before byte-ID packing when it beats a conservative lower bound, avoiding disposable byte-ID buffers. Encoded coefficients remain bit-identical. Coefficient generation still sets the remaining Mie host peak.

The final wrapper reuses source- and compile-option-verified identical CUDA/planner objects from the preceding axis build. Its object reuse is recorded in `proof.json`. It changes ownership/preparation, not core CUDA arithmetic. The builder now supports this checked wrapper-only iteration explicitly.

## Accuracy and validation

| Case | Worst observable relative L2 | Worst flux peak-scaled error | All 134 inputs / all 30 states vs previous candidate |
| --- | ---: | ---: | --- |
| mie | 2.29e-06 | 1.44e-07 | Identical / bitwise identical |
| cosine | 1.62e-05 | 2.76e-07 | Identical / bitwise identical |
| ring | 1e-06 | 7.52e-08 | Identical / bitwise identical |
| mmi | 1.14e-06 | 1.64e-07 | Identical / bitwise identical |

Selected check groups: **52 passed, 1 skipped** (source/ABI/packing), plus **153 native numerical checks**. The selected seeded 16-step sanitizer macro passes all 30 state comparisons and reports **0 memcheck errors**, with instrumentation restricted to `TemporalEffects` and capped at 12 matching launches. This is a scoped positive check, not full-scene sanitizer coverage.

All 30 exported state leaves and every observable are bitwise identical to the previous shared candidate for each workload, and all 134 prepared-input fingerprints match. Thus the optimization and preparation changes themselves introduce no numerical deviation from that candidate.

Against main's usual GPU preparation, observables pass relative-L2 <5×10⁻⁵ and port-flux peak-scaled error <10⁻⁶. The detailed errors, including individual field and CPML arrays, are preserved in each comparison JSON. Full final fields are **not** bitwise identical to main's outputs under its usual GPU preparation: for the pulsed Mie and cosine cases, near-decayed residuals can show large relative errors. These are not represented as a successful final-state relative-error gate. A separate complete 9,260-step Mie ordinary-CUDA run on identical prepared inputs matches all 21 evolving field/CPML/DFT arrays bit for bit. Ring/MMI matched-input prefix oracles also match all 21 arrays bit for bit.

The selected native module passes 153 numerical/continuation checks across 17 source/material/shape cases, plus explicit topology rejection controls. CPU checks cover mask encoding, zero facts, ABI/fixture handling, material round trips and architecture guards. A bounded, filtered memcheck instruments the shared effect kernels; its exact scope and result are in the evidence manifest. Broader multi-call memcheck attempts were stopped to preserve the one-hour budget and are not counted as completed sanitizer passes.

Compiler resource auditing was expanded to recognize the new effect kernels. Both the accepted starting binary and the new CUDA objects have 56 update-kernel entries (excluding the epoch utility), no `LOCAL` allocation in cuobjdump, and existing small stack/spill frames up to 24 bytes. They are **not spill-free**. Generic boundary/observer entries have 64 registers and zero stack; the remaining older source/bulk/NoIO spills remain a concrete allocation target.

## Experiments and tracing interpretation

| 1,024-step screen | Mie s | Cosine s | Ring s | MMI s |
| --- | ---: | ---: | ---: | ---: |
| Previous shared candidate | 4.2721 | 2.1669 | 1.4040 | 1.8950 |
| Paired generic effects | 4.2734 | — | 1.3595 | 1.8485 |
| Effects + profile staging | 4.3041 | — | 1.3639 | 1.8381 |
| Effects + observation masks | 4.2808 | — | 1.3548 | 1.8422 |
| Selected CUDA axis/effect lowering | 4.2691 | 2.0873 | 1.3566 | 1.7885 |
| Previous candidate, depth 4 | — | — | 1.9342 | 2.1120 |
| Paired effects, depth 4 | — | — | 1.7890 | 2.0977 |

Profile staging is not retained: its differences are small and inconsistent. Depth 4 is slower than depth 16 on ring and MMI. The selected CUDA lowering yields about 3.5%/5.9%/3.8% prefix throughput gains on ring/MMI/cosine respectively; Mie is a tie. Full-solve confirmations above use the final shared wrapper and frontend.

The source-free observation and CPML/boundary kernels account for substantial summed instrumented duration in the ring profile, whereas DFT accumulation is a small fraction. These are overlapping kernel-duration shares, not critical-path wall-time fractions. The data supports targeting effect lowering and storage schedules; it does not establish one hardware stall as the unique bottleneck.

Mahlau's quoted post describes a custom PyTorch tracer recording cell operations, then compiling the graph into time-blocked CUDA. That is an architectural description; I could not verify that rewrite in the public main/development code. The [public FDTDX README](https://github.com/ymahlau/fdtdx) still describes JAX. We should not claim to have reproduced his unpublished tracer or its benchmark.

BeamZ already traces scalar Yee/CPML arithmetic with explicit rounding and lowers it into moving CUDA tiles. This campaign adds actual captured effect facts for observations, CPML axes and zero-state ownership. The important remaining difference is that CTA register/shared-memory allocation and much phase scheduling are still manually selected; arbitrary Python/PyTorch operations do not automatically become an optimized BeamZ schedule.

The [systolic FDTD paper](https://arxiv.org/abs/2502.20610) uses moving subdomains with one-way face trails to keep interior updates local. BeamZ shares that dependency principle. The paper's reported H100 benchmark uses 16-bit fields, no sources/output fields and adiabatic absorption rather than CPML. Its headline throughput is therefore not an equivalent FP32/CPML12 comparison. [AN5D](https://arxiv.org/abs/2001.01473) also supports treating temporal depth and allocation as compiler/model choices, rather than assuming that deeper blocking always wins.

The remaining Mie setup-memory target is to compact each coefficient component before computing the next one, or prove constant decay before allocating a full grid. The early block attempt also needs fallback-cost measurements on uncompressible materials before general rollout.

The strongest next architectural step is to make CPML, source injection and observation publication first-class effect nodes in the trace, then generate alternative storage/phase schedules with explicit liveness and traffic costs. Search should be keyed by semantic facts and GPU resource limits, with thin and cubic domains in every acceptance gate. Removing remaining spills and reducing shared-memory/barrier work deserve controlled experiments; another 2× from tracing alone is not established by this campaign. The core already executes native CUDA, so removing the frontend's JAX dependency by itself does not supply that speedup.

## Reproduction and limits

Selected module: `/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-four-geometries-2026-11-14-1409/masked-lazy-zero/_cuda_native.cpython-311-x86_64-linux-gnu.so`

SHA256: `511dfe2a7651dae7bc23249e8a8b0b98e7885cca88e0e0393950a8c0a48d78dc`

Frozen frontend: `/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-four-geometries-2026-11-14-1409/frontend-block-first`. This is pinned main plus the earlier source-ROI fixes and this campaign's material-packing changes. Both main/candidate frontend CUDA modules are `1df694d92e254720e1868fcb6a562bb331be50c3245b53ca93c370b2555f9e72`; the installed workspace default remains `34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836`.

Evidence: `/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-four-geometries-2026-11-14-1409`. Inspect `summary.json`, `evidence.json`, `report.html`, the four `*-selected-comparison.json` files, each `*-accepted/results.json`, `selected-validation.json`, `mie-full-oracle-final/result.json`, `lazy-zero-reset-mutation.json`, `memcheck-macro.log`, the build proofs and resource reports.

Reproduce with the benchmark driver for each case, and aggregate with:

```bash
/home/quentinwach/Code/.venv-cuda-pr233/bin/python scripts/summarize_cuda_four_geometries.py \
  --area .cache/cuda-four-geometries-2026-11-14-1409 --candidate-suffix accepted \
  --main-commit 387cc8fa8a48f3f1e69077bb50c299df21940916
```

The installed default CUDA binary was not replaced. This remains an experimental native candidate whose builder depends on an earlier frozen ABI artifact; it is not yet a standalone production backend. Scope remains Cartesian FP32, scalar H, lossless packed diagonal E, CPML12, canonical supports and supported coincident post-H/post-E sources. Full tensors, dispersive updates, PEC and unrelated/noncoincident source groups require further compiler work rather than silent omission.
