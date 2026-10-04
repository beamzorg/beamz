# RTX3090 CUDA streaming optimization — 2026-09-17

The [follow-up domain-shape study](cuda-shape-study-2026-09-18.md) fixes CPML at 12 cells and excludes lossy materials. The historical results below retain their original settings; the benchmark now defaults to 12 CPML cells.

The consistent **9 GCUPS** target for arbitrary realistic simulations is **not yet achieved**. This change improves eligible large CPML simulations, preserves a fallback for cases that do not benefit, and supplies a reproducible workload matrix. It does not establish H100 performance.

`origin/main` was merged locally at `b0a5ff32` (main: `14f8c47d`). Nothing was pushed. The pre-merge notebook edits were preserved separately; the notebook stash was retained.

## Measurements

| Case (z × y × x) | Steps | CPML cells | Baseline GCUPS | Candidate GCUPS | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small regression control (64 × 96 × 128) | 160 | 8 | [5.34](rtx3090-2026-09-17/baseline/small.json) | [5.49](rtx3090-2026-09-17/candidate/small.json) | +2.9% |
| Saturated fallback (128 × 256 × 256) | 256 | 8 | [7.69](rtx3090-2026-09-17/baseline/medium.json) | [7.70](rtx3090-2026-09-17/candidate/medium.json) | +0.2% |
| Large guide (128 × 256 × 512) | 256 | 8 | [7.84](rtx3090-2026-09-17/baseline/large.json) | [8.63](rtx3090-2026-09-17/candidate/large.json) | +10.1% |
| Wider guide (128 × 256 × 768) | 256 | 8 | [7.76](rtx3090-2026-09-17/baseline/xlarge.json) | [8.89](rtx3090-2026-09-17/candidate/xlarge.json) | +14.6% |
| Long run (128 × 256 × 512) | 1,024 | 8 | [7.87](rtx3090-2026-09-17/baseline/long.json) | [8.68](rtx3090-2026-09-17/candidate/long.json) | +10.3% |
| Thin guide domain (64 × 512 × 512) | 256 | 8 | [7.28](rtx3090-2026-09-17/baseline/thin.json) | [8.01](rtx3090-2026-09-17/candidate/thin.json) | +10.1% |
| Unequal dimensions (96 × 384 × 449) | 256 | 10 | [7.38](rtx3090-2026-09-17/baseline/odd.json) | [7.86](rtx3090-2026-09-17/candidate/odd.json) | +6.5% |
| Thicker CPML (128 × 256 × 512) | 256 | 16 | [7.56](rtx3090-2026-09-17/baseline/thick.json) | [7.95](rtx3090-2026-09-17/candidate/thick.json) | +5.1% |
| Dense E coefficients (128 × 256 × 512) | 256 | 8 | [7.06](rtx3090-2026-09-17/baseline/smooth.json) | [7.49](rtx3090-2026-09-17/candidate/smooth.json) | +6.1% |
| Compact mode monitors (128 × 256 × 512) | 256 | 8 | [8.31](rtx3090-2026-09-17/baseline/mode_monitors.json) | [9.22](rtx3090-2026-09-17/candidate/mode_monitors.json) | +10.9% |
| Nine-frequency field monitors (128 × 256 × 512) | 256 | 8 | [7.21](rtx3090-2026-09-17/baseline/nine_frequencies.json) | [8.23](rtx3090-2026-09-17/candidate/nine_frequencies.json) | +14.1% |
| Lossy / dense H coefficients¹ (128 × 256 × 512) | 256 | 8 | [5.40](rtx3090-2026-09-17/baseline/lossy.json) | [5.36](rtx3090-2026-09-17/candidate/lossy.json) | -0.6% |
| Homogeneous control¹ (128 × 256 × 512) | 256 | 8 | [7.76](rtx3090-2026-09-17/baseline/scalar.json) | [8.82](rtx3090-2026-09-17/candidate/scalar.json) | +13.7% |
| After 1,024 conditioning steps (128 × 256 × 512) | 256 | 8 | [7.85](rtx3090-2026-09-17/baseline/developed.json) | [8.64](rtx3090-2026-09-17/candidate/developed.json) | +10.0% |

¹ Gaussian-source controls; the other rows use solved mode sources. Links contain every timing sample and binary fingerprint. The medium row is the repeated comparison after fixing the source-launch regression.

Each GCUPS value counts material-grid cells × complete H/E timesteps, divided by the median synchronized executable duration. Setup, mode solving, compilation, graph warmup, and host result conversion are excluded. The benchmark reports setup and compilation separately. This is complete warm executable throughput, including sources, CPML and DFT accumulation; it is not timing a single update kernel. The same large case measured **7.66 GCUPS through the warm public `Simulation.advance` call**, versus **8.63 GCUPS** for the executable. That additional orchestration/result-handling cost must be addressed separately if the target is public-call throughput.

Unless specified otherwise: FP32 fields and CPML memories, 80 nm uniform spacing, a dielectric guide in a lower-index background, one solved TE mode source (eight compiled source specifications), two full clear-aperture DFT monitor planes recording Ey/Ez/Hy/Hz, three frequencies, eight CPML cells on every face, 256 timesteps. Candidate results use four warmups and seven samples in a fresh process per case; the baseline records contain 5–11 samples each. The long case uses 1,024 timesteps. The developed-field case advances the pulse for 1,024 steps before timing a further 256. Scalar and lossy controls use a Gaussian source; the current ModeSource API rejects conductive simulations. Mode-monitor planes are compact and therefore involve less gathering than the full field-monitor planes.

The 8.4–25.2 million-cell cases are large enough to saturate this GPU: sampled utilization reached 100%, and draw was approximately 368–370 W. The baseline plateau across increasing volume confirms that the remaining gap is not simply an undersized benchmark. The small case is retained only as a regression control.

All measurements here were made at **370 W**, because `nvidia-smi -pl 420` failed with insufficient permissions and passwordless sudo is unavailable. The board reports a supported maximum of 450 W. A request for the user to apply `sudo nvidia-smi -i 0 -pl 450` was left pending. No higher-power result is claimed. There is no guarantee that a higher limit alone will close the gap.

The machine also hosts the desktop/T3 and an existing notebook holding about 9.3 GiB of GPU memory. That notebook was left running. Benchmark processes ran sequentially, with JAX preallocation disabled and its allocation fraction limited to 0.45. This preserves application headroom but is not an exclusive-device laboratory measurement. Results should be treated as local medians, not confidence bounds or guarantees for every geometry. The small differences between experimental variants often fall within clock/power variability.

## What changed

- The CPML interior can now fuse H and E in a two-plane shared-memory ring. It reuses updated H values for E within each tile. The outer CPML shell remains separate. Interior halo H values are recomputed from the frozen input bank, including H-source contributions; shell halo values come from the completed shell update. This avoids cross-block reads of unfinished interior writes.
- Sources with the same timing are batched across components for small and fused runs. Larger unfused runs preserve the original source-launch sequence: indiscriminate batching reproduced a 3.5% regression there, and restoring that sequence recovered 7.70 versus 7.69 baseline GCUPS. Overlapping groups retain additive semantics. Coincident-source products and additions are explicitly rounded separately to match JAX's scatter-add ordering; this fixed a strict dense-material parity failure without changing test tolerances.
- Scalar and dense E coefficients with scalar H coefficients can use the existing combined CPML queue and two-bank schedule. Material-codebook eligibility no longer unnecessarily controls that spatial partition. Dense/lossy materials remain supported by the general fallback. Dense-H two-bank experiments measured 5.15 GCUPS with fusion and 5.38–5.54 without it, versus 5.40 baseline, while adding roughly 4 GiB of allocation in the large case. The gain did not reproduce reliably, so that scheduling extension was rejected and the original dense-H path retained.
- Rectilinear-grid metric validation now checks the transverse components that actually differentiate along each axis. The old check incorrectly required an extra H metric element from the longitudinal component's staggered extent.

Automatic interior fusion currently applies to compute capability 8.6, uniform symmetric CPML, an eligible two-bank schedule, at least 12 Mi cells, a contiguous x extent of at least 384, and scalar H update coefficients. These are conservative empirical dispatch limits, not a universal crossover model. Smaller/narrower grids keep the established combined phase queue. Other architectures keep that queue until measured. Graded grids and asymmetric absorbers retain their existing general paths.

`BEAMZ_CUDA_CPML_CORE_FUSION=0` disables this fusion; `=1` forces it for otherwise eligible schedules. The override is useful for parity tests and further profiling. The graph-cache key includes the selection, so switching it does not replay the wrong graph. Small correctness tests explicitly exercise both settings. No change to field precision or CPML thickness is used to obtain the improvements.

Rejected prototypes included dense-H interior fusion, full-domain H/E/CPML fusion, different tile dimensions, source-overlap checks per tile, and an alternate DFT point-gather kernel. Full-domain fusion was markedly slower; the others did not demonstrate consistent gains. They are not retained in the production path. In particular, the monitor rewrite was removed after a direct comparison showed essentially unchanged large-case throughput.

## Why throughput still varies

The original specialized CPML path combined core and shell work *within each phase*, but H and E were separate kernels. That loses interphase field reuse even when a small source or monitor is the only extra feature. A streaming traffic model gives approximately 72 field bytes per cell/step for separate phases versus an ideal 48 for full-step reuse, before halo redundancy, coefficients, recurrence memories, monitors, and cache effects. These are source-level traffic budgets, not measured DRAM counters.

CPML work depends strongly on shape. An approximate shell fraction is `1 - (1-2p/z)(1-2p/y)(1-2p/x)`. Making a domain thin or increasing absorber thickness raises recurrence work per counted cell even when the GPU is fully saturated. Dense material coefficients also require more traffic than compact exact codebooks. Twelve dense FP32 decay/source coefficient grids can add 48 bytes/cell/step, versus approximately 3 bytes for packed lossless E indices with scalar H coefficients. This is a substantial traffic-budget difference even before CPML and monitors; it is not a measured DRAM-byte count. DFT work grows with monitored points, components, frequencies and cadence; a GCUPS target cannot be independent of an unbounded observation workload.

The [retained fused-path Nsight Systems timeline](rtx3090-2026-09-17/profiles/profile-final-stats.csv) on 128 × 256 × 512 assigns 61.2% of native kernel time to the interior, 31.2% to H/E CPML shells, 7.3% to DFT work, and 0.3% to source injection. The denominator excludes setup kernels. The ring kernel uses 38 registers/thread on SM86, no compiler-reported local-memory spills, and 7,128 bytes of dynamic shared memory per block. Profiling durations are diagnostic and are not used for the GCUPS table.

The large-grid profiling result supports prioritizing interior field traffic and absorber execution. Source launch batching helps smaller cases, but source launches are too small a share of a saturated large run to supply the remaining gain. Hardware performance counters are restricted on this machine, so bandwidth saturation, L2 partition behavior and occupancy stalls have not been established by counter measurements. The narrower-grid fusion regression is observed; its precise microarchitectural cause remains unproven.

## Validation and reproduction

- **39 hardware tests passed**, with two Hopper-only tests skipped on this RTX3090. These compare complete field, CPML recurrence, source, DFT, clock and continuation state against JAX. Both fusion overrides are exercised; the binary-material fallback comparison includes a 1.18-million-cell grid.
- **44 targeted unit/contract tests passed** for scheduling, ABI, material encoding and benchmark contracts. Ruff and `git diff --check` pass.
- Compute Sanitizer **memcheck: zero device errors** on six JAX parity cases, filtered to the modified source/CPML kernels. API-return reporting was disabled for that run after the initial attempt reported XLA `cuModuleGetGlobal_v2` symbol-lookup errors. The original API-diagnostic log remains at `.cache/perf/memcheck-api-probes.log`; these checks do not certify XLA API handling.
- The [standalone sanitizer harness](https://github.com/beamzorg/beamz/blob/fa151d28/cuda/tests/cpml_core_sanitizer.cu), linked against the built component's update object, passes **memcheck with zero errors** and **racecheck with zero hazards, errors or warnings**, without suppressing API diagnostics. It exercises packed/dense coefficients, overlapping H sources, odd dimensions and partial z tiles. The full JAX racecheck attempt was stopped after more than five minutes of CPU processing without completing its first test; it is not reported as passing.
- Built with CUDA **13.3.73**, precise default math, for SM80/86/89/90; JAX **0.9.0**, NVIDIA driver **610.43.03**. CUDA 12 compatibility CI, H100 validation and Nsight Compute hardware-counter gates remain outstanding. This is a locally tested branch candidate, not a completed cross-device release qualification.

[Validation logs](rtx3090-2026-09-17/validation/) and [environment metadata](rtx3090-2026-09-17/environment.json) accompany the raw timings.

Run from the checkout with its freshly built CUDA extension and a CUDA-enabled JAX environment. Measurements used `/home/quentinwach/Code/.venv-cuda-pr233/bin/python` with this worktree on `PYTHONPATH`; this checkout's `.venv` has CPU-only JAX:

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
  python scripts/benchmark_cuda_realistic.py \
  --shape 128 256 512 --pml 8 --steps 256 --samples 7 --output result.json
```

The script defaults to a 16.8-million-cell case. Options cover shape, steps, CPML thickness, material type, source type, field/mode monitors, frequency count, and preconditioning steps. Optional `--public-samples` measures the warm `Simulation.advance` call separately. `--in-place` is an explicitly labeled diagnostic control for the old single-bank scheduling alternative.

JSON records include each duration, workload settings, compiled plan flags, coefficient shapes, extension SHA-256, checkout commit, GPU telemetry, and JAX version. New records also include source and harness fingerprints. The baseline extension was built from the merged revision, not copied from a different working checkout. Source fingerprints describe the measured checkout; the extension fingerprint identifies the executable code. Measurements were collected in stages. The final material/launch dispatch fixes were retested on their affected cases; measurements of unchanged fused paths retain their original binary fingerprints. Rejected prototypes are recorded separately.

## Remaining work toward consistent 9 GCUPS

1. Repeat this exact matrix after the user applies a higher supported power limit. Record sustained clocks, temperature and power alongside performance; retain T3/notebook memory headroom. Do not extrapolate achieved throughput from the power-limit ratio.
2. Profile memory transactions, cache hit rates and stalls with hardware counters when access is available. Use the measured large/narrow regression to guide layout or tiling work; avoid hard-coding more shape exceptions without evidence.
3. Reduce CPML shell cost and dense-coefficient traffic. In particular, investigate exact codebooks for **decay/source pairs in both H and E**, rather than only lossless E source coefficients; measure actual pair cardinalities before extending the ABI. Then extend profitable fusion to asymmetric and graded cases. Validate overlapping sources, odd tile tails, continuation, and complete DFT/recurrence state for each extension.
4. Profile public-API overhead separately from native executable time, then re-run the entire matrix, including long/developed fields and heavier monitors, after every retained change. Acceptance means the broad realistic envelope improves, with no hidden fallback regressions—not merely one case exceeding 9.

H100 optimization remains a separate hardware-validation task. The initial [source review](cuda-performance-2026-09-17.md) explains the 33-GCUPS traffic budget and current Hopper integration limits; this RTX3090 work does not verify that target.
