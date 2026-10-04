# RTX3090 storage padding, tile variants, and two-timestep blocking

Implemented and measured the three proposed experiments. **Consistent 9 GCUPS
is still not achieved.** Storage padding does not improve the tested cases;
the two-timestep prototype is correct within the parity tolerances but is
substantially slower. All new controls remain opt-in. The default retains
unpadded storage, one timestep per interior invocation, and the previous tile
selection.

This follows the [domain shape study](cuda-shape-study-2026-09-18.md), which
covered 26 domain shapes. The new experiments compare different storage and
kernel choices on identical physical simulations. They do not enlarge the
physical domain or count inactive allocation cells as useful updates.

## Implementation

- ABI 16 adds explicit row and plane pitches. Each staggered field can round its
  allocated width to 32 or 64 floats and optionally its height to eight rows.
  Logical field extents, material indexing, sources, DFT indices, and CPML
  positions are preserved. Padding and cropping occur around a complete native
  program, and their cost is included in executable timings.
- The fused single-step interior has compiled 32×8×8, 64×4×8, and 32×4×8
  variants. The combined CPML queue has 64×4, 32×8, and 32×4 variants.
  The previous automatic fusion eligibility is unchanged; selecting a tile
  does not force fusion onto a shape that uses separate phase queues.
- An explicit two-timestep deep-interior kernel loads a halo and advances
  H/E/H/E in shared memory. A third field bank prevents writes from corrupting
  other blocks' old inputs. It publishes intermediate fields for per-step
  monitors, advances the CPML and coupling band on each substep, and handles
  odd trailing steps and continuation. This is true temporal blocking, separate
  from CUDA graph batching and the existing single-step H/E fusion.

The two-step path requires uniform CPML, scalar H coefficients, a nonempty deep
interior, and no before-H source groups. Unsupported cases retain one-step
execution. Padding also applies only to eligible temporal-CPML programs.
Controls and eligibility are documented in [cuda/README.md](https://github.com/beamzorg/beamz/blob/fa151d28/cuda/README.md).

## Controlled reference measurements

128×256×512 material cells, 12-cell FP32 CPML, lossless binary dielectric,
one solved mode source, two fixed-aperture mode monitors, three frequencies,
256 timesteps. Four warmups and seven synchronized samples per current-build
configuration. Values are useful-cell GCUPS from median elapsed time.

| Current build configuration | GCUPS |
| --- | ---: |
| Default, unpadded, one step | [8.69](rtx3090-2026-09-18-layouts-v2/none/volume_128_256_512.json) |
| Row pitch rounded to 32 | [8.57](rtx3090-2026-09-18-layouts-v2/pad32/volume_128_256_512.json) |
| CPML tile 32×8 | [8.68](rtx3090-2026-09-18-layouts-v2/shell32x8/volume_128_256_512.json) |
| CPML tile 32×4 | [8.63](rtx3090-2026-09-18-layouts-v2/shell32x4/volume_128_256_512.json) |
| Row pitch 32 and CPML tile 32×8 | [8.47](rtx3090-2026-09-18-layouts-v2/pad32_shell32x8/volume_128_256_512.json) |
| Two-timestep prototype | [4.70](rtx3090-2026-09-18-layouts-v2/pair/volume_128_256_512.json) |
| Two-timestep prototype, CPML tile 32×8 | [4.67](rtx3090-2026-09-18-layouts-v2/pair_shell32x8/volume_128_256_512.json) |

An initial implementation sweep also tried row pitches 64, 32×8 and 64×8, and
the two alternative interior tiles. None beat its unpadded control. That build
used conditional pitch helpers and an out-of-line device function in the
two-step kernel, so its records are retained separately in
[the initial sweep](rtx3090-2026-09-18-layouts/). Removing the device call's
stack use improved the prototype from 3.46 to 4.70 GCUPS, without closing the
gap to one-step execution.

The current two-step prototype also measured
[4.85 GCUPS on 71×479×503](rtx3090-2026-09-18-layout-shapes/pair/irregular_71_479_503.json)
and [4.63 GCUPS over 1,024 steps](rtx3090-2026-09-18-layout-shapes/pair/long.json).
Longer graph execution does not remove its deficit.

## Shape dependence

All dimensions below are z×y×x. These cases contain approximately 16–17 million
material cells; narrow dimensions are compensated by long other dimensions.
The baseline is a frozen copy of the pre-padding backend used in the previous
shape study, not an older upstream implementation. Each entry uses five
samples after four warmups, and the same physical problem across variants.

| Shape / monitor workload | Frozen baseline | Current default | Pitch 32 | CPML 32×8 |
| --- | ---: | ---: | ---: | ---: |
| 1024×64×256, mode | 8.36 | 8.22 | 8.10 | 8.25 |
| 64×1024×256, mode | 7.41 | 7.34 | 7.22 | 7.33 |
| 1024×256×64, mode | 7.00 | 6.80 | 6.58 | 7.16 |
| 71×479×503, mode | 8.22 | 8.18 | 8.05 | 8.17 |
| 128×256×512, full-field | 8.35 | 8.26 | 8.11 | 8.24 |

[Raw shape records, manifests, and telemetry](rtx3090-2026-09-18-layout-shapes/).
The current default was 0.5–2.8% below the frozen baseline in this sequence.
Fresh-process repeats reversed that ranking on two cases, so a uniform default
regression is not established. The baseline reference ranged from 8.81 in the
initial sweep to 8.52 in the final repeat; these shared-device measurements do
not support interpreting small differences as a reliable gain or loss.

The final sequence ran CPML 32×8 first, then the current default, then the
frozen baseline, with seven samples per case:

| Shape / monitor workload | Frozen baseline | Current default | CPML 32×8 |
| --- | ---: | ---: | ---: |
| 128×256×512, mode | 8.52 | 8.69 | 8.68 |
| 1024×256×64, mode | 6.75 | 6.79 | 7.15 |
| 256×1024×64, mode | 6.95 | 6.77 | 7.10 |
| 1024×256×64, full-field | 5.36 | 5.32 | 5.52 |

[Raw repeat records](rtx3090-2026-09-18-layout-repeats/). The CPML tile improves
the two short-x cases by 5.3% and 4.9% against the current default. On the first
of these shapes, the earlier comparison was 7.16 versus 6.80. The full-field
control improves by 3.9%, but remains far from 9 GCUPS. This supports an explicit
tuning option; the tested envelope is too narrow to establish a universal
automatic tile rule.

## Why the proposed changes do not automatically win

**Storage alignment has a cost.** Yee components have different logical widths.
At x=64, several components have width 65; rounding those to 96 increases the
total field allocation by 24.0%. At x=512, pitch 32 adds only 3.0%. Padding can
improve address alignment while increasing memory footprint, pad/crop work,
and address-remapping work for monitors. The observed throughput does not
justify enabling it globally. Allocating a multiple of a tile also does not
remove the need to mask logical boundaries or preserve the 12-cell absorber.

**The two-step tile has substantial halo overhead.** Its useful tile is
16×8×4 = 512 cells; the staged 20×12×8 region contains 1,920 cells, a 3.75×
ratio. H1/E1/H2/E2 update shrinking halo regions, requiring approximately 1.87×
the ideal deep-interior update work before boundary clipping. Six field arrays
consume 46,080 bytes of dynamic shared memory per block. The current SM86
compiler report gives 36 registers/thread and no stack or local-memory usage
for this kernel. Intermediate field writes, the third field bank, source
handling, and the separately advanced coupling band remain additional costs.
These are implementation/resource facts, not a hardware-counter attribution
of the slowdown.

**CPML face orientation and monitor area still matter.** A different CPML tile
may help a narrow direction without helping other shapes. The earlier study
also showed that full-cross-section monitors in a short propagation domain
can dominate enough work to reduce throughput substantially. Fixed-aperture
mode monitors do not establish a guarantee for arbitrary monitor areas,
frequency counts, or device/source geometries.

The earlier [Nsight Systems breakdown](cuda-optimization-2026-09-17.md)
assigned about 61% of native kernel time to the fused interior, 31% to CPML,
and 7% to DFT on the reference case. Hardware performance counters remain
restricted on this machine; no measured DRAM-saturation or cache-stall claim
is made here.

## Validation and measurement limits

The [combined CSV](rtx3090-2026-09-18-layout-validation/summary.csv) contains
51 fresh-process benchmark records across six distinct physical domain shapes,
including the initial and refined implementations. Every GCUPS result was
recomputed from its raw sample median, and the CPML, material, precision,
selected temporal depth, and telemetry constraints were checked. This extends
the earlier volume/alignment study; it is not another sweep of all 26 shapes.

**103 distinct non-lossy hardware cases passed** on the current native build:
the existing non-lossy suite plus the new layout and temporal cases. Two
Hopper-only tests were skipped and two lossy tests were excluded. After
increasing the new cases' seeded field amplitudes to 1e-3 to expose small
update errors, all **66 new cases passed again**:

- 48 padding/tile cases: four storage layouts, fusion on/off, binary and smooth
  lossless materials, three interior/shell tile combinations, odd dimensions,
  mode sources, field monitors, and 17+16-step continuation.
- 15 temporal cases: 2, 3, 4, 5, and 33 steps across three layout/material/tile
  combinations, with continuation and complete state comparisons to JAX.
  The five-step cases include coincident mode sources at the CPML/core
  coupling band.
- Three temporal I/O cases: no source/monitor, source only, and six-component
  mode-monitor gathering. Tests assert that temporal depth two was actually
  selected. Every new parity case uses 12-cell CPML.

Comparisons include fields, CPML recurrence, DFT accumulators, and clocks.
They use the existing float32 tolerances (`rtol=3e-5`, leaf-scale absolute
floor), not bitwise equality. These small correctness domains complement the
large performance domains; they are not long-duration physical-accuracy tests.

The standalone interior-kernel harness passed memcheck with zero errors and
racecheck with zero hazards/errors/warnings. A separate full-pipeline memcheck
passed six selected temporal cases with zero device-memory errors. That run
used `--report-api-errors no` for XLA's existing CUDA module-symbol probes;
device-memory checking remained enabled. The standalone racecheck covers
shared-memory interior kernels, not a claim of exhaustive pipeline race
verification. **32 unit/ABI checks**, generated-binding consistency, lint, and
whitespace checks also passed.

[Validation logs and compiler resource report](rtx3090-2026-09-18-layout-validation/).
The final extension SHA-256 is
`4633145baad805971297529c4e4a06a6edffd5039c2ad8f6fe1aad1a96ab5d54`;
the frozen baseline is
`67d647a29ab09025bf13d4bdfdfc4c2e09b5ceee888b639d171553e97b5aa0be`.
Within each implementation sweep, source and binary fingerprints were constant.
Builds used precise math and CUDA 13.3.73 for SM80/86/89/90; execution validation
here is RTX3090 only. Oldest-supported CUDA 12.4 and H100 checks were not run.

All new performance cases use **exactly 12 CPML cells**, lossless materials,
FP32 fields/CPML, and unchanged logical domains. No lossy performance study was
run. Timings cover the synchronized warm compiled executable, including
sources, per-step DFT, CPML, padding and cropping where enabled. Mode solving,
compilation, and public-API/result processing are excluded.

The RTX3090 is shared with the desktop/T3 and an existing notebook process.
Benchmarks run sequentially and preserve at least 2 GiB of free VRAM. The board
remains at its existing 370 W limit because the earlier attempt to raise it was
denied by system permissions; these are not higher-power measurements. No
H100 results or universal shape-independent throughput guarantee are claimed.
Across the 51 runs, sampled utilization reached 100%, peak sampled temperature
was 73°C, and minimum sampled free VRAM was 9.22 GiB. Telemetry includes setup
and compilation as well as execution; 100% utilization is not proof of DRAM
bandwidth saturation.

## Next performance work

Keep padding and two-step blocking off by default. The measured CPML tile option
can help the tested short-x domains, but does not close the general-workload
gap. A faster temporal design would need less redundant halo work and fewer
intermediate global writes, for example by retaining only the intermediate
planes required by monitors and the coupling boundary. It must preserve
per-timestep source and monitor semantics. Simply increasing temporal depth
or rounding every physical dimension to a tile multiple is not justified by
these measurements.

Further automatic dispatch should be based on repeated CPML face-orientation
and monitor-aperture measurements across more sizes, then validated on full
device geometries. H100 and oldest-supported-toolkit checks remain separate
release gates. No changes were pushed.

## Reproduction

Build the ABI-16 component before running current Python sources. For example:

```sh
python scripts/benchmark_cuda_shapes.py --output /tmp/beamz-pad32 \
  --padding 32 --samples 7 \
  --cases volume_128_256_512 irregular_71_479_503 aspect_1024_256_64
python scripts/benchmark_cuda_shapes.py --output /tmp/beamz-pair \
  --temporal-steps 2 --samples 7 --cases volume_128_256_512 long
python scripts/benchmark_cuda_shapes.py --output /tmp/beamz-shell \
  --shell-tile 32x8 --samples 7 \
  --cases aspect_1024_256_64 aspect_256_1024_64 full_field_1024_256_64
```

Each directory contains the exact settings, case manifest, raw elapsed samples,
native schedule selection, logical field shapes, source/binary fingerprints,
and one-second telemetry. GCUPS uses logical material cells × timesteps only.
