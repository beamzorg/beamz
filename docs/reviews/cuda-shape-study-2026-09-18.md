# RTX3090 domain size, aspect ratio, and alignment study

This follow-up fixes CPML at **12 cells on every face** and excludes lossy
materials. It tests the previously built CUDA streaming candidate; no native
kernel or field-layout change is made in this study.

**Consistent 9 GCUPS is not achieved.** Equal-volume aspect ratios measured
6.95–8.35 GCUPS under normal dispatch, with fixed-aperture mode monitors. The
128 × 256 × 512 reference measured 8.81 and 8.58 in separate processes. Fully
irregular domains measured 8.15 and 8.20. These are executable throughputs.

The results support testing storage padding, but do not establish a beneficial
domain-rounding rule. Nearby-dimension results overlap fresh-process variation:
127 × 256 × 512 initially beat the reference (8.84 versus 8.81), then trailed it
on repeat (8.48 versus 8.58). The 504-wide candidate similarly changed ranking
against the 512-wide reference. A same-logical-domain padded/unpadded comparison
is needed before claiming an alignment optimization.

Shape effects remain after controlling volume, CPML fraction, and monitor area.
In the same fallback schedule, swapping y and z at x=256 changed throughput from
7.09 to 8.31 initially and from 7.38 to 7.97 on repeat. Forcing fusion everywhere
is unsuitable: the 1024 × 256 × 64 case fell from 6.99 to 5.73 GCUPS. Conversely,
turning fusion off for 64 × 256 × 1024 fell from 8.14 to 7.61. Hardware counters
were not collected, so these observations do not establish a cache or bandwidth
mechanism for the directional differences.

## Measurements

Completed **40 fresh-process benchmark runs**, covering **26 distinct domain
shapes** and 35 configurations including monitor, duration, and schedule controls.
Every case used 12-cell CPML and lossless materials. The primary sweep used five
samples per case; supplementary/repeat and irregular cases used seven. Case-order
seeds were 19, 73, and 19 respectively. Each table value links to its raw record;
the [combined CSV](rtx3090-2026-09-18-shapes/summary.csv) includes all 40 runs.

### Equal-volume aspect ratios

All six permutations contain 16,777,216 material cells, use the same fixed-size
mode-monitor apertures, and have an approximate 44.7% CPML shell fraction. Normal
dispatch uses fusion when x=1024 and the fallback when x=64 or 256. The y/z pair
with x=256 uses the same schedule, so its difference is not a dispatch switch.

| Case | z × y × x | First GCUPS | Repeat GCUPS |
| --- | --- | ---: | ---: |
| aspect | 1024 × 256 × 64 | [6.99](rtx3090-2026-09-18-shapes/aspect_1024_256_64.json) | — |
| aspect | 1024 × 64 × 256 | [8.31](rtx3090-2026-09-18-shapes/aspect_1024_64_256.json) | [7.97](rtx3090-2026-09-18-shapes-checks/aspect_1024_64_256.json) |
| aspect | 256 × 1024 × 64 | [6.95](rtx3090-2026-09-18-shapes/aspect_256_1024_64.json) | — |
| aspect | 256 × 64 × 1024 | [8.35](rtx3090-2026-09-18-shapes/aspect_256_64_1024.json) | — |
| aspect | 64 × 1024 × 256 | [7.09](rtx3090-2026-09-18-shapes/aspect_64_1024_256.json) | [7.38](rtx3090-2026-09-18-shapes-checks/aspect_64_1024_256.json) |
| aspect | 64 × 256 × 1024 | [8.14](rtx3090-2026-09-18-shapes/aspect_64_256_1024.json) | — |

### Volume and irregular dimensions

The 64 × 128 × 256, 96 × 192 × 384, 128 × 256 × 512, and
144 × 288 × 576 cases share a 1:2:4 aspect ratio. Smaller cases are scale controls;
the large aspect-ratio comparisons are not relying on undersized grids.

| Case | z × y × x | First GCUPS | Repeat GCUPS |
| --- | --- | ---: | ---: |
| volume | 64 × 128 × 256 | [6.67](rtx3090-2026-09-18-shapes/volume_64_128_256.json) | — |
| volume | 128 × 128 × 256 | [7.88](rtx3090-2026-09-18-shapes/volume_128_128_256.json) | — |
| scale | 96 × 192 × 384 | [7.94](rtx3090-2026-09-18-shapes-checks/scale_96_192_384.json) | — |
| volume | 128 × 256 × 256 | [8.15](rtx3090-2026-09-18-shapes/volume_128_256_256.json) | — |
| irregular | 97 × 289 × 593 | [8.15](rtx3090-2026-09-18-shapes-irregular/irregular_97_289_593.json) | — |
| volume | 256 × 256 × 256 | [8.83](rtx3090-2026-09-18-shapes/volume_256_256_256.json) | — |
| volume | 128 × 256 × 512 | [8.81](rtx3090-2026-09-18-shapes/volume_128_256_512.json) | [8.58](rtx3090-2026-09-18-shapes-checks/volume_128_256_512.json) |
| irregular | 71 × 479 × 503 | [8.20](rtx3090-2026-09-18-shapes-irregular/irregular_71_479_503.json) | — |
| scale | 144 × 288 × 576 | [8.47](rtx3090-2026-09-18-shapes-checks/scale_144_288_576.json) | — |
| volume | 128 × 256 × 768 | [8.66](rtx3090-2026-09-18-shapes/volume_128_256_768.json) | — |

### Nearby dimensions and alignment candidates

The reference is 128 × 256 × 512. No dimensions were padded: these are distinct
physical domains probing potential alignment and tile-tail effects. Differences
of a few percent need to be interpreted alongside fresh-process repeat variation.

| Case | z × y × x | First GCUPS | Repeat GCUPS |
| --- | --- | ---: | ---: |
| volume | 128 × 256 × 512 | [8.81](rtx3090-2026-09-18-shapes/volume_128_256_512.json) | [8.58](rtx3090-2026-09-18-shapes-checks/volume_128_256_512.json) |
| alignment | 128 × 256 × 504 | [8.76](rtx3090-2026-09-18-shapes/alignment_x_504.json) | [8.69](rtx3090-2026-09-18-shapes-checks/alignment_x_504.json) |
| alignment | 128 × 256 × 511 | [8.82](rtx3090-2026-09-18-shapes/alignment_x_511.json) | — |
| alignment | 128 × 256 × 513 | [8.54](rtx3090-2026-09-18-shapes/alignment_x_513.json) | — |
| alignment | 128 × 256 × 520 | [8.70](rtx3090-2026-09-18-shapes/alignment_x_520.json) | — |
| alignment | 128 × 256 × 536 | [8.68](rtx3090-2026-09-18-shapes/alignment_x_536.json) | — |
| alignment | 128 × 256 × 544 | [8.50](rtx3090-2026-09-18-shapes/alignment_x_544.json) | — |
| alignment | 128 × 255 × 512 | [8.47](rtx3090-2026-09-18-shapes/alignment_y_255.json) | — |
| alignment | 128 × 257 × 512 | [8.50](rtx3090-2026-09-18-shapes/alignment_y_257.json) | — |
| alignment | 127 × 256 × 512 | [8.84](rtx3090-2026-09-18-shapes/alignment_z_127.json) | [8.48](rtx3090-2026-09-18-shapes-checks/alignment_z_127.json) |
| alignment | 129 × 256 × 512 | [8.47](rtx3090-2026-09-18-shapes/alignment_z_129.json) | — |

### Monitor and duration controls

| Case | z × y × x | First GCUPS | Repeat GCUPS |
| --- | --- | ---: | ---: |
| After 1,024 conditioning steps | 128 × 256 × 512 | [8.49](rtx3090-2026-09-18-shapes/developed.json) | — |
| Full-cross-section field monitors | 1024 × 256 × 64 | [5.46](rtx3090-2026-09-18-shapes/full_field_1024_256_64.json) | — |
| Full-cross-section field monitors | 128 × 256 × 512 | [8.33](rtx3090-2026-09-18-shapes/full_field_128_256_512.json) | — |
| Full-cross-section field monitors | 64 × 256 × 1024 | [7.86](rtx3090-2026-09-18-shapes/full_field_64_256_1024.json) | — |
| 1,024 steps | 128 × 256 × 512 | [8.54](rtx3090-2026-09-18-shapes/long.json) | — |

### Alternative scheduling controls

| Case | z × y × x | First GCUPS | Repeat GCUPS |
| --- | --- | ---: | ---: |
| Force fusion on | 1024 × 256 × 64 | [5.73](rtx3090-2026-09-18-shapes-checks/dispatch_1024_256_64_on.json) | — |
| Force fusion on | 1024 × 64 × 256 | [7.33](rtx3090-2026-09-18-shapes-checks/dispatch_1024_64_256_on.json) | — |
| Force fusion on | 64 × 1024 × 256 | [7.41](rtx3090-2026-09-18-shapes-checks/dispatch_64_1024_256_on.json) | — |
| Force fusion off | 64 × 256 × 1024 | [7.61](rtx3090-2026-09-18-shapes-checks/dispatch_64_256_1024_off.json) | — |

All runs used the same extension (`67d647a29ab09025bf13d4bdfdfc4c2e09b5ceee888b639d171553e97b5aa0be`), source
fingerprint, and realistic benchmark fingerprint. Recomputing GCUPS from all raw
sample medians, checking the physical/staggered field shapes, and checking the
12-cell CPML/material/precision settings succeeded for every record. These are
performance runs, not an additional JAX numerical-parity sweep.

GPU utilization reached 100% on the large cases. The power limit remained 370 W;
no higher-power result is claimed. Peak sampled temperature across the study was
69°C, and minimum sampled free VRAM was 5.41 GiB. Telemetry samples span
setup as well as execution and are not a hardware-counter attribution of stalls.

## Method

All dimensions in this report are material-grid cells in **z, y, x** order. The standard
case contains a straight lossless dielectric waveguide, a solved TE mode source,
and two mode monitors with a fixed **1.6 × 0.8 µm** aperture and three frequencies.
Grid spacing is 80 nm. Fields and CPML state use FP32. Unless identified otherwise,
each fresh process performs four warmups and five synchronized samples of 256
complete timesteps. Setup, compilation, and modal result projection are excluded.
These are warm executable measurements, not public `Simulation.advance` timings.

The primary sweep varies volume, permutes all three axes of a 64 × 256 × 1024
domain, and perturbs each dimension around 128 × 256 × 512. Additional x sizes
probe both material-domain and CPML-interior tile boundaries. Equal-volume axis
permutations keep the approximate CPML shell fraction and material filling
fraction constant. The waveguide still propagates along x: this is a domain-shape
study, not a rotation of the entire device or a source-orientation study.

Separate full-field controls replace the small mode monitors with two
full-clear-aperture field monitors recording Ey/Ez/Hy/Hz. They measure a different
observation workload and must not be read as a same-area monitor-type comparison.
Long and developed-field controls use 1,024 timesteps and 1,024 conditioning steps,
respectively. Supplementary cases keep a 1:2:4 aspect ratio while scaling volume,
and force the alternative fusion selection on selected aspect ratios.

Case ordering is shuffled with a recorded seed. Raw JSON contains every timing,
field shape, monitor region, source specification count, and source/binary hash.
Per-case telemetry records GPU utilization, memory, clocks, temperature, and power.
The runner keeps cases sequential and aborts its own benchmark if free GPU memory
falls below 2 GiB or GPU temperature reaches 85 C. Existing desktop/T3 and notebook
processes are left running. This is a shared-device local study, not an exclusive
GPU measurement or a statistical guarantee.

## Interpreting padding

Storage padding is a plausible experiment, but domain rounding and storage
padding are different changes. Increasing the physical domain changes distances,
CPML placement, and potentially the physics. A transparent padded implementation
must preserve the requested logical extents and all physical source, monitor,
material, and CPML coordinates. Padding must never enter the useful-cell GCUPS
numerator or act as an extra electromagnetic region.

The current native `BeamzBuffer` has dimensions but no independent row/plane
strides. CUDA field accesses compute `(z * ny + y) * nx + x`. Yee staggering means
that a nominal x extent of 512 produces component row widths of both 512 and 513.
Rounding the nominal domain to a multiple of 32 therefore does not align every
component. Fused interior tiles are 32 × 8 × 8; the FP32 combined CPML queue uses
64 × 4 tiles. With CPML fixed at 12, interior tile alignment and allocation-row
alignment also occur at different domain dimensions.

A real padding experiment needs separate logical extents and physical pitches
per component, with consistent addressing for field kernels, source injection,
monitor gathering, material codebooks, and CPML state. Candidate row pitches
rounded to 32 or 64 floats should be compared on the **same physical simulation**.
Those are experimental choices, not an established optimum. Numerical validation
must compare complete fields, CPML recurrence, and monitor accumulators with the
unpadded implementation, including odd dimensions and continuation.

This dimension sweep alone cannot establish that padding would help: it changes
physical extents, CPML fraction, and tile tails as well as memory strides. It can
identify useful candidates for a controlled pitch experiment. No padding has been
implemented or validated by these measurements.

## Recommended next experiments

1. Add an experimental storage-pitch option preserving logical field extents,
   12-cell CPML, and all physical coordinates. Compare padded and unpadded runs
   on identical cases from this matrix, reporting useful-cell GCUPS and memory
   overhead. Validate complete numerical state before performance claims.
2. Investigate CPML face tiling separately. Its work depends on face orientation
   as well as shell fraction; the current FP32 queue uses 64 × 4 tiles. Measure
   orientation-specific tile choices on the equal-volume permutations before
   adding further dispatch conditions. Global fusion already failed that test.
3. Retain fixed-aperture mode monitors for layout experiments and rerun the
   full-aperture controls for application acceptance. The latter reached only
   5.46 GCUPS in the short-x case; storage alignment cannot eliminate that extra
   monitoring workload. Extend acceptance to source orientations and more device
   geometries after identifying a repeatable gain on the controlled study.

## Reproduction

Use a CUDA-enabled Python environment and this checkout's built extension:

```bash
python scripts/benchmark_cuda_shapes.py \
  --python /path/to/cuda-enabled/python \
  --output /path/to/new-results-directory
```

The output directory must not already exist. `--dry-run` prints the case list;
`--cases NAME ...` selects a subset. The runner always passes `--pml 12`,
`--material binary`, and `--source mode`. The underlying realistic benchmark now
also defaults to 12 CPML cells. Historical benchmark results in the earlier report
retain their original absorber thicknesses.
