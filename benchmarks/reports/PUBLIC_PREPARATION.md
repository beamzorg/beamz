# Public preparation: bounded material generation

Status: **implemented and locally validated for the partitioned FP32 3D path;
H100 acceptance is pending.** The previous 15-billion-cell public run still has
no completed preparation/compilation/stepping result. These changes do not turn
that timeout into a pass.

## Changes

The ordinary `Simulation.compile` / `advance` pipeline now selects the actual
mesh from shape metadata before volume allocation. Partitioned FP32 3D grids
with diagonal constitutive materials and CPML/PEC boundaries use the new path;
2D, full tensors, sponge boundaries and x64 retain their established paths.
Direct Yee material inputs are supported after ordinary rasterization.

- Yee materials are immutable region recipes. A requested region gathers only
  its required neighboring cells and preserves the existing axis-ordered
  interpolation, including physical edges and partition seams.
- Final CUDA coefficients or JAX constitutive arrays are generated in 16 MiB
  output tiles and written into donated buffers on the destination device.
  Transfers complete before reading the next tile. Intermediate reader scratch
  is a small multiple of a tile (including interpolation neighbors), rather
  than another global volume or a complete host shard.
- Compilation retains recipes for source normalization and diagnostics, rather
  than dense sampled Yee material copies. Runtime coefficient arrays are placed
  once and reused. A change in continuation horizon reuses the same material
  buffers while rebuilding source/monitor plans.
- Initial zero fields are broadcast host metadata during planning, then
  allocated with final device sharding. CPML zero templates likewise use scalar
  broadcast storage.
- PEC masks are stored as separable axis profiles, including padded cells.
- Mode-source symmetry uses tiled float64 comparisons with the same tolerance;
  scalar permeability is broadcast instead of expanded into a global volume.
- CPML extrusion preserves the immutable input when exact tiled comparison
  proves no values would change. If extrusion changes values, its established
  full-array copy remains. Public `MaterialGrid` input-copy guarantees remain.

No stepping kernel, CPML precision, source waveform, or monitor frequency count
was changed. Local state and spectral tolerances were not relaxed.

`BEAMZ_TRACE_PREPARATION=/path/phases.jsonl` records start/end/error events,
host elapsed time and process peak RSS (current RSS too on Linux). Phase times
include host work/enqueues, not implicit device synchronization; nested phases
are not additive. A killed process leaves its last start record for diagnosis.

## Local measurements

Fresh processes on macOS ARM, Python 3.11.14, JAX 0.9.0, two virtual CPU devices,
z partition. Each case uses the **public** binary waveguide workload at 80 nm,
a mode source, 12-cell FP32 CPML, and two 101-frequency mode monitors.
The measured endpoint includes initial state and coefficient placement with a
completion barrier. It excludes executable compilation and stepping.
Baseline is `a32227d` (previous solver plus tracing); fixed implementation is
`2ec4b0d` for 128³/256³ and `3f82dd8` for 384³. The latter adds continuation reuse
and dtype compatibility; these cases start fresh and use FP32 inputs.

| Cells | Baseline peak RSS | Fixed peak RSS | Reduction | Baseline preparation | Fixed preparation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128³ (2.10M) | 0.824 GiB | 0.719 GiB | 12.7% | 3.645 s | 4.142 s |
| 256³ (16.78M) | 2.265 GiB | 1.360 GiB | 40.0% | 4.522 s | 4.160 s |
| 384³ (56.62M) | 7.357 GiB | 2.184 GiB | 70.3% | 4.214 s | 5.306 s |

These are **one process per configuration**, with ordinary compilation-cache
behavior and unmanaged host load (some runs overlapped local tests). Preparation
timing is mixed and diagnostic only; no speedup claim is made. CPU RSS also
includes simulated device storage, so it must not be extrapolated directly to
H100 host memory. This establishes reduced local peak memory, not 15B capacity,
GPU transfer performance, or the <5% steady-state regression requirement.

Raw measurements and phase logs are in `public-preparation-local/`. Reproduce:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  BEAMZ_TRACE_PREPARATION=phases.jsonl \
  python -m scripts.benchmark_public_preparation \
  --shape 256 256 256 --devices 2 --axis z --backend jax --output result.json
```

## Validation

Final local suite: **163 passed** (log alongside raw measurements): compiled
engine, native design rasterization, mode launch planner, preparation tracing,
and the new region preparation tests. The latter check global-vs-region
sampling at edges, coefficient parity, bounded placement reads, public modal
preparation with full recipe materialization forbidden, JAX/native-CPU CUDA
x/z partitions, continuation, and shared coefficient ownership across horizons.
An earlier broader sharded suite also passed all **27 tests**: pure-JAX local
sharding/gradients/DFT, CUDA CPU-reference features and memory policy. Compact
boundary-mask tests passed separately. CPU reference execution is not CUDA FFI
or real cross-GPU communication validation.

## Outstanding work and acceptance

1. Run one-/two-H100 numerical and continuation gates, including native CUDA
   placement and paired baseline/fixed throughput wherever baseline fits.
2. Run `(10000,1250,1200)` through the **same public modal pipeline**, record
   phase-level RSS and every GPU allocator peak, compile, and execute repeated
   donating segments. Keep CUDA and JAX capacity conclusions separate.
3. Require under 5% warm-stepping slowdown using repeated paired runs; local
   preparation wall times above do not test that requirement.
4. Geometry rasterization still constructs a dense `MaterialGrid` before this
   compiler path. Bounded native rasterization into owned immutable material
   storage is a subsequent stage, requiring independent smoothing/overlap/seam
   parity tests. This change is not an out-of-core geometry implementation.
5. Required CPML extrusion copies, full-tensor/x64/sponge fallback preparation,
   optional whole-domain energy diagnostics, and public cropping when no logical
   dimension is divisible by the device count remain separate capacity limits.
   The 15B acceptance shape has divisible logical dimensions.

No new RunPod resource was provisioned for these local changes. Additional
hardware budget has been requested before resuming H100 acceptance.
