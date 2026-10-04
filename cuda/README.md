# BeamZ CUDA component

CUDA is an optional native part of BeamZ, not a separately published package. This
directory builds the private component into the `beamz._cuda` module, which
registers typed JAX FFI targets while BeamZ remains usable with JAX alone. The
component's packaging metadata contains the `Private :: Do Not Upload` classifier,
so package indexes reject accidental publication.

The streamed backend replaces JAX's elementwise 3D Yee/CPML programs with fused
magnetic and electric CUDA launches. Bounded multi-step CUDA graphs own complete
3D runs, and one grouped program path covers arbitrary source batches and DFT
monitor schedules. JAX still owns tracing, buffers, and orchestration around that
small native interface, preserving BeamZ's public numerical semantics and JAX
fallback without duplicating configuration-specific FFI targets.

The native boundary has three typed FFI targets: a phase update, a sharded phase,
and a complete multi-step program. The implementation is
split by responsibility:

- `ffi_handler.cc` decodes JAX buffers into a validated `BeamzProgramLaunch`;
- `program.cu` owns timestep order and selects the in-place or temporal schedule;
- `graph.cu` owns bounded capture, semantic cache keys, and replay;
- `update.cu` owns Yee and CPML kernels;
- `sharded.cu` launches the global-coordinate-aware cells in `sharded_cell.h`;
- `io.cu` owns source injection and DFT monitor accumulation.

For monitor-heavy graphs, `io.cu` first prepares one window and complex phase per
monitor/frequency into small XLA-owned scratch buffers, then reuses those values
across every gathered point and field component. Short plans retain the original
single-kernel path so an extra launch cannot dominate their work.

`abi_layout.json` is the source of truth for target names, layout selectors, and
positional buffer constants. After editing it, regenerate both language bindings
with `python scripts/generate_cuda_abi.py`; CI uses `--check` to reject drift.

Install BeamZ from this checkout, then build its optional component in a CUDA 12
development environment:

```console
python -m pip wheel ./cuda --no-deps
python -m pip install cuda/dist/beamz_cuda_component-*.whl
```

Release builds use precise CUDA division and square-root behavior. Approximate CUDA
intrinsics are available only for controlled experiments with
`-DBEAMZ_CUDA_FAST_MATH=ON`; they must pass the same hardware parity suite before a
result can be used for promotion.

The wheel compiles SASS for SM80, SM86, SM89, and SM90. `backend="auto"` detects and
registers it lazily; `backend="cuda_streamed"` requests it explicitly. The first
release supports float32 3D grids. `auto` retains JAX for multi-GPU and 2D
simulations; only the explicitly selected CPML recurrence state may use BF16.

## Experimental sharded streamed execution

Explicit CUDA requests use the existing sharding API for 3D simulations, including
CPML, mixed PEC/absorbing faces, sponge layers, conductive materials, axis-uniform
and rectilinear grids, and full-tensor permittivity. The existing source/monitor
compiler supports mode, Gaussian-beam, overlapping, and custom phased sources;
mode/flux/field monitors and field recording use the shared observation path.

```python
result = simulation.advance(
    num_steps=100,
    backend="cuda_streamed",
    sharding={"axis": "z", "num_devices": 2, "backend": "gpu"},
)
```

This path has native C++ arithmetic and CPU FFI coverage, but has **not yet been
validated on multiple CUDA GPUs**. `auto` continues selecting JAX for sharded
requests. Rebuild the optional CUDA extension: this path requires component
**0.21.0 / ABI 21**, including `beamz_cuda_sharded`. ABI 20 passes separate neighbor faces to sharded
phases, avoiding full-field halo concatenations. ABI 21 adds an integer DFT
chunk offset so every graph uses the same invocation time origin. Rebuild the native wheel when
updating Python code across this ABI boundary.

All six components share partition interfaces along the selected x, y, or z
axis. `shard_map` exchanges one-cell neighbor halos before each H/E phase and
invokes the sharded FFI on local arrays. The native cells use global physical
coordinates for derivatives, boundary conditions, and CPML profile lookup.
Global JAX source, boundary,
and monitor operations preserve timestep ordering; results retain their logical
Yee shapes and continuation interface. Packed CPML state is cropped back to its
physical support so continuation can change the partition axis/device count or
resume with JAX without resetting the absorber memory.

CPML slabs transverse to the partition axis are sharded with the fields. Slabs
normal to it remain compact replicated low/high faces: each device updates only
its owned entries, then a sum assembles the disjoint results. Full-volume CPML
state is never allocated. Separable profiles and metric vectors are replicated.
Full-tensor electric updates use native curls/CPML followed by JAX's coupled
constitutive operator on physical supports; this can require extra collectives.
The existing lossless-electric-material restriction for full tensors still applies.

Multi-step native graphs and packed material codebooks are disabled on this
path. They need communication-aware scheduling and local material packing before
they can be restored. Sharding remains 3D-only, as in the existing JAX planner.
The first implementation exchanges both halo faces of all three source fields;
communication overlap and tangential-field-only exchange remain optimizations.

Run the CPU contract checks without CUDA:

```console
python -m pytest tests/unit/test_cuda_sharding.py tests/unit/test_cuda_sharded_features.py
```

These require a C++17 compiler. They compile the production FFI decoder and
`sharded_cell.h` with a test-only CPU launcher, execute actual collectives on
two/four CPU devices, and compare complete state against JAX. Cases include all
axes, uneven supports, asymmetric CPML/PEC intersections, source types, mode and
flux monitors, recordings, nonuniform metrics, tensor coupling, BF16 recurrence
state, and continuation. `BEAMZ_CUDA_CPU_SANITIZE=1` compiles this harness with
AddressSanitizer/UBSan (preload the compiler's ASan runtime when invoking Python).
The GPU launch wrapper, device races, and scaling still require GPU validation:

```console
python -m pytest tests/hardware/test_cuda_backends.py -k sharded_streamed
```

The hardware gate requires two/four GPUs and checks fields, monitor state and
continuation. Before promoting automatic selection, also run Compute Sanitizer
and inspect device memory/communication in a GPU profile for unintended full-grid
gathers and scaling. Host sanitizer coverage does not replace device checks.

BeamZ validates the component's explicit ABI version and complete streamed-target
manifest before registering any FFI handler. An older or partial component makes
`auto` fall back to JAX and causes explicit CUDA requests to fail with a
compatibility diagnostic.

Eligible regular-grid CPML simulations with packed source groups use two
XLA-owned field banks. Alternating frozen inputs and outputs removes in-place
read/write hazards and enables safe spatial fusion in the CPML-free core. The
combined core/shell queue accepts scalar, dense, and exact codebook E materials
with scalar H coefficients. Dense H coefficients retain the general schedule.
On SM86, grids with scalar H update coefficients, at least 12 Mi cells, and
x extent at least 384 use an H/E
interior kernel with a two-plane shared-memory ring. H sources are included in
interior and halo recomputation; CPML shell updates remain separate. Smaller or
narrower grids and unmeasured architectures retain separate phase queues.
See the [local RTX3090 measurements](../docs/reviews/cuda-optimization-2026-09-17.md)
for the measured envelope, remaining validation gates, and limitations.

`BEAMZ_CUDA_CPML_CORE_FUSION=0` disables interior fusion; `=1` forces it for
otherwise eligible schedules. Unset it for automatic selection. This diagnostic
override is read at native invocation and included in the graph-cache key; it
changes scheduling only. It allows small parity and sanitizer workloads to
exercise fusion, including odd tile tails and sources crossing the shell.

Experimental layout and temporal-blocking controls (ABI 20):

- `BEAMZ_CUDA_STORAGE_AXES=012|120|201` selects an internal cyclic storage
  permutation. The default `012` preserves canonical `(z,y,x)` storage;
  `120` stores `(y,x,z)` and `201` stores `(x,z,y)`. For example, `120` stores
  a physical `1024x256x64` domain as `256x64x1024`, with no geometry change.
  Fields, packed material IDs, CPML terms, sources and DFT component/gather
  plans are remapped together, and returned state remains canonical. The
  choice is snapshotted in the compiled-program cache key. Conversions occur
  around each native chunk and are included in executable timings. This
  opt-in currently requires a uniform 3D CPML native graph; supported sources
  and DFT monitors may be present or absent. Unsupported execution paths raise
  an error. A longer contiguous axis improved the measured narrow cases, but
  one fixed permutation is not universally faster.
- `BEAMZ_CUDA_FIELD_PADDING=none|32|64|32x8|64x8` rounds each field's storage
  width to 32 or 64 floats, optionally rounding its allocated height to eight
  rows. Logical Yee extents, CPML thickness, source positions and DFT indices
  remain unchanged. Padding/cropping happens once around a multi-step program,
  not once per timestep. The default is `none`; these are measured experiments,
  not universal performance recommendations.
- `BEAMZ_CUDA_CPML_TILE=auto|32x8x8|64x4x8|32x4x8` selects a compiled fused
  interior tile. `auto` retains 32x8x8 and the existing fusion eligibility rules.
  The native graph key includes the tile choice.
- `BEAMZ_CUDA_CPML_SHELL_TILE=64x4|32x8|32x4` selects a combined CPML queue
  variant. The default preserves the existing FP32 64x4 / BF16 32x4 selection;
  the other values explicitly select their named tile. The core queue remaps
  threads to a 64-wide tile, preserving logical coverage.
- `BEAMZ_CUDA_TEMPORAL_STEPS=1|2` selects an experimental paired schedule at
  `2`. The default rolling interior completes two Yee timesteps using a
  16x8x16 tile and three two-plane stages (about 13 KB shared memory). A third
  field bank keeps old fields frozen. Only a two-cell coupling rim and the
  monitor gather regions publish intermediate fields. CPML recurrence state
  still advances separately at both substeps; it is not temporally blocked.
  Sources are injected at their original H/E phase and timestep.
- `BEAMZ_CUDA_PAIR_TILE=16x8x16|32x4x16|32x8x16|single` selects the paired
  schedule's interior. `16x8x16` retains the bare-stencil winner. `single`
  preserves the ordinary, automatically selected field kernels and pairs only
  monitor processing; it avoids the temporal coupling band. This override is
  read at native invocation and included in the graph key.
- `BEAMZ_CUDA_CPML_TEMPORAL=1` opts into complete two-step CPML tiles,
  including on-chip intermediate auxiliary state and source injection. This
  also requests temporal depth two. Final auxiliary banks swap once per pair;
  odd tails retain one-step semantics. FP32 and BF16 auxiliary storage are
  supported. This experiment is currently slower than the established schedule.
- `BEAMZ_CUDA_CPML_PAIR_TILE=16x8x16|16x8x4|oriented` selects those CPML
  boundary tiles while retaining the winning 16x8x16 two-step interior.
  The oriented family uses 15x16x16 x-face and 16x15x16 y-face tiles with
  384 threads, 32x4x16 z-face tiles and 16x8x4 edge/corner tiles. The y-face
  kernel bounds register use to permit three resident blocks on SM86.
- `BEAMZ_CUDA_CPML_SPATIAL=1` selects an experimental fused single H/E step
  across the domain, including CPML. It is a control for measuring the cost of
  temporal halos and intermediate auxiliary storage. Complete temporal CPML
  takes precedence if both controls are set. Both native tile controls and the
  spatial choice are included in graph-cache keys.

Both paired schedules gather each monitor point at both times once, share the
interpolation geometry, reuse the samples across frequencies, and accumulate
DFTs in chronological order. Record intervals, Hann windows, normalization,
masked components, odd trailing steps and continuation retain their semantics.
Scratch samples cost `2 * monitor_count * 6 * max_points * sizeof(float)` bytes;
the conservative publication map uses one int32 per 16x8x1 logical region.
Paired execution requires uniform CPML, scalar H coefficients, a nonempty deep
interior, and no before-H source groups. Other cases retain the ordinary path.
`BEAMZ_CUDA_TEMPORAL_STEPS=1` remains the default; these experiments have not
established consistent 9 GCUPS for arbitrary realistic simulations.

Padding, shell tiling and temporal depth are snapshotted into the compiled
program's CUDA flags. Returned fields always have the original logical shapes.
Inspect `native_plans` in benchmark JSON to confirm whether the requested
temporal depth was eligible. Run `scripts/benchmark_cuda_shapes.py --help` for
controlled layout comparisons with 12-cell CPML and lossless mode-source cases.
The [RTX3090 layout and temporal study](../docs/reviews/cuda-layout-temporal-study-2026-09-18.md)
records the measured regressions and shape dependence. These controls are
experimental; padding and two-timestep blocking are not enabled automatically.
The [expanded CPML and bare temporal study](../docs/reviews/cuda-cpml-tiles-and-bare-temporal-2026-09-18.md)
tests tile generality and a separate rolling-plane temporal prototype. Two-step
rolling reuse helps the tested bare periodic stencil. Its rolling design is now
available through the backend's temporal flag; realistic CPML coupling adds
substantial cost. The standalone source is
`cuda/tests/temporal_bulk_benchmark.cu`.
The [realistic integration study](../docs/reviews/cuda-realistic-temporal-integration-2026-09-18.md)
records paired source/monitor integration, correctness checks and large-domain
comparisons. `scripts/benchmark_cuda_cpml_tiles.py --study temporal_schedule`
compares the ordinary schedule, rolling two-step core and paired monitors on the
same initialized problem, with interleaved timing rounds.

For memory-constrained CPML runs,
`BEAMZ_CUDA_CPML_PSI_PRECISION=bf16` stores only absorber recurrence state in
BF16 while fields and recurrence arithmetic remain FP32. The default is `fp32`;
use BF16 only after validating application-level accuracy. On GA102 the combined
queue uses a precision-specific `32 × 4` absorber tile and remaps the same 128
threads to a `64 × 2` recurrence-free core tile; FP32 retains its measured-optimal
`64 × 4` queue.

`BEAMZ_CUDA_DISABLE_GRAPH_CACHE=1` remains as a diagnostic switch.
`BEAMZ_CUDA_GRAPH_CACHE_CAPACITY` sets the completed-entry LRU target (default
`32`, valid range `0`–`4096`); a capacity of zero disables persistent graph
executables. In-flight entries may temporarily exceed the target so the native
cache never destroys an executable still referenced by a CUDA stream. These
choices and the CPML precision setting are snapshotted into the immutable
compiled-program key. The scheduling-only fusion override above is separate.
Set `BEAMZ_CUDA_GRAPH_CACHE_STATS=1` while profiling to emit an initial and then
periodic cache hit-rate and mean graph-instantiation-time summary. It is off by
default and has no counter/timing work on normal graph replays.

No CUDA result is promoted without all of the following on real hardware:

- compile with the oldest supported CUDA toolkit and import beside supported JAX;
- compare fields, CPML recurrence buffers, monitor accumulators, clocks, and
  continuation state with JAX over bare, lossy, PEC, CPML, source, and DFT cases;
- run Compute Sanitizer memcheck and racecheck;
- capture Nsight Compute memory throughput and the canonical H100 GCUPS records.

For a standalone CPML interior sanitizer check without JAX initialization:

```console
mkdir -p .cache/cuda-check
nvcc -std=c++17 -arch=sm_86 -Icuda/src \
  cuda/tests/cpml_core_sanitizer.cu cuda/src/update.cu \
  -o .cache/cuda-check/cpml_core_sanitizer
compute-sanitizer --tool memcheck --error-exitcode 99 .cache/cuda-check/cpml_core_sanitizer
compute-sanitizer --tool racecheck --error-exitcode 99 .cache/cuda-check/cpml_core_sanitizer
```

This exercises packed/dense coefficients, overlapping H/E sources, partial tiles,
padded storage, all three ring tiles, and the two-timestep core. It checks memory
access and synchronization in these interior kernels;
the complete-state numerical oracle remains `tests/hardware/test_cuda_backends.py`.

The CUDA workflow always compiles the private component and imports it through the
BeamZ namespace in a CUDA development container. It is retained only as a CI
artifact and is never published independently. Repositories with an H100
self-hosted runner can set the Actions variable
`BEAMZ_H100_RUNNER_ENABLED=true` and label that runner `h100` to additionally run the
32-step PEC/CPML parity envelope and publish canonical benchmark JSON artifacts.

The host FFI decoder deliberately has no CUDA-header dependency and can be checked
on developer machines with the JAX headers alone:

```console
clang++ -std=c++17 \
  -I"$(python -c 'import jax; print(jax.ffi.include_dir())')" -Icuda/src \
  -fsyntax-only cuda/src/ffi_handler.cc
```

### Automatic RTX3090 storage and shell selection

With no manual `BEAMZ_CUDA_*` overrides, `Simulation.compile()` estimates storage
for eligible RTX3090 programs from geometry. **The default performs no calibration
runs**, reads no tuning profiles and does not launch candidate executables.
It counts thread slots in the existing 64x4 CPML queue for the three cyclic
orders (`012`, `120`, `201`), including Yee staggering and partially filled
tiles. It changes layout only for an estimated work reduction of at least 25%.
This conservative margin describes geometric work, not predicted runtime gain.
The shell tile remains 64x4, and padding stays off.

The physical grid, CPML12, source/monitor sampling and FP32 precision remain
unchanged. Existing automatic H/E fusion still applies within each layout;
experimental CPML temporal blocking is not selected. The estimate repairs the
recorded narrow-x cases while retaining canonical storage on the wide/irregular
controls. It is not a guarantee of the global optimum or of beating another
revision on every workload. See the [analysis and offline replay](../docs/reviews/cuda-layout-prediction-2026-09-18.md).

```python
program = sim.compile(num_steps=256, backend="cuda_streamed")
from beamz.simulation.cuda.tuning import tuning_report
print(program.config.cuda_storage_axes)
print(tuning_report(program))  # choice, geometric costs, reason, calibration_s=0

# Inspect a shape without compiling or accessing the GPU:
from beamz.simulation.cuda.tuning import predict_layout
print(predict_layout((1024, 256, 64))["choice"])  # 120 / 64x4
```

`BEAMZ_CUDA_AUTOTUNE=auto` is the predictive default; `off` disables selection.
Other explicit `BEAMZ_CUDA_*` overrides disable automatic selection, preserving
manual experiments. Configuration is frozen into the immutable program and
isolated in the compiled-program cache. Prediction reports have `validated=False`
because no numerical validation trial was performed for the current request.

The initial automatic scope is one RTX3090, at least 8,388,608 logical cells,
at least 32 steps, uniform 3D lossless diagonal materials, CPML12, and native-compatible
slab sources / DFT monitors. Other workloads retain existing dispatch. The cost
proxy omits cache effects, fused-core timing, transpose overhead and source/monitor
cost. In particular, it cannot resolve all y-versus-z orientation effects. Short
runs or heavy monitors may have a different optimum. The previously recorded
branch-versus-main CPML numerical differences remain unresolved.

Explicit `BEAMZ_CUDA_AUTOTUNE=calibrate` enables the measured selector and its
persistent cache; `refresh` forces recalibration after clearing the program cache
or restarting. This mode compares all three orders and 64x4/32x8 shell tiles,
requires bitwise complete-state parity against the branch canonical layout, and
accepts a timing improvement only above 3% with wins in both balanced halves.
It uses a private initial state, two additional warmups and six timing rounds,
including conversion costs. Long runs probe at most 256 steps while retaining
the requested execution horizon. First-use cost was 33–37 seconds in recorded
large cases. This cost is absent in default predictive mode.

Only explicit calibration uses GPU telemetry, a 4-GiB free-memory guard and an
85-C temperature limit. It makes no power-limit changes. Profiles persist under
`$XDG_CACHE_HOME/beamz/cuda-tuning` (otherwise `~/.cache/beamz/cuda-tuning`), or
`BEAMZ_CUDA_TUNING_CACHE`. Keys include exact workload, solver fingerprints, JAX,
GPU, driver and power limit. Writes are atomic; unwritable caches do not prevent
execution. Default prediction ignores these profiles.
