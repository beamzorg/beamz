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

The native boundary has four typed FFI targets: a phase update, a sharded phase,
a complete multi-step program, and the explicit Hopper experiment. The implementation is
split by responsibility:

- `ffi_handler.cc` decodes JAX buffers into a validated `BeamzProgramLaunch`;
- `program.cu` owns timestep order and selects the in-place or temporal schedule;
- `graph.cu` owns bounded capture, semantic cache keys, and replay;
- `update.cu` owns Yee and CPML kernels;
- `sharded.cu` launches the global-coordinate-aware cells in `sharded_cell.h`;
- `io.cu` owns source injection and DFT monitor accumulation.

`abi_layout.json` is the source of truth for target names, layout selectors, and
positional buffer constants. After editing it, regenerate both language bindings
with `python scripts/generate_cuda_abi.py`; CI uses `--check` to reject drift.

On SM90, the experimental `beamz_cuda_hopper` target uses the same ABI and arithmetic
but maps each component to `32 × 4 × 2` spatial tiles. Each derivative input stages
only its directional halo in shared memory, reusing values across neighboring
updates while keeping the x direction warp-contiguous. Backend selection only
exposes this target on compute capability 9.0 or newer. It remains explicit-only
until hardware parity and throughput gates justify promotion; `auto` and generic
`cuda` use the streamed path.

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
registers it lazily; `backend="cuda_streamed"` requests it explicitly and
`backend="cuda_hopper"` requests the tiled target. The first
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
**0.13.0 / ABI 13**, including `beamz_cuda_sharded`.

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
they can be restored. Sharding remains 3D-only, as in the existing JAX planner;
the explicit Hopper variant remains single-device.
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

Regular-grid, lossless CPML simulations with packed source groups use two
XLA-owned field banks. Alternating frozen inputs and outputs removes in-place
read/write hazards and enables safe spatial fusion in the CPML-free core. The
validated temporal, packed-source, material-codebook, and combined CPML queues
are unconditional; the legacy experimental schedule switches and their dead
kernels have been removed.

For memory-constrained CPML runs,
`BEAMZ_CUDA_CPML_PSI_PRECISION=bf16` stores only absorber recurrence state in
BF16 while fields and recurrence arithmetic remain FP32. The default is `fp32`;
use BF16 only after validating application-level accuracy. On GA102 the combined
queue uses a precision-specific `32 × 4` absorber tile and remaps the same 128
threads to a `64 × 2` recurrence-free core tile; FP32 retains its measured-optimal
`64 × 4` queue.

`BEAMZ_CUDA_DISABLE_GRAPH_CACHE=1` remains as a diagnostic switch. It and the
CPML precision choice are snapshotted into the immutable compiled-program key;
native execution never rereads environment variables.

No CUDA result is promoted without all of the following on real hardware:

- compile with the oldest supported CUDA toolkit and import beside supported JAX;
- compare fields, CPML recurrence buffers, monitor accumulators, clocks, and
  continuation state with JAX over bare, lossy, PEC, CPML, source, and DFT cases;
- run Compute Sanitizer memcheck and racecheck;
- capture Nsight Compute memory throughput and the canonical H100 GCUPS records.

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
