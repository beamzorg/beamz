# Independent native CUDA runtime

`_cuda_native` executes the existing CUDA solver directly from prepared NumPy
buffers. Its build and runtime do not require JAX. It currently consumes the
prepared native ABI; it does **not** provide an independent scene/material/source
compiler or the high-level `Simulation` API.

## Build and import

Use a Python environment containing NumPy and nanobind. The local reference build
uses Python 3.11.15, NumPy 2.4.1, nanobind 3.1.0, CUDA 13.3, and architecture 86.
No existing `_cuda` installation needs to be replaced.

```bash
cmake -S cuda -B /tmp/beamz-native -G Ninja \
  -DPython_EXECUTABLE=/path/to/native-venv/bin/python \
  -Dnanobind_DIR=/path/to/nanobind/cmake \
  -DCMAKE_CUDA_COMPILER=/path/to/nvcc \
  -DBEAMZ_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release -DBEAMZ_CUDA_FAST_MATH=OFF \
  -DBEAMZ_CUDA_BUILD_FFI=OFF -DBEAMZ_CUDA_BUILD_NATIVE=ON
cmake --build /tmp/beamz-native --target _cuda_native -j2
PYTHONPATH=/tmp/beamz-native /path/to/native-venv/bin/python -c \
  'import _cuda_native; print(_cuda_native.jax_independent)'
```

`nanobind.cmake_dir()` gives the nanobind CMake directory. Native-only configuration
skips JAX header discovery entirely. The optional install destination is top-level,
so `import _cuda_native` does not execute `beamz/__init__.py`.

The existing `cuda/pyproject.toml` still packages the JAX FFI component and lists
JAX as a build/runtime dependency. The native-only path above uses CMake directly;
an independent native wheel and NumPy scene preparation are separate work.

For a matched FFI/direct comparison, enable both `BEAMZ_CUDA_BUILD_FFI` and
`BEAMZ_CUDA_BUILD_NATIVE` in one build using a JAX-capable build environment.
Both modules embed the **same compiled** `beamz_cuda_core` OBJECT-library files
for graph, source/DFT, program, and update code. There is no additional runtime
shared library to install. The native module still runs in a JAX-free environment.

## Prepared packet and session

Inputs are ordered NumPy arrays plus the existing scalar attribute dictionary.
The supported buffer types are native-endian float32, int32, and NumPy bfloat16
arrays where that dtype is available. Arrays must provide contiguous storage.
Plans, coefficients, sources, and monitor metadata must belong to the same
prepared ABI version; device-side queue contents are trusted compiler metadata.

| Packet | Inputs | Outputs |
| --- | ---: | ---: |
| Mstep with CPML | 134 | 36 |
| Mstep without CPML | 84 | 24 |
| In-place program with CPML | 117 | 24 |
| In-place program without CPML | 67 | 12 |

`kind="program"` also accepts a full Mstep packet and ignores its extra plan and
workspace entries. Its output layout then retains those extra workspace entries.
Tensor, dispersive, adjoint, and sharded packet frontends are not implemented here.

```python
import _cuda_native

session = _cuda_native.NativeSession(
    inputs, attributes, preserve_inputs=False,
    kind="mstep", initialize_scratch=True,
)
session.reset()                         # host-to-device reset, outside run timing
timing = session.run_schedule([256, 3])  # one invocation with two native chunks
outputs = session.download()            # independent NumPy output arrays
clocks = session.clocks()               # t/time, current_step, elapsed_steps
metadata = session.metadata()
session.close()
```

`run(nsteps=-1)` executes one chunk, using the prepared step count by default.
Both run methods preserve resident state for subsequent continuation. Source
waveform bounds and int32 clock limits are checked before execution. Source and
DFT clocks advance at chunk boundaries; the original observation origin is held
fixed. `reset()` restores persistent state and clocks from private host snapshots.
Host input arrays are never written by native execution.

Exposed state time uses separately rounded FP32 multiplication and addition,
matching the existing frontend. `scripts/validate_cuda_native_clocks.py` exports
an independent FFI reference and checks it in a separate JAX-free process,
including a nonzero-origin case that distinguishes this expression from an FMA.

```bash
/path/to/jax-venv/bin/python scripts/launch_cuda_component.py \
  --component /path/to/_cuda.so --sha256 FFI_SHA256 \
  scripts/validate_cuda_native_clocks.py --mode reference \
  --fixture /path/to/seeded-fixture.json --expected-component-sha FFI_SHA256 \
  --output /tmp/clocks.json
/path/to/native-venv/bin/python scripts/validate_cuda_native_clocks.py \
  --mode native --fixture /tmp/clocks.json \
  --module /path/to/_cuda_native.so --output /tmp/native-clocks.json
```

The native check includes all 30 state leaves, elapsed/origin clocks, preserving
and owned calls, repeat/reset, and odd continuation schedules.
Add `--method deep4` to validate an enabled experimental D4 module against the
same reference.

`preserve_inputs=True` instead restores the initial persistent state from a
preallocated device snapshot **inside** each timed invocation. A multi-chunk
schedule copies that snapshot once, not once per chunk. This mode requires extra
device storage. Owned mode performs no automatic state copy; reset explicitly
between repeated benchmark trials.

`output_bank` is relative to the **last launch's output tuple**, not necessarily
the original physical allocation. For Mstep, choose fields at output offset zero
or `state_count` (18 with CPML, 6 without) according to that returned bank. Native
clock outputs are exposed separately by `clocks()`. Fixture metadata identifies
the remaining state leaves and unchanged arrays.

Scratch is zeroed once per chunk when `initialize_scratch=True`. Final-field and
phase workspaces are reused without a fill. Metadata reports these policies and
their byte counts: differences from the FFI frontend's initialization must not be
mislabelled as dispatch overhead. The `full_intermediates` module capability
indicates whether the compiled solver can use minimal unused scratch placeholders.

`cuda_ms` includes measured preparation, solver execution, and clock advancement;
`host_ms` includes submission and synchronization. Allocation/upload, explicit
reset, and download are separate operations. Preparation/physics subintervals
are diagnostics, not a substitute for matching complete timing scope and chunks.
Warm up the actual schedule before measuring it.

Methods serialize on each session; waiting for that lock releases the GIL. Runs
and resets also release the GIL during GPU work. Download, clock queries, and
close are synchronous. A run waits for completion before returning. Each session
owns its stream and buffers; this is not a multi-GPU or asynchronous submission
frontend.

## Experimental D4 path

`BEAMZ_CUDA_DEEP_D4` is **OFF by default**. When compiled, `deep_d4` is true and
`session.run_deep4(chunks)` selects the research four-step kernel. Otherwise that
method raises an error. It currently requires Cartesian FP32 fields, uniform
12-cell CPML, scalar H coefficients, lossless packed E coefficients, no PEC,
field extents above 32, and no pre-E sources. Unsupported cases are rejected.
The `deep_admission_kind` module value identifies the generated kernel contract:
zero accepts the generic H/E source and packed-material paths; one additionally
requires only Hz and Ey sources (signature 20) and rank-one byte-packed E codes.
The disabled build reports minus one. Session admission checks these restrictions
before allocating D4 workspace or capturing a graph and raises `ValueError` with
the reason.

The candidate allocates a separate packed-psi bank, sparse observed-cell samples,
and a byte-per-envelope-cell publication mask once. It keeps clocks fixed within
each requested chunk, passes explicit macro and tail offsets, and advances clocks
only at chunk boundaries. Remaining one to three steps use the existing ordinary
source/H/E/DFT sequence. D4 download fields
are normalized to output bank zero. Metadata reports the additional allocations.
Its physics timing interval includes its chunk clock kernel; compare complete
`cuda_ms` across algorithms. This is an experimental correctness/performance
candidate, not an enabled production optimization.
