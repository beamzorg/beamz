# Direct frozen production core validation

On 2026-11-11 at 13:22:58 UTC, a standalone CUDA harness passed an independent
two-step scalar Yee comparison for the frozen v11 packed core. Both z schedules
(`segment=0` and `segment=2`) produced 9,216 owned FP32 values bitwise correctly,
with no changes to inputs, output guards, or unowned output cells. Compute
Sanitizer memcheck and synccheck reported zero errors; racecheck reported zero
hazards, errors, and warnings for the default segment. The five runs took 1.64 s.

## What actually ran

[`probe_cuda_frozen_mstep.py`](../../../scripts/probe_cuda_frozen_mstep.py)
includes the immutable candidate's `source/src/update.cu` directly. It launches
the actual `MstepPackedColumnCore<32,16,1,false>` template; it does not copy or
reimplement that kernel. Launches bypass JAX/XLA, `graph.cu`, and native CUDA
graph capture/instantiation. Cache-disabled Python launches still reach native
graph construction, so these probes isolate a different execution path.

The fixture uses material extents z/y/x = 11/40/72 and the six real staggered
field supports. It supplies nonzero input fields, heterogeneous eight-entry
electric coefficient tables and packed byte codes, scalar magnetic
coefficients, z bounds [3,7), and two partially owned tiles with quadrant masks
8 and 5. Halo reads include clamped discarded tile regions. The segmented run
adds bounded z work and uniformly inactive z CTAs. A handwritten scalar CPU
oracle advances both full steps with explicit FP32 operations and `std::fma`;
comparison covers every owned output. Sentinel words surround each allocation
and occupy every initially unwritten output. Inputs are checked unchanged.

This validates this packed-core fixture only. It does not validate edge kernels,
CPML, sources, DFT, runtime graph behavior, full-bank alias/continuation rules,
or all supported shapes and coefficient layouts. CPU dependency/taint proofs
and complete-solve numerical gates provide separate evidence for those paths.
Sanitizer timings are not performance measurements.

## Provenance

- Frozen candidate: `.cache/cuda-autoresearch-2026-11-11/builds/full-typed-traced-v11`.
- Frozen source tree SHA-256:
  `9dbc587cb4b7757f33645afc28c7b89c22ed1f0b18cd30909adc0994fa41276e`.
- Original production component SHA-256:
  `4cdde58fc618dad17b0c7f04ec6050e8ec356eacdf51d153c4b98ba714500d1a`.
- Direct harness executable SHA-256:
  `ba421c7ad847491d2fd8ff1451d50d3b81b45be51b4568721f9ee56322a2f419`.
- Harness source SHA-256:
  `3efc901b98cc985d6b6de1379cc38c0a331e61a8d422fbee913d2bb66d237102`.

The standalone executable uses the same frozen kernel source and effective
mstep BOOL options, recovered from its frozen `CMakeCache.txt`; it is a
different binary from the production component. The six enabled options are
FULL_INTERMEDIATES, LAYOUT_SPECIALIZE, EXACT_INDEX, SKIP_FINAL_BARRIER,
FULL_EDGE_TYPED, and TRACED_CELL. Other recorded mstep options are disabled.
Compilation uses CUDA 13.3, C++17, `-O3 -DNDEBUG -arch=sm_86 -lineinfo`,
`--expt-relaxed-constexpr`, and `--relocatable-device-code=true`, matching native
separable compilation. Host `-ffp-contract=off` constrains the independent CPU
oracle; device arithmetic retains the production explicit rounding/FMA rules.

The raw archive is
`.cache/cuda-autoresearch-2026-11-11/direct-frozen-v11/rdc/`:
`manifest.json`, generated source, executable, compiler log,
`validation.json`, and individual functional/sanitizer logs. The manifest
records the full compile command, effective options, CMake-cache hash, and
exact mangled kernel filter extracted from the compiled device binary.
Earlier non-RDC compile artifacts remain in the parent directory and were not
used for these GPU results. No production component was replaced.

## Reproduce

From the repository root, choose a fresh output directory:

```sh
python scripts/probe_cuda_frozen_mstep.py \
  --build .cache/cuda-autoresearch-2026-11-11/builds/full-typed-traced-v11 \
  --output-dir /tmp/beamz-direct-frozen-v11-repro \
  --nvcc /home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/nvcc \
  --compile
/tmp/beamz-direct-frozen-v11-repro/direct_mstep 0
/tmp/beamz-direct-frozen-v11-repro/direct_mstep 2
```

The archived `run_validation.py` replays all five runs against the archived
executable and verifies its SHA first. It enforces 30 s per sanitizer, a 115 s
total deadline, and stops on the first error or timeout while terminating the
entire process group. Run it only during an exclusive GPU slot:

```sh
python .cache/cuda-autoresearch-2026-11-11/direct-frozen-v11/rdc/run_validation.py
```

Each sanitizer command uses `--launch-count 1 --error-exitcode 99` and the
manifest's exact mangled-name filter. A rebuild can change anonymous-namespace
symbol names; extract its target symbol again rather than reusing this filter.
