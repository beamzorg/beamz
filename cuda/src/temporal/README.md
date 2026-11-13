# Shared temporal CUDA candidate

`kernel.cuh` is the common temporal engine for rectangular domains and cubes.
It uses the existing generated Yee cell update and bounded wavefront scheduler:
depth 16 by default, eight overlapping macros and three raw-observation slots.
There are no grid-size, source-mask or contributor-count kernel selectors.

The kernel families describe numerical effects: bulk updates, boundaries,
observations and source-bearing updates. Byte and block material encodings share
the exact material certificate in `scripts/cuda_native_block_material_facts.inc`.
Mixed interfaces retain their original coefficients. Canonical field supports
and CPML strides follow the validated grid dimensions rather than cube constants.

Generic boundary, observation and source effect classes use the same paired-plane storage schedule and
ordered coincident-source loop. Captured counts and bounds may differ by field
component. Single-source FMA and pre-rounded coincident products preserve the
ordinary CUDA injection arithmetic. Planes of any orientation and finite source
volumes use that same computation. Equal starts within a component are verified
before bounds are shared; unrelated/noncoincident sources are not silently
treated as coincident.

Observation tracing packs six exact field-presence bits into each existing
coarse lookup word. An absent field skips its leaf lookup; raw observation IDs,
DFT arithmetic and buffer sizes remain unchanged. The general boundary/effect
class also specializes single active CPML axes from the existing trajectory
flags, without making separate launch cohorts.

Fresh native field, CPML and unused workspaces remain descriptors until upload.
Bitwise positive-zero facts reconstruct initial state and deferred scratch with
device clears, avoiding private domain-sized host snapshots. Nonzero values and
negative zero retain the copy path. A deferred zero scratch buffer gets writable
host backing only if ordinary execution uses it and moving execution retires it.

The current native physics contract remains Cartesian FP32 Yee fields, scalar H
coefficients, lossless packed diagonal E coefficients, CPML12 on all faces and
post-H/post-E injection. Tensor corrections, PEC and pre-E injection are outside
this candidate's existing contract. Depth is fixed on first session preparation;
ownership/preservation is an independent caller choice.

Build the candidate from the previously validated bounded native ABI:

```bash
python scripts/build_cuda_unified_temporal.py \
  --parent .cache/cuda-memory-parity-2026-11-14-1204/lazy-plan-v2 \
  --output <new-output-directory>
```

The builder copies this header, installs exact block material certification,
removes old source/cube selectors from the private wrapper, and records source
hashes and compile/link commands. It reuses unchanged ordinary CUDA objects and
does not replace the installed default backend. The parent artifact is still
required for this experimental native ABI; this is not yet a standalone public
backend build.

For wrapper-only iterations, `--reuse-kernels <completed-candidate>` verifies
identical kernel/planner sources and compile options before reusing their objects.
Use `scripts/benchmark_cuda_four_geometries.py` for matched Mie, cosine, ring and
MMI workloads, and `scripts/summarize_cuda_four_geometries.py` for numerical gates,
timing/memory tables and the interactive comparison.

Validate the built module with `scripts/validate_cuda_unified_temporal.py`.
Use `scripts/compare_cosine_cuda_notebook.py` for complete notebook timing and
`scripts/benchmark_cuda_large_runtime.py --method moving` for the original Mie
fixture. Both call `NativeSession.run_moving` on the same module without workload
selection flags. `tests/unit/test_cuda_unified_temporal.py` guards against adding
benchmark-specific source or dimension selectors to this header.
