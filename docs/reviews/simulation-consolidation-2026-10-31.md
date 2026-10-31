# Simulation consolidation, 2026-10-31

This pass shares existing simulation mechanics without changing the source/monitor
compiler contracts or adding another execution planner.

- `compile_program` uses one ordered CUDA capability ledger for automatic fallback
  and explicit request errors. Hardware availability and schedule-specific proofs
  retain their existing owners.
- Ordinary, progress and automatically terminated runs use `_execute_chunk` for
  restoration, placement, execution, synchronization and cropping. Timed execution
  still compiles before starting its timer; untimed execution remains lazy.
  `_run_results` owns shared final material/source/acquisition detachment.
- `DftBufferLayout` interprets existing compiled component-major slices for
  accumulation, detached results, differentiable acquisitions and spectral adjoint
  inputs. Empty and populated monitor arenas now use the same initializer.
- Uniform and metric 3D CPML kernels share derivative/material assembly and static
  phase adapters. Metric differences reuse `lattice.metric_adjacent_difference`.
  Uniform division order and incremental source-scale specialization are retained.
- Native CUDA phase calls and graph/capability paths share the six-operand material
  ABI order. `NativeSchedulePlan`, native VJPs and `forward_step` remain authoritative.

Absolute integer clocks, coefficient/source AD arguments, donation ownership,
packed CPML storage, lazy material preparation, local sharding and cache identities
retain their existing contracts. No CUDA variant or physics feature was retired.

## Physical Python lines

Counted with `len(path.read_text().splitlines())` over every `*.py` under each root,
including shared code added by this pass, relative to baseline `433d59f5`:

| Scope | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Entire `beamz` package | 52,291 | 52,134 | 157 |
| `beamz/simulation` | 18,222 | 18,065 | 157 |

This is a modest reduction toward the package goal; the review's multi-thousand-line
estimate is not realized by this conservative pass. Documentation is excluded from
these counts. No implementation was moved outside the package.

## Validation

- 318 CPU tests passed: kernel updates/boundaries/CPML, backend capability/cache
  policies, CUDA runtime/autodiff/schedule/storage contracts, metamorphic execution,
  termination, clocks, result archives, monitor clocks, long-run DFT phase,
  rectilinear tensor identities and directional gradients.
- 35 final CPU tests passed: engine equivalence, metamorphic execution, result
  archives and dense preparation reuse, including the unified empty arena path.
- Twelve JIT-compiled direct baseline comparisons passed: H/E updates for uniform,
  axis-uniform and graded 3D grids with/without CPML, comparing both primal arrays
  and dynamic-material JVPs (`rtol=2e-6`, `atol=1e-10`).
- Ruff and `git diff --check` passed.

The current Rust raster extension was built outside the checkout and supplied via
an external test launcher/sitecustomize, including subprocess tests. An initial
attempt used an incompatible older extension; its failures were environmental.
The native sharded CPU suite initially failed because subprocesses could not load
that extension; after fixing the external harness, eight cases passed at handoff,
with the last case still running. These CPU FFI tests do not establish GPU parity.
CUDA initialization fails on this host (`CUDA_ERROR_UNKNOWN`), so GPU execution,
performance, allocator capacity and hardware sharding remain unverified.

## Deferred integration

A full acquisition plan must originate in the Devices compiler and include
sampling geometry, recorder ownership, normalization and native packing metadata.
This pass interprets the existing layout rather than changing that cross-package
ABI. CUDA storage transforms still consume native packed monitor counts.

A complete capability/configuration value spanning availability, mathematical
support, storage, schedule and tuning needs coordinated cache-key and diagnostic
changes. The ordered problem checks here do not replace algorithm-specific proofs.

Further curl sharing should start with shared derivative descriptors in lattice
and boundary compilation, with measured GPU compile/runtime/memory evidence.
Local halo exchange and CUDA variants should stay specialized. Sharing eager and
region material preparation requires preserving FP32 rounding and bounded reads;
no eager expansion was introduced here. General and topology differentiation still
have different material/source binding contracts; native reverse-mode and spectral
adjoint algorithms remain distinct.
