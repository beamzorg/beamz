# FP32 memory-capacity follow-up on eight H100s

Draft PR #288 is stacked on #287 (`bench/h100-backend-comparison`). This report
separates capacity evidence from performance evidence. The new RunPod budget is
$40 total, including a replacement allocation. **The 15B stepping target is not
achieved:** preparation fits, but compilation/loading fails. The largest new
stepped case is 5.75B. Both owned allocations are deleted and verified absent.

## Implementation

- NumPy and single-device CPU JAX arrays are sliced on the host and transferred
  directly to their destination shards. Existing distributed arrays retain the
  device-to-device reshard path. Initial fields are padded per destination shard,
  avoiding six full-domain padded host copies.
- CUDA runtime coefficients omit original permittivity/conductivity/magnetic-loss
  grids after the required update coefficients are computed. Tensor inverse
  coefficients remain present. JAX's material representation is unchanged.
- `BEAMZ_TRACE_PLACEMENT` optionally records each leaf's source, shape, target,
  and before/after allocator statistics. Tracing synchronizes placement and is
  diagnostic, not an uninstrumented setup-time benchmark.
- `benchmark_modal_stepping.py --donate-state` passes each output into the next
  invocation and records compiler aliases and memory after every continuation.
  Its original retained-input protocol remains the default. Donated inputs must
  not be reused. `benchmark_h100_memory.py` runs isolated capacity cases under an
  explicit deadline; infrastructure teardown is independently guarded.

After the GPU attempts, a separate correctness fix preserves wide global 3D
monitor/recorder indices on the host and decodes them into per-axis coordinates
before JAX sampling. Native unsharded monitor packing rejects unsupported wide
indices. This avoids global int32 wrap without changing FP32 fields or enabling
JAX x64. The capacity runner now also saves preparation statistics and compact
lowered IR before compilation, plus allocator statistics on failure. These later
changes are CPU-validated; they were not deployed in the recorded H100 attempts.

Native CUDA kernels and the native wheel are unchanged (SHA256
`6753641e1a682751ad00c10abda53bd4cd33fd3569ccf048baea018bb94be00d`).
Fields and CPML state remain FP32. These changes do not implement shard-local
geometry/material rasterization: global host construction is still a capacity
and setup-time bottleneck.

## Measured results

All cases use a mode source, two 101-frequency mode monitors, and 12-cell CPML.
Resolution is 80 nm. The target physical extent is 800 × 100 × 96 µm
in `(z, y, x)` order, with a binary dielectric straight waveguide and a large
cladding region. This is a capacity stress case with realistic solver features,
not evidence that every arbitrary material distribution fits.
Shapes are `(z, y, x)`; cell counts exclude staggered padding. Allocator peaks
are live JAX allocations, not reserved pools or total NVML device residency.

| Case | Physical cells | Partition | Protocol | Maximum allocation/GPU | Status |
| --- | ---: | --- | --- | ---: | --- |
| Retained control `(512,512,4096)` | 1,073,741,824 | x | 256-step warmup + five independent 256-step calls | 10.19 GiB | finite |
| Previous failure probe `(896,896,7168)` | 5,754,585,088 | x | six donating 256-step calls | 41.73 GiB | finite, 1,536 steps |
| Target `(10000,1250,1200)` | 15,000,000,000 | z | six donating 64-step calls planned | 64.72 GiB after preparation | compilation/loading OOM; no stepping |

The 5.75B case formerly failed during coefficient placement, before compilation.
It now prepares successfully with **25.04 GiB live on each GPU**, then reaches
**41.73 GiB peak on each GPU** during stepping. Live storage returns to 25.04 GiB
after every call, with no growth across five continuations. Both monitor weights
reach 1,536 and final state is finite. Setup took 700.88 s; compilation 92.95 s;
worker wall time was 973.82 s. These timings include placement tracing.

Compiler memory analysis for that x-partitioned case, per device:

| Category | Bytes |
| --- | ---: |
| Arguments | 26,884,853,908 |
| Outputs | 18,230,479,708 |
| Aliased output storage | 18,230,479,468 |
| Temporaries | 17,921,857,096 |

Donation reuses nearly all output-state storage. The existing x-axis execution
layout still requires a large temporary bank; the 15B probe therefore uses the
existing contiguous z partition. Its geometry differs from the x sweep, so it
is not a same-geometry throughput comparison.

The retained 1.074B control peaks at 10.19 GiB on every GPU, versus 13.84 GiB on
GPU 0 in the prior study. Its timings drift from 2.76 to 4.16 s per sample. The
5.75B median is 44.73 GCUPS. **Neither is accepted as a throughput comparison:**
GPU 3 on the first allocation reaches 87 C and 345 MHz while seven peers run at
39–44 C and 1,980 MHz. The allocation was deleted after evidence capture; no
suitable eight-H100 allocation was available outside the same Norway location.
GPU UUIDs confirm the replacement landed on the same eight physical GPUs. It
is used only for capacity validation, with short continuations; its timings are
also excluded from performance conclusions.

## Why 15B still fails

Preparation completes in **1,738.69 s (28.98 minutes)**. Every GPU has exactly
69,493,824,512 live bytes (**64.72 GiB**), with the same peak. The allocator pool
is 80,766,617,600 bytes (**75.22 GiB**) per device; its limit is 80,766,617,815
bytes. Thus, live prepared arrays fit, but there is only about 10.50 GiB per
allocator for further work. Sampled NVML residency reaches about 76.64 GiB on
GPU 0, including allocator/runtime overhead.

During `lower(...).compile()`, GPU 0's BFC allocator reports an additional
**7.02 GiB** request and compilation/loading raises `RESOURCE_EXHAUSTED`. The
worker exits with code 1 after 1,980.78 s total. No executable memory analysis,
warmup, continuation, or finite final state exists for this attempt. It must not
be reported as a successful 15B simulation or assigned a GCUPS result.

The evidence establishes that direct placement fixed the former preparation
failure, but not the executable's full memory demand. It does **not** identify
which compiler allocation made the request, or distinguish additional live
buffers from contiguous-space fragmentation. A compact lowered-IR and failure
allocator checkpoint were added afterwards to resolve that on a future run.
Small representative lowering confirms separable CPML profiles; it does not
attribute the large GPU failure.

Peak worker RSS sampled from `/proc` is **1,443,477,123,072 bytes (1.44 TB)**;
the container peaks at **1,448,584,007,680 bytes**. There is no host OOM event,
swap use, or sustained memory pressure. Global CPU construction remains costly.
A fresh setup alone would cost approximately $13.48 at this hourly rate, more
than the $9.38 left under the conservative budget estimate, before compilation
or stepping. No additional allocation was started.

The next capacity gate is to explain and remove that executable allocation peak
(or reduce persistent coefficients enough to accommodate it), then repeat all
six continuations. Material palettes could save the three dense electric update
grids for suitable geometries, but require exact staggered/interface/loss
semantics and are not implemented here. Host-local material construction and
avoiding global initial-field copies should reduce the 29-minute setup as well.

## Validation and limits

- 53 unique targeted CPU tests pass across overlapping runs. They cover direct
  placement, native CPU-FFI CUDA orchestration, tensor/nonuniform/CPML/monitor
  paths, JAX local sharding/gradients, and donating modal continuation.
- After adding per-shard field padding, 14 relevant tests pass again. The
  placement test rejects global-array `device_put` and global field padding.
- Final repository Makefile-scope Ruff checks pass; 346 files are formatted.
- The propagated 80 nm gate compares full state and mode artifacts at 2,048
  steps: one-GPU JAX versus eight-GPU CUDA passes for x and z partitions. After
  the local-padding change, the replacement allocation repeats and passes the
  z gate with the original tolerance: `rtol=3e-5`,
  `atol=max(1e-12, 2e-6 * max(abs(reference array)))`.
- The later monitor-index correction has a synthetic 15B plane-plan and jitted
  analytic-field sampling test with JAX x64 disabled. All **49 tests** in its
  final regression run pass, covering monitors, recorders, native CPU FFI, and
  pure-JAX local sharding. These overlap the earlier test runs and are not added
  to their count. Results are in `large-index-tests-final.log`. It has not been revalidated on
  H100. The recorded 5.75B run predates this fix; its monitor weights are capacity
  evidence, not validated large-domain optical spectra.
- A short large-domain capacity run is not a converged optical simulation. Its
  distant output monitor need not have received the pulse yet; nonzero monitor
  weights verify accumulation, not transmitted optical power. Propagated parity
  is tested separately on the smaller domain.
- The pre-existing 64 nm raw-DFT discrepancy is unresolved. This work neither
  loosens that gate nor claims a full-suite audit. PR #287's unrelated audit
  findings remain recorded in its report.

## Scaling beyond the target

Memory is not the only upper bound. The current native sharded implementation
rejects local buffers larger than `INT_MAX` elements (`ValidBuffer` in
`cuda/src/sharded_cell.h`) and uses 32-bit local offsets. The 15B geometry stays
below that limit; padding included, its largest local field has 1,879,566,201
elements. Eight times `INT_MAX` is about 17.18B, and padding/halos lower the
physical-cell ceiling. A later 17–20B investigation must audit those guards and
offsets as well as material storage and HBM; spare memory alone is insufficient.

## Reproduction

After a propagated parity gate passes on the intended build, run from the
repository root with the CUDA extension and eight H100s available:

```sh
python scripts/benchmark_h100_memory.py --target-only --timesteps 64 \
  --output /path/to/new-evidence \
  --deadline YYYY-MM-DDTHH:MM:SS+00:00
```

Use a real future deadline and an independent infrastructure deletion guard.
The runner deadline stops the worker; it does not delete a RunPod allocation.
Keep the original retained-state protocol when comparing prior throughput.

## Evidence and cost

Compact machine-readable evidence is in [h100-memory-20260927](h100-memory-20260927/).
Source manifests distinguish local commits from the remote archive's temporary
Git snapshot. The GPU attempts used memory changes through `b60ee28`; the 5.75B
case predates local field-padding changes. Subsequent monitor-index and compiler
checkpoint changes have separate commits and local validation.

| Owned allocation | Deletion verified UTC | SHA-verified remote files | Conservative compute estimate |
| --- | --- | ---: | ---: |
| `a34xuj0x6ofmw5` | 2026-09-26 23:25:42 | 36 | $12.1967 |
| `oj4cf9kjtwxokp` | 2026-09-27 00:09:38 | 27 | $18.4210 |

Both delete calls returned 204 and subsequent GET calls returned 404. Both
independent teardown guards were cancelled only after deletion verification.
Total conservative compute is **$30.6177**, using the entire creation-to-GET404
lifetimes at $27.92/hour, with ample headroom for the 80 GB ephemeral disks under
$40. Billing is incomplete at the final snapshot: first pod $8.5199 including
$0.0037 disk; second pod no records yet. Empty/partial billing is not treated as
free usage or a final invoice.

The full raw evidence is preserved locally and described by `archive.json` and
the two remote SHA manifests; compact results and test logs are committed. The
local archive is `benchmarks/results/h100-memory-20260927/h100-memory-20260927.tar.gz`
(6,446,206 bytes), SHA256
`ea381b8917607595c872e4d1ecfa2d50abda4adbab3ecbb024ed047d5c9c275c`.
Human review is pending. OpenAI Codex authored the implementation, validation,
and report; the PR remains a draft.
