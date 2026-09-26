# FP32 memory-capacity follow-up on eight H100s

Draft PR #288 is stacked on #287 (`bench/h100-backend-comparison`). This report
separates capacity evidence from performance evidence. The new RunPod budget is
$40 total, including a replacement allocation.

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
| Target `(10000,1250,1200)` | 15,000,000,000 | z | six donating 64-step calls | pending | running |

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
The replacement is used only for capacity validation, with short continuations.

## Validation and limits

- 53 unique targeted CPU tests pass across overlapping runs. They cover direct
  placement, native CPU-FFI CUDA orchestration, tensor/nonuniform/CPML/monitor
  paths, JAX local sharding/gradients, and donating modal continuation.
- After adding per-shard field padding, 14 relevant tests pass again. The
  placement test rejects global-array `device_put` and global field padding.
- Final repository Makefile-scope Ruff checks pass; 345 files are formatted.
- The propagated 80 nm gate compares full state and mode artifacts at 2,048
  steps: one-GPU JAX versus eight-GPU CUDA passes for x and z partitions. After
  the local-padding change, the replacement allocation repeats and passes the
  z gate with the original tolerance: `rtol=3e-5`,
  `atol=max(1e-12, 2e-6 * max(abs(reference array)))`.
- A short large-domain capacity run is not a converged optical simulation. Its
  distant output monitor need not have received the pulse yet; nonzero monitor
  weights verify accumulation, not transmitted optical power. Propagated parity
  is tested separately on the smaller domain.
- The pre-existing 64 nm raw-DFT discrepancy is unresolved. This work neither
  loosens that gate nor claims a full-suite audit. PR #287's unrelated audit
  findings remain recorded in its report.

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
Git snapshot. First allocation `a34xuj0x6ofmw5` was deleted and GET returned 404;
36 downloaded remote files were SHA256-verified. Its conservative compute cost
is $12.20 at $27.92/hour. Replacement `oj4cf9kjtwxokp` has a $25.50 compute limit,
leaving headroom for ephemeral disk under the combined $40 cap.

The 15B result, second cleanup record, and final total will be filled in after
measurement. Human review is pending. OpenAI Codex authored the implementation,
validation, and report; the PR remains a draft.
