# Selectable storage order for sharded CUDA

Uniform FP32 x-sharded CPML phases currently rotate local arrays from
`(z, y, x)` to `(x, z, y)` around every native phase. This preserves physical
ownership, source/monitor coordinates and boundaries, but transposes and kernel
layout have a workload-dependent cost. Keep that default and allow the existing
`BEAMZ_CUDA_STORAGE_AXES` compilation input to select the sharded order:

```sh
# Original physical storage, bypassing the per-phase rotation:
BEAMZ_CUDA_STORAGE_AXES=012
# Existing normal-first layout:
BEAMZ_CUDA_STORAGE_AXES=201
```

An unset variable retains `201` for eligible sharded execution and `012` for
single-device execution. The order is recorded in the immutable run config
and compilation cache key. The existing uniform FP32 CPML eligibility guard
remains; this does not add storage transforms to nonuniform or tensor cases.
`120` is also an existing valid right-handed cyclic order, but has no timing
measurement in this study. Benchmark a layout on the intended workload before
selecting it: a different aspect ratio, shard axis or CPML fraction can favor
rotation, bypassing it, or show little difference.

## Eight-H100 directional-coupler measurement

The four-port silicon directional coupler has 500 nm guide width and 220 nm
core thickness. Si and silica indices are 3.47 and 1.44, nondispersive. The full
domain is **46 × 10.5 × 4.22 µm**, including nominal 1 µm CPML per face.
Uniform **20 nm** cells give **2,300 × 525 × 211 = 254,782,500 cells** including
CPML and excluding device padding. Excitation is TE0, with four modal monitors,
21 wavelengths from 1500–1600 nm and a 3 ps configured ceiling.

Measurements used Beamz `ac76659f62747ff8e72a5657132cb393d1b5f12f`, CUDA
component 0.21.0 / ABI 21, JAX 0.9.0, FP32, and H100 80 GB HBM3 GPUs with
NVLink. The relevant native/sharding source is unchanged on the PR base
`3847a523f5a908d3fa88e77350b22d885735415f`.

| Configuration | Median warm GCUPS |
|---|---:|
| 1 H100, original single-device CUDA | 19.65 |
| 8 H100s, explicit arithmetic, rotated | 57.78 |
| 8 H100s, explicit arithmetic, bypassed | **89.79** |

The paired eight-GPU gain is **55.4%**. Both use the same deterministic nonzero
initial fields: component `c` is initialized with
`((13*x + 7*y + 3*z + 11*c) % 31 - 15)`, scaled by `1e-4` for E or `1e-6`
for H; padding is zero. One 256-step warmup precedes five continued synchronized
256-step samples, 1536 steps total. This is a synthetic initial condition on
the physical coupler, not a completed optical solve. GCUPS covers warm
device-complete dispatch; preparation, placement, JIT, warmup, stopping checks
and extraction are excluded. The paired sample ranges are 57.77–57.80 and
89.61–89.83 GCUPS.

The single-device result is an earlier unpatched zero-initial-field pilot on
the same workload and host, before the source pulse, using the same timing
protocol. It was not rerun with the patch. It provides context rather than a
controlled patched scaling measurement. Its native graph path remains
separate from the sharded phase wrapper.

## Numerical evidence and limits

Changing storage layout changes which cells enter the interior versus shell
kernels. Their implicit multiply/FMA contraction previously differed. Use the
shared explicit derivative, Yee field, CPML recurrence and CPML correction
primitives in both uniform paths; retain the host arithmetic fallback.

A 49 × 97 × 521 heterogeneous guide with active Gaussian excitation, six-cell
CPML, three DFT frequencies and eight x-shards compared **every physical E/H
cell and complete DFT** at steps 1, 8, 64, 256 and 1536. Corrected rotated and
bypassed layouts had identical array SHA256 digests at every checkpoint.
All state leaves, including CPML, were finite. Owner-local CPML buffers were
not compared directly across backends. On the large coupler, 4096 probes per
field and global maxima/norms matched exactly; complete DFT relative error was
at most **7.008e-7**, below the unchanged 5e-5 gate.

Actual zero-initial-field 75 nm coupler solves on JAX and corrected CUDA both
converged at 10752 steps. All four complex modal spectra agreed within
**3.791e-5** relative to their reference spectrum's peak magnitude. This is a
coarse physical control, not final 20 nm mesh convergence.

The separate long synthetic-field JAX comparison still fails 5e-5 at 1536
steps (**8.617e-5**); the original CUDA implementation also failed this control.
The patch changes original rounding, and the large corrected cases fail the
unpatched saved field-probe comparison (**1.671e-4**). These failures remain
limitations; layout equality does not establish final optical accuracy.
