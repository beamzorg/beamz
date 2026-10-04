# CUDA cyclic storage: identical physics, faster narrow domains

Follow-up: eligible large RTX3090 CPML12 runs now use [automatic geometry-based layout prediction](cuda-layout-prediction-2026-09-18.md), without calibration. The opt-in controls and measurements below document the original storage study.

An opt-in cyclic storage permutation recovers most of the narrow-domain gap.
On the same physical 1024x256x64 simulation, the isolated layout comparison
measures **6.750 -> 8.380 logical GCUPS** with H/E fusion disabled
in both variants. This is a 24.1% throughput gain. Fields, materials, source,
monitors, FP32 arithmetic and 12-cell CPML are unchanged; every final state
leaf matches bitwise. Conversion costs are included. Consistent >=9 GCUPS,
and a complete CPML two-step schedule faster than the ordinary path, remain
unachieved.

## Implementation and use

`BEAMZ_CUDA_STORAGE_AXES=120` stores canonical (z,y,x) arrays in (y,x,z) order:
1024x256x64 becomes 256x64x1024 internally. `201` uses (x,z,y). `012`, the
unchanged default, keeps canonical storage. The setting is frozen into both
the run configuration and compiled-program cache key. No native ABI or binary
change is required. Returned state always has the canonical physical layout.

The transformation moves the six Yee fields and reorders their vector
components, repacks material codebook IDs, permutes CPML profiles/terms/state
and PEC faces, transposes source slabs and origins, remaps DFT gather indices,
and reorders ragged DFT component arenas on entry and exit. It preserves
source addition order, interpolation-neighbor order, monitor point order and
sample timing. Only cyclic, right-handed permutations are accepted, so no
pseudovector sign convention changes are needed. Modes are not solved again.

The opt-in supports the existing uniform 3D CPML native graphs, including
ordinary stepping, the original interior pair and complete CPML pairing, with
supported sources and DFT monitors present or absent. Unsupported execution
paths raise an explicit error. Broader automatic layout selection is not yet
implemented. FP32 remains default; optional BF16 storage is also tested.

For the measured narrow case:

```bash
BEAMZ_CUDA_STORAGE_AXES=120 BEAMZ_CUDA_CPML_CORE_FUSION=0 ...
```

The second setting disables fusion for the diagnostic below. It is not a
universal recommendation for other domain shapes. Changing only storage axes
retains the existing automatic fusion eligibility.

## Six-case physical-layout study

Each case uses one mode source, two compact mode monitors, three frequencies,
lossless binary or smooth material, exactly 12 CPML cells, and about 16.5–17.0
million logical cells. Each executable advances 256 timesteps. Twelve timing
rounds use a balanced cyclic order and its reverse after three warmups per
variant. All entry/exit field and auxiliary transposes, material repacking and
DFT arena conversions are inside the timed compiled call. Setup, compilation
and validation host copies are outside timing. These are warm executable
rates, not cold-start or public-call throughput.

This initial study used an isolated wrapper around the unchanged native graph;
the production integration was then checked separately below. Ordinary runs
use automatic H/E fusion, and complete CPML pairs use the oriented tile family.
The developed smooth case is preconditioned for 1,024 steps before timing.
All 36 case/variant results have exact complete-state agreement with their
canonical ordinary reference.

| Case | Physical shape (z,y,x) | Ordinary 012 | Ordinary 120 | Ordinary 201 | Best CPML pair (axes) |
|---|---|---:|---:|---:|---:|
| narrow63 | 1024x256x63 | 6.788 | 8.317 | 7.106 | 6.145 (201) |
| narrow64 | 1024x256x64 | 6.676 | 8.303 | 7.157 | 6.183 (201) |
| narrow65 | 1024x256x65 | 6.562 | 8.188 | 7.133 | 6.212 (120) |
| wide | 128x256x512 | 8.379 | 7.905 | 8.385 | 7.131 (012) |
| irregular | 257x193x341 | 8.141 | 7.625 | 7.782 | 6.826 (012) |
| developed_smooth | 1024x256x64 | 6.391 | 7.974 | 6.885 | 5.943 (201) |

Values are logical GCUPS. Small differences between axes, such as the wide
ordinary control, are not evidence of a reliable win. Narrow widths 63/64/65
gain about 23–25% with order 120; the developed smooth case gains about 25%.
A fixed permutation regresses some wider controls, which is why dispatch is
still explicit. The CPML pair remains slower in every case; selecting a good
ordinary layout does not establish the original >=9-GCUPS temporal goal.

## Integrated repeat and fusion ablation

The integrated option, without monkeypatching, reproduced narrow64 at
**6.610 -> 8.211 GCUPS** under automatic fusion. Its six
variants all matched exactly. Between-run timing drift is visible, so the
same-process comparisons are the appropriate basis for speedup claims.

A further balanced run separated layout from fusion and tested padding:

| Variant | GCUPS |
|---|---:|
| canonical_queue | 6.750 |
| canonical_fused | 5.574 |
| permuted_queue | 8.380 |
| permuted_fused | 8.313 |
| permuted_fused_pad32 | 8.105 |
| permuted_fused_pad64 | 8.104 |

All six variants matched the complete reference state bitwise. Layout alone
accounts for the recovery; enabling H/E fusion does not supply an additional
measured gain in this case. Padding after permutation still loses here.
The permutation changes memory access patterns and core/shell thread mapping.
It does not change CPML volume, and counters are unavailable to assign the
remaining cost to specific hardware stalls.

## Validation and limits

- 49 integrated hardware cases passed across two cyclic orders, FP32/BF16,
  binary/smooth material, mode/field monitors, source-only and source-free
  graphs, ordinary/interior-pair/complete-CPML-pair schedules, seeded fields
  and auxiliary state, 33-step runs and two-step continuation. Ragged field
  monitors vary aperture, frequency count and component selection; initial
  DFT values are seeded. One case exercises public `Simulation.advance` and
  a one-step tail with the CPML-pair option enabled.
- 12 earlier prototype seeded cases also passed exact state comparisons.
- 63 distinct CPU backend/runtime tests passed, including cache separation,
  configuration snapshotting, invalid-order rejection and unsupported-path
  rejection. Ruff and `git diff --check` passed.
- The native kernels/ABI were unchanged. A new full-release or sanitizer sweep
  was not run for this Python integration; prior CUDA validation remains scoped
  to its reported workloads.
- Maximum GPU temperature during the large studies: 79.0 C;
  minimum free GPU memory: 14842.0 MiB. Power settings were unchanged.
- Nothing was pushed. The branch changes remain local and uncommitted.

The subsequent [y-face CPML study](cuda-cpml-yface-study-2026-09-18.md)
profiles that boundary cost and retains a 16x15x16/384-thread y-face tile with
bounded register use. It improves the experimental complete pair by 11–13%
on flat cases, reaching about 7.0 GCUPS. The ordinary-layout results above
remain faster; the complete pair remains opt-in.

Reproduce integrated comparisons with
`scripts/benchmark_cuda_storage.py --shape 1024 256 64 --output result.json`;
add `--study fusion` for the ablation. Raw samples, state digests, native hashes
and telemetry are in the [six-case study](rtx3090-2026-09-18-cyclic-storage/),
[integrated repeat](rtx3090-2026-09-18-cyclic-storage-integrated/), and
[fusion/padding ablation](rtx3090-2026-09-18-cyclic-storage-fusion/).
