# CPML storage precision: RTX3090 JAX and CUDA comparison

The comparison changes only CPML auxiliary storage: fields, coefficients and
recurrence/correction arithmetic stay FP32. All simulations are lossless, have
exactly 12 CPML cells, a mode source and two compact mode monitors at three
frequencies. No precision defaults have been changed.

## Throughput

Each large domain contains about 16.8–16.9 million logical cells. Eight balanced
rounds rotate all four variants through every timing position twice; each
variant has two warmups. Measurements synchronize the complete compiled scan,
include source and monitor work, and exclude setup, compilation and modal
postprocessing. CUDA uses the established schedule, not the slower complete
CPML pair. Physical padding is disabled and inactive storage cells are not
counted. No GPU power settings were changed.

| Domain (z,y,x) | Material | JAX FP32 | JAX BF16 psi | CUDA FP32 | CUDA BF16 psi |
|---|---|---:|---:|---:|---:|
| 128x256x512 | binary | 3.613 | 3.384 | 8.481 | 8.889 |
| 1024x256x64 | binary | 3.915 | 3.706 | 6.852 | 7.601 |
| 257x193x341 | smooth | 3.902 | 3.539 | 8.238 | 8.473 |

Units are logical GCUPS. BF16 improves CUDA by 4.8%, 10.9% and 2.9%, respectively.
JAX slows by 6.3%, 5.3% and 9.3%. These three cases support roughly 7.6–8.9 GCUPS
for this CUDA workload, not a guarantee for arbitrary sources/monitor sizes,
materials, domains or run lengths. They do not support a BF16 performance
default for JAX. We have not measured hardware counters to attribute the JAX
slowdown to a particular instruction or memory bottleneck.

On the wide domain, canonical state storage falls from 471.0 MiB to 428.8 MiB.
Only auxiliary state is halved; fields and native execution scratch are not.
These figures are canonical state sizes, not total process GPU memory.

## Accuracy and implementation

JAX previously cast derivatives and CPML coefficients to the auxiliary array's
dtype. Simply making that array BF16 would also reduce arithmetic precision.
The comparison fixes `correct_cpml_term` to promote arithmetic to at least FP32,
use the unrounded recurrence for field correction, and round only the returned
auxiliary state. The four new recurrence checks cover FP32/BF16 storage and 1/31
steps against an independently evaluated recurrence. All 34 selected lossless
kernel tests passed on CPU, including these checks.

BF16 is a lossy representation of the auxiliary state. Identical numerical
accuracy must not be assumed from identical FP32 field dtypes. In particular,
a short throughput run can finish before an absorber's return reaches a mode
monitor. The 256-step large-case snapshots are recorded but are insufficient
on their own to justify a precision default. Separate 4096-step runs and a
12-cell, source-free packet-reflection comparison are used below.

The CUDA build also includes explicit derivative rounding and field/CPML FMA
order. Its ordinary and temporal schedules agreed exactly in the two large
BF16 schedule-comparison cases. That is schedule parity at fixed precision;
it is not BF16-versus-FP32 accuracy equivalence.

Raw throughput samples, checksums, full-state error summaries, modal-power
errors and telemetry: `rtx3090-2026-09-18-backend-precision/`.
Reproducible comparison: `scripts/benchmark_cpml_precision.py`.

### Long-run accuracy

Two lossless waveguides were advanced for 4096 steps in 256-step continuations,
with the same pulse, exact 12-cell CPML and requested monitor sampling in every
variant. Errors below compare BF16 against FP32 **within the same backend**.
Field error is the maximum component-peak-normalized difference; power error
is normalized to each reference mode's peak over the monitored frequencies,
then maximized across the two monitors. DFT error is relative L2 over the full
accumulation vector. These are not per-frequency worst relative errors at
near-zero reference power.

| Case | Backend | DFT L2 error | Field peak error | Forward-power error | Backward-power error |
|---|---|---:|---:|---:|---:|
| 128x96x256, binary | JAX | 0.01261% | 0.5666% | 0.00799% | 0.5121% |
| 128x96x256, binary | CUDA | 0.01124% | 0.4322% | 0.00413% | 0.4168% |
| 129x95x257, smooth | JAX | 0.01694% | 0.5057% | 0.01213% | 2.2143% |
| 129x95x257, smooth | CUDA | 0.01520% | 0.4389% | 0.01239% | 1.0259% |

Forward transmission is close in these examples. Reflected/modal backward
signals are more sensitive. Backward mode power includes source and grid
errors and is not an isolated absorber reflection coefficient. All recorded
fields, auxiliary state and modal values were finite. These runs do not cover
high-Q cavities, oblique/grazing incidence, or arbitrary physical run lengths.

### Source-free packet return

A separate JAX 2D normal-incidence packet experiment follows the existing
analytical validation's time-separated characteristic-energy measurement,
using exactly 12 physical CPML cells. There are no sources or monitors during
propagation. It measures return energy in the interior relative to the
initial incident packet energy; it is not a throughput benchmark.

| Background index | Points per medium wavelength | FP32 return | BF16 return | Increase |
|---|---:|---:|---:|---:|
| 1.0 | 10 | -69.49 dB | -64.13 dB | 5.37 dB |
| 1.0 | 20 | -46.42 dB | -46.34 dB | 0.07 dB |
| 1.5 | 15 | -64.80 dB | -59.81 dB | 5.00 dB |

The BF16 results remain below -46 dB in these cases, but the two best-absorbed
FP32 cases return roughly three times more energy with BF16. Thus “same
numerical accuracy” is not established, even though transmission errors are
small. Raw data: `rtx3090-2026-09-18-backend-precision/reflection.json`.
Reproduce with `scripts/benchmark_cpml_reflection_precision.py`.

## Default recommendation

Keep FP32 as the shared default. BF16 auxiliary storage is a useful opt-in CUDA
performance choice for workloads whose accuracy requirements tolerate the
measured tradeoff. The existing override is
`BEAMZ_CUDA_CPML_PSI_PRECISION=bf16`; `fp32` explicitly selects full auxiliary
precision. JAX's tested BF16 path is slower and currently uses explicit state
conversion in the experiment; the CUDA environment override does not enable
BF16 for public JAX runs. No universal 7–9 GCUPS or equivalent-accuracy claim
follows from these samples, and no precision defaults were changed.

Across all five GPU comparison cases, telemetry stayed at or below 73 C and
at least 14,976 MiB remained free. Nothing was pushed.

Final targeted hardware validation passed all 30 seeded 33-step cases, covering
three irregular shapes/padding choices, both precisions and all five
spatial/temporal schedules, including continuation. These require exact CUDA
schedule parity and preserve the independent JAX comparison tolerances.
The user confirmed retaining FP32 defaults for both backends after reviewing
the precision comparison.
