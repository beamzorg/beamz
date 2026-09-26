# H100 backend comparison and multi-GPU scaling

Historical measurement snapshot: the worktree subsequently removed the explicit
Hopper backend. Archived sources preserve this experiment; current benchmark
runners execute only JAX and streamed CUDA. See
[the follow-up analysis](H100_PERFORMANCE_ANALYSIS.md) for the removal and targets.

Run date: 2026-09-25. Source: `origin/main` at `2f41186`, plus the two-condition
Hopper coefficient preparation fix described below. The measurements do not
represent unmodified main.

The 35-case pilot favors `cuda_streamed` over `cuda_hopper` by 5.3–7.5× on one
H100. The current Hopper implementation should not replace streamed execution.
Adding GPUs does not improve streamed throughput on the medium domains or the
512³ follow-up. Changing partition axis does not close that gap. JAX is faster
than either CUDA backend on the physical crossing. All 53 timing configurations
completed without failures. Full crossing outputs are finite, with the measured
numerical differences below; no universal no-regression claim is made.

## Hardware and scope

RunPod had no available 8-GPU H100 SXM pod when this run was provisioned. The
initial pilot therefore uses one node with four H100 SXM 80 GB GPUs in CA-MTL-1,
with all pairs reporting NV18 connectivity. Each trial exposes exactly 1, 2,
or 4 GPUs. There is no measured 8-GPU result and no extrapolated 8-GPU claim.
`cuda_hopper` is a single-GPU backend; only JAX and `cuda_streamed` are tested
with multiple GPUs.

The approved cap is $100. The node's compute rate is $13.96/hour. It was created
at 10:02:38 UTC and terminated at 11:45:34 UTC after verified downloads, giving
an elapsed-time compute estimate of **$23.95**. Termination returned HTTP 204;
the subsequent pod lookup returned 404. The existing unrelated stopped pod was
not modified. `resource.json` records the resource lifecycle. The provider
billing snapshot is partial ($5.85 posted at the earlier read), so the estimate
is not presented as a settled invoice; storage is separate and small.

Software: Python 3.12.3, JAX/jaxlib 0.9.0, CUDA compiler 12.8.93, BeamZ CUDA
component 0.19.0 / ABI 19, SM90 release build, fast math disabled. The exact
installed package list, extension hash, solver source hash, and patch are
retained with the artifacts.

## Correctness gate and necessary fix

The initial selected hardware suite had 29 passes and three Hopper failures.
Both existing Hopper complete-state tests (PEC and CPML) and the new modal
Hopper test failed with `BeamZ CUDA kernel launch failed: 1`.

`compile_simulation` prepared native H/E material coefficients only for
`cuda_streamed`. Hopper instead received empty placeholders intended for the
JAX path. Including `cuda_hopper` in those two conditions fixes the launch
contract without changing CUDA kernels or update arithmetic.

After the fix, all three previously failing hardware cases passed. The targeted
Hopper selection also passed seven unit tests, including a new assertion that
all 12 H/E source/decay coefficients match streamed preparation and are nonempty.
The full 32-case hardware selection was not rerun after the fix; its other 29
cases had already passed.

The new modal hardware test uses an odd 37×49×65 grid, 12-cell CPML, a solved TE
mode source, two mode monitors at three frequencies, and seeded nonzero fields.
It compares a 33-step JAX reference with 17+16-step continuation for streamed
on 1/2/4 GPUs and Hopper on one. Every state leaf is compared, including CPML
memories and monitor accumulators. Existing test tolerances apply. These checks
establish backend agreement for the tested cases, not mesh convergence.

## Measurement protocol

All canonical trials use FP32 fields and CPML, 256 timesteps, a fresh process,
one discarded warmup and five fully synchronized samples. They preserve input
state (`donate_state=False`). Compilation is timed separately. The reported
warm throughput is useful material-grid cells × timesteps / median seconds /
1e9; it includes the complete compiled execution, not just individual CUDA
kernels. The harness checks all final-state leaves for finiteness outside the
timed interval.

The default partition axis is `auto`: the longest dimension, with ties choosing
z before y before x. Thus the cubes use z and the elongated pilot grids use x.
The explicit y/z controls use the same 128×256×384 CPML + DFT workload.

Public-path measurements time `sim.advance` after a warmup, including allocation,
placement, and result decoding. They exclude cold geometry and mode preparation.
These short-run latencies can rank backends differently from warm execution.
Peak allocator statistics include preparation, and summed per-device peaks
are not simultaneous measured peaks.

The five pilot workloads are: bare PEC; heterogeneous guide with CPML10,
Gaussian source and a three-frequency field DFT plane at three shapes; and an
80 nm silicon/cladding guide with CPML12, a solved TE source and two mode
monitors at three frequencies. The first four use the repository's canonical
`H100Workload`; the modal case uses `benchmark_cuda_realistic.build_simulation`.
These are fixed-step performance scenarios, not converged optical simulations.

## Completed pilot results

Warm aggregate GCUPS; larger is faster. Dimensions are z×y×x. All 35 trials
completed without execution failures and produced finite final states.

| Workload / grid | JAX, 1 GPU | Streamed, 1 GPU | Hopper, 1 GPU | Streamed, 2 GPUs | Streamed, 4 GPUs |
|---|---:|---:|---:|---:|---:|
| Bare / 128×256×384 | 25.523 | 30.757 | 4.114 | 5.873 | 10.136 |
| CPML + DFT / 128×256×384 | 16.961 | 19.581 | 3.685 | 4.937 | 6.541 |
| CPML + DFT / 256³ | 8.622 | 19.239 | 3.626 | 4.824 | 6.715 |
| CPML + DFT / 96×256×768 | 18.104 | 21.765 | 3.831 | 5.452 | 7.637 |
| Modal / 128×256×384 | 11.371 | 18.754 | 3.357 | 4.830 | 6.930 |

For the medium CPML + DFT case, public-path throughput is 9.360 GCUPS for JAX,
8.596 for streamed and 3.082 for Hopper. Thus the streamed warm-execution win
does not establish an end-to-end win for every workload or run duration.

## Larger-domain scaling

The 512³ CPML + DFT domain contains 134,217,728 material cells. Its automatic
partition axis is z. All seven configurations completed and had finite final
states.

| Backend | GPUs | Warm GCUPS | Public-path GCUPS | XLA compile seconds |
|---|---:|---:|---:|---:|
| JAX | 1 | 10.071 | 7.173 | 14.34 |
| Streamed | 1 | 23.626 | 11.999 | 4.72 |
| Hopper | 1 | 4.011 | 3.411 | 6.30 |
| JAX | 2 | 7.630 | 5.601 | 19.81 |
| Streamed | 2 | 5.653 | 4.517 | 125.84 |
| JAX | 4 | 10.117 | 6.986 | 21.63 |
| Streamed | 4 | 9.008 | 6.206 | 107.30 |

Four-GPU streamed warm throughput is 0.381× its single-GPU baseline; four-GPU
JAX is 1.004× its own. Single-GPU streamed remains the fastest tested choice
for this fixed-size domain. Sharded streamed compilation also takes about
107–126 seconds here, independent of the slower warm execution.

The medium 128×256×384 grid's four-GPU streamed axis control gives 6.541 GCUPS
for automatic x, 6.933 for y, and 6.079 for z. Corresponding public-path values
are 4.037, 4.050, and 3.761 GCUPS. None approaches the single-GPU 19.581 warm /
8.596 public-path result. The small x/y difference needs paired repetition
before it could justify a partition-policy change.

## Physical cosine crossing

The repository's cosine-crossing benchmark holds the physical domain and
428.34 nm CPML thickness fixed while changing resolution. It uses a TE mode
source, a three-component field DFT plane, two flux monitors, and 101 frequencies
from 1.26–1.36 µm. The grids are 55×381×381 at 36 nm and 79×548×548 at 25 nm.
All three backends use the same workload and FP32 state.

These throughput trials donate state, with a fresh zero state allocated and
synchronized outside each timing; one warmup precedes five 256-step samples.
They must not be pooled with the preserved-state canonical measurements.

| Resolution | JAX GCUPS | Streamed GCUPS | Hopper GCUPS |
|---|---:|---:|---:|
| 36 nm | 6.238 | 4.484 | 3.050 |
| 25 nm | 8.021 | 6.044 | 3.183 |

Streamed is 1.47–1.90× faster than Hopper on these crossings, but JAX is
1.33–1.39× faster than streamed. The canonical pilot's much larger streamed
advantage over Hopper does not generalize to every source/monitor workload.

The full 36 nm runs advance 14,570 steps over the original 1 ps duration. They
use one cold executable launch each, so the following are not warm medians:

| Backend | Execution seconds | GCUPS | Max transmission difference vs JAX | Max crosstalk difference vs JAX |
|---|---:|---:|---:|---:|
| JAX | 18.570 | 6.264 | reference | reference |
| Streamed | 24.779 | 4.694 | 0.003459 dB | 0.015949 dB |
| Hopper | 38.212 | 3.044 | 0.000001264 dB | 0.000003204 dB |

All six final field components and both 101-bin flux spectra are finite, and
there are no negative through/cross flux bins. Source launch power is identical
across backends. Against JAX, streamed's relative L2 differences are 0.0448% for
through flux and 0.1736% for cross flux; Hopper's are approximately 1.08e-7 and
3.23e-7 as fractions. No full-spectrum acceptance tolerance was specified, so
these are measured differences rather than a declaration of equivalence.
The larger streamed difference deserves investigation before a strict
no-regression gate. Neither backend agreement nor a single resolution proves
optical convergence.

## Diagnostic GPU traces

Six separate traces capture one warmed 32-step execution after three warmups.
Three use the canonical medium workload (streamed 1/4 GPUs and Hopper 1 GPU),
and three use the 36 nm crossing (JAX/streamed/Hopper on one GPU). All traces
preserve input state, including the crossing diagnostic. The reported benchmark
samples above never run under a profiler.

On one GPU, Hopper's `UpdateTiled` kernels account for 95.9% of summed GPU
event duration (192 component launches). Streamed's combined CPML update
kernels account for 92.9% (64 H/E launches). The update kernels themselves are
a substantial part of the Hopper gap; eliminating host launch overhead alone
would not account for it.

On four GPUs, native sharded updates account for about 40.1% of summed GPU
event duration, with NCCL kernels contributing about 35.7%. Both native update
work and communication deserve attention. These sums include overlapping
events across streams/devices and are not percentages of application wall time.
NCCL kernel duration includes waiting and does not establish network bandwidth
utilization. No hardware-counter claim about achieved bandwidth or stalls is
made.

For the crossing, streamed's `AccumulateDftGroups` kernel alone accounts for
**74.5%** of summed GPU event duration; native H/E updates account for 23.7%.
Hopper's update kernels account for 83.8% of its corresponding event time.
This identifies a concrete workload-dependent priority: improving streamed
monitor accumulation matters more here than replacing its already faster H/E
updates. The trace identifies time attribution, not the underlying bandwidth,
instruction, or occupancy cause.

## Implications for a single CUDA path

Keep `cuda_streamed` as the public CUDA path and treat Hopper-specific work as
an internal implementation candidate that must earn its dispatch conditions.
The current numbers do not justify replacing streamed kernels with Hopper's
shared-memory tiled kernels.

The earlier `HOPPER_CPML12.md` and `HOPPER_SEARCH.md` studies optimize
`cuda_streamed` on Hopper hardware. Their filenames should not be read as
measurements of the explicit `cuda_hopper` backend.

The implementations currently differ in both kernels and scheduling. Streamed
single-GPU execution can use native multi-step graphs, packed material codebooks,
and native source/DFT handling. The Hopper and sharded paths use phase-by-phase
execution. Hopper launches separate component kernels, and sharded execution
adds halo exchange and CPML communication. The benchmark does not isolate the
individual cost of every difference. The traces above narrow the next work to
Hopper update kernels, sharded update/communication work, and native monitor
accumulation on the crossing.

A unification change should retain streamed behavior as a control, isolate
kernel changes from scheduling changes, pass complete-state/continuation and
full-spectrum parity, and measure both warm and public-path latency. Repeated,
interleaved comparisons across these grids and hardware families are needed
before claiming no regression. Single-GPU improvements alone do not establish
multi-GPU scaling improvements.

Recommended order: retain the coefficient fix; investigate the long-run
spectral delta; optimize and validate high-frequency-count DFT accumulation;
improve sharded updates and collective scheduling; then consider an internal
SM90 kernel candidate only after paired tests show a gain. Measure any proposed
change against the unchanged streamed control, including the physical crossing.
An 8-GPU run remains outstanding because capacity was unavailable, not because
the $100 cap was exhausted. This initial sweep does not establish weak scaling,
distributed memory capacity beyond one GPU, or independent-simulation batch
throughput.

Five samples characterize this initial sweep, not a statistical no-regression
gate. Some pilot trials have isolated timing outliers (the largest max/min ratio
is 1.35); raw samples and medians are retained rather than discarding outliers.

## Reproduction and artifacts

Use a fresh directory for each invocation. On a prepared H100 node:

```sh
export BEAMZ_BENCHMARK_PYTHON=/opt/beamz-h100/bin/python
bash scripts/run_h100_backend_pilot.sh results/h100-comparison
/opt/beamz-h100/bin/python scripts/benchmark_h100_followup.py \
  --output results/h100-followup
```

The pilot runner performs the selected hardware correctness gate before timings.
The follow-up adds 512³ fixed-domain scaling, y/z partition controls, and
36/25 nm cosine-crossing throughput plus a 36 nm full 1 ps comparison.

Local raw evidence: `benchmarks/results/h100-backend-comparison/` (ignored).
Each trial retains its JSON and log; `summary.json` retains failed/timeout trials
as well as successes. `sweep/REPORT.md` and `sweep/measurements.csv` contain the
complete pilot table and sample ranges. `throughput.png` and `throughput.svg`
are generated by `scripts/plot_h100_backend_comparison.py`.

Versioned compact tables and numerical comparisons are in
[`h100-backend-comparison/`](h100-backend-comparison/). The profiler summary is
reproducible with `scripts/summarize_h100_profiles.py` on the raw `profiles/`
directory. The main comparison chart is available locally at
[`throughput.png`](../results/h100-backend-comparison/throughput.png).

Both archives were downloaded and SHA256-verified before terminating the pod.
Seven archived source files matched the local worktree byte for byte at download,
before the subsequent backend removal. The
binary extracted from the saved wheel matches every CUDA pilot record.

- Evidence archive: `beamz-h100-evidence.tar.gz`, SHA256
  `3dd97b83687dfe65794aa8b3138885e9f67eaf339169dc388bbe4d1cd06e360e`.
- Build archive: `beamz-h100-build.tar.gz`, SHA256
  `79964795c36ead264f949b05037eaf20a19c0cc243e1d1369c574325b08b5d55`.
- CUDA extension SHA256:
  `d7f29fbfb0651ea7e5323b43d13390a5006471ca72ebb74cf595533e77eba5e4`.

The local branch is `bench/h100-backend-comparison` in the separate
`beamz-h100-benchmark` worktree, leaving the original notebook edits intact.
No CUDA kernel or automatic backend policy was changed, and no performance
unification improvement is claimed by this patch.
