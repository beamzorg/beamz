# Checkpointed CUDA material gradients on RTX 3090

The baseline is BeamZ `82e21533`, which already has differentiable streamed
CUDA timestep graphs, a native material transpose, and direct replay for
equal-width CPML slabs. [PR #550](https://github.com/beamzorg/beamz/pull/550)
provides a separate host checkpoint driver. It was open when this work began
and is not a dependency of this change.

## Reproduce

Build the BeamZ CUDA extension for SM 86 and use a CUDA-enabled JAX
environment. Run each case in a fresh process for comparable VRAM readings.

```bash
python scripts/benchmark_cuda_checkpoint_adjoint.py \
  --grid uniform --size 48 --steps 256 --checkpoint 32 --validate \
  --trace /tmp/uniform-trace --output /tmp/uniform.json
python scripts/benchmark_cuda_checkpoint_adjoint.py \
  --grid graded --boundary pec --size 48 --steps 256 --checkpoint 32 --validate \
  --trace /tmp/graded-trace --output /tmp/graded.json
```

Use `--grid axis --boundary pec` for constant but unequal axis spacings and
`--size 96` for a larger test. Each case uses a localized electric source,
random spatially varying permittivity, CPML, and a multifrequency vector field
monitor. The mixed boundary cases replace one CPML face with PEC. Ordinary
CUDA uses the streamed FFI schedule with the same initial state, compiled Yee
permittivity, source, monitor, and step count. Its objective is checked before
reporting a timing. Both paths exclude result decoding and host transfers.

The measurements use JAX 0.10.2, CUDA 12.9.86, NVIDIA driver 610.43.03, FP32,
and one RTX 3090. Wall times are medians of five synchronized samples after
three warm-up calls. The benchmark records individual samples, setup and
compilation time, and XLA temporary bytes. Peak VRAM is sampled per process after
warm-up. It includes the CUDA context and XLA allocator pool. Recorded forward
and reverse are compiled and timed independently. The two medians need not
sum to the single value-and-gradient call. Synthetic setup took 1.2–1.5 s per
fixture and compiling the combined gradient took about 0.3 s.

## Diagnosis and implementation

- Baseline CUPTI traces attributed 22–23 ms of the 32–39 ms small-fixture
  reverse pass to the Yee transpose kernels. The native transpose now prepares
  buffer strides once on the host and specializes the stencil for uniform,
  unequal-axis, and rectilinear metrics. The compact electric kernel handles
  all three dense, active electric source-coefficient gradients. The general
  transpose remains available for other coefficient layouts and active columns.
- Graded and asymmetric CPML used a per-step replay scan in the baseline. The
  native replay graph now writes magnetic fields and electric CPML memories
  straight into its tape on these grids. The last replayed timestep in each
  chunk stops after the magnetic update and its source injection.
- Native monitored chunks can prepare all DFT phases in one launch for forward
  and reverse. The batched phase workspace is bounded at 3 MiB. For the
  48-cube graded fixture this reduced forward phase-preparation launches from
  256 to eight.
- The warm path already uses CUDA graphs and XLA command buffers. Disabling
  BeamZ's graph cache on the 48-cube graded fixture changed the warm total by
  about 1–2% in a pre-final A/B run. This option still allows graph capture.
  The work prioritized reducing device kernel time and tape copies.
- Forward simulation was already close to ordinary CUDA. A differentiable
  temporal-blocking implementation would need replay and transpose support for
  its intermediate fields and CPML state. The ordinary mstep kernels are
  specialized for uniform metrics and provide no VJP. The measured gains here
  come from the streamed schedule.

## Synthetic results

Times are milliseconds, with baseline → updated medians. Reverse includes
replay, and total is a single call to `jax.value_and_grad`.

| Grid | Ordinary forward | Differentiable forward | Recorded forward | Reverse | Total | Total speedup | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform 48³, CPML | 6.94 → 7.08 | 7.23 → 7.23 | 7.64 → 7.68 | 32.26 → 21.38 | 40.27 → 29.18 | 1.38× | 420 → 416 |
| Unequal-axis 48³, CPML + PEC | 8.21 → 8.42 | 8.74 → 8.69 | 8.77 → 8.87 | 39.16 → 23.61 | 48.03 → 32.44 | 1.48× | 418 → 416 |
| Graded 48³, CPML + PEC | 8.57 → 8.76 | 9.14 → 8.62 | 9.05 → 8.87 | 38.40 → 23.14 | 48.14 → 31.93 | 1.51× | 420 → 416 |
| Uniform 96³, CPML | 43.93 → 44.55 | 43.70 → 43.95 | 45.15 → 45.25 | 176.84 → 137.27 | 223.18 → 166.62 | 1.34× | 1430 → 1426 |
| Graded 96³, CPML + PEC | 44.36 → 44.77 | 43.70 → 43.96 | 45.12 → 45.17 | 200.81 → 130.39 | 248.27 → 169.49 | 1.46× | 1430 → 1426 |

CUPTI replay compute measures the sum of the replay Yee and source kernel
durations. Device copies are shown separately because the trace includes
checkpoint, tape, and other XLA copies. These are device durations from a
profiled call, which can differ from the unprofiled wall medians above.

| Grid | Replay compute (ms) | Reverse device copies (ms) | Yee transpose (ms) |
| --- | ---: | ---: | ---: |
| Uniform 48³ | 6.33 → 6.44 | 0.50 → 0.51 | 22.83 → 11.93 |
| Unequal-axis 48³ | 7.41 → 7.90 | 3.59 → 0.49 | 22.47 → 12.39 |
| Graded 48³ | 7.28 → 7.66 | 3.57 → 0.48 | 21.97 → 12.24 |
| Uniform 96³ | 38.22 → 40.73 | 1.19 → 1.24 | 139.24 → 74.04 |
| Graded 96³ | 40.55 → 41.52 | 11.71 → 1.11 | 145.36 → 77.42 |

At 256 steps and a checkpoint interval of 32, direct replay executes 256 H
and 248 E updates. The reverse Yee transpose executes 256 updates in each
phase. The full gradient took roughly 3.6–4.1 ordinary forward calls on the
matched fixtures. Replay still costs approximately one forward pass.
Checkpoint storage includes both field phases and CPML state per chunk
boundary. The local replay tape grows with the interval and holds only H
fields and E-side CPML memories. Direct tape writes remove intermediate
copies without changing these retained-state requirements.

All five material gradients were within `1.7e-7` relative L2 error versus
the existing JAX VJP. Directional central differences differed by less than
`1.5e-5` relative. The hardware tests also cover mixed CPML and PEC,
conductivity and source-amplitude gradients, nonzero initial fields and CPML
memories, and ragged DFT plans and observation windows.

## Graded SOI transfer

The local archived v6 mesh and bulk continuation used a folded
`122 × 324 × 714` ZYX grid, a PEC symmetry plane, CPML, the original Gaussian
excitation, and the five-wavelength TE₀ monitor projection. The 64-step measured
window began after 4,608 forward steps and used six timesteps per checkpoint.
The incoming fields and CPML state were fixed for differentiation. The
observation accumulators started at zero for this window.

| Measurement | Baseline | Updated |
| --- | ---: | ---: |
| Ordinary CUDA forward | 0.347 s | 0.350 s |
| Differentiable forward | 0.376 s | 0.376 s |
| Recorded forward | 0.414 s | 0.405 s |
| Reverse including replay | 1.411 s | 1.109 s |
| Total gradient | 1.845 s | 1.539 s |
| Peak process VRAM | 22,934 MiB | 22,922 MiB |

The updated reverse CUPTI trace attributed 0.311 s to replay computation and
0.741 s to the Yee transpose. The objective was bit-identical and the largest
absolute gradient difference was `7.7e-34`. The gradient norm was `7.86e-14`,
so the synthetic fixtures provide the well-conditioned gradient accuracy
comparisons. The untimed prefix took about 28 s per process. Both builds
logged one recoverable allocation retry on this large case. The timing covers
only the fixed-state window. The historical v3 56 nm and full 39,769-step v6
runs describe different workloads.

## Compact replay and dense loss gradients

A replay variant records the corrected electric curl on each Yee support.
It uses scratch magnetic fields and CPML memories while replaying the
forward steps. The transpose reads the stored curl directly. For the
lossless material path, the planner selects this tape when its predicted
storage, including the extra scratch state, is smaller. Dense loss gradients
save the old electric fields as well as the curl so the same native chunk
can form both electric coefficient gradients.

On the RTX 3090, the curl tape reduced XLA temporary storage on the
uniform 96³ fixture from 702 to 624 MiB and on the graded 96³ fixture from
647 to 600 MiB. These are compiler temporary allocations. The change in
process-wide peak VRAM depends on allocator reuse.

To compare the native dense loss route with the per-step loss VJP, run the
benchmark below with the environment variable set to 0 and to 1 in separate
processes.

```bash
BEAMZ_CUDA_ADJOINT_CURL_REPLAY=1 python scripts/benchmark_cuda_checkpoint_adjoint.py \
  --grid graded --boundary pec --size 48 --steps 128 --checkpoint 16 \
  --conductivity 120 --validate
```
