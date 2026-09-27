# H100 public-preparation acceptance — 2026-09-27

This report records a new $40-capped hardware session after the public preparation
fixes. Public 15B CUDA preparation and executable compilation succeeded, but
execution failed on source-related workspace. JAX preparation was stopped at
the budget deadline. The pod is deleted; conservative new-session cost is $38.03.

## Reproducibility

- Fixed solver: `b673256`; baseline: `a32227d` (previous solver plus preparation tracing).
- Capacity harness adds monitor DFT capture at `0482d09`; solver code is unchanged.
- One secure CA-MTL-1 host, eight H100 SXM 80 GB GPUs, NV18 links between every pair.
  Smaller checks restrict `CUDA_VISIBLE_DEVICES`; the eight-GPU host remains billed.
- Python 3.12.3, JAX 0.9.0 CUDA 12; identical native CUDA extension for both revisions.
  Exact packages, binary/source hashes, commands and allocator settings are in `raw/`.
- FP32 CPML, automatic CUDA memory policy, donated state, CUDA autotuning disabled.
  JAX preallocates 90% of device memory. Pool reservation is not live-array memory.
- All GPU workloads run sequentially. No numerical tolerance or monitor frequency
  count was relaxed. Hardware health rates differed by approximately 1.6%.

## Correctness

All 30 hardware tests passed: 14 with one H100, eight with two, and eight with
all eight H100s. Coverage includes real x/z sharding, automatic/capacity schedules,
CUDA/JAX field parity and continuation. The earlier eight-GPU crop-contract
assertion has now been rerun successfully.

A separate 128×96×256 modal run propagates for 2,048 steps. Eight-H100 CUDA and
JAX spectra both match the one-H100 JAX reference under the unchanged tolerances
(`rtol=3e-5`, `atol=max(1e-12, 2e-6*reference_peak)`). Both monitors receive signal.
Worst relative L2 errors are 6.90e-7 (CUDA) and 6.47e-7 (JAX).

These are numerical acceptance cases, not the 15B capacity measurement.

## Throughput protocol

Three independent baseline/fixed process pairs per configuration; reverse their
order for the middle pair. Every process compiles, warms up, then measures five
128-step donating continuations. Compare within-pair median segment time. Shapes
are 256³ for one H100 and 256×256×512 for two, with both x and z partitions on two.
The acceptance criterion is less than 5% slowdown in every pair.

**All 18 process pairs pass.** The worst slowdown is 0.549%. GCUPS below are
medians across the three process medians; slowdown is computed within pairs.

| H100s / axis | Backend | Baseline GCUPS | Fixed GCUPS | Median paired slowdown | Worst paired slowdown |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 / x | cuda_streamed | 20.134 | 20.158 | -0.108% | -0.085% |
| 1 / x | jax | 7.394 | 7.396 | -0.053% | +0.044% |
| 2 / x | cuda_streamed | 25.529 | 25.464 | +0.263% | +0.549% |
| 2 / x | jax | 14.638 | 14.649 | -0.079% | -0.007% |
| 2 / z | cuda_streamed | 21.775 | 21.781 | -0.031% | +0.406% |
| 2 / z | jax | 11.739 | 11.745 | -0.028% | +0.023% |

Run `python benchmarks/reports/h100-prep40/summarize.py` to rebuild `summary.json`.
Raw timings are retained so compilation, preparation and warm execution remain
separate. GCUPS counts cell updates, not component updates.

## Capacity protocol

The ordinary `ModalWorkload.build()` → `Simulation.compile()` preparation path
uses `(z,y,x)=(10000,1250,1200)`, exactly 15,000,000,000 cells, at 80 nm resolution
(physical x/y/z extent 96/100/800 µm), binary waveguide materials, a TE mode source,
12-cell CPML and two 101-frequency mode monitors. This starts with public dense
`MaterialGrid` inputs; it does not substitute the prepared coefficient fixture.

The short capacity gate compiles a 32-step segment and executes it three times.
A subsequent longer test, budget permitting, requires nonzero DFT accumulation
at both monitors. That requirement demonstrates propagated signal, not optical
convergence. Smaller propagated parity tests provide numerical reference evidence.

The core reproduction command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH='' \
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=.90 \
BEAMZ_CUDA_CPML_PSI_PRECISION=fp32 BEAMZ_CUDA_AUTOTUNE=off \
NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
PYTHONPATH=. BEAMZ_TRACE_PREPARATION=phases.jsonl \
python scripts/benchmark_compile_capacity.py --workload modal \
  --shape 10000 1250 1200 --devices 8 --axis z --backend cuda_streamed \
  --frequencies 101 --steps 32 --samples 3 --output result.json
```

Replace the backend with `jax` for its separate capacity test. The longer CUDA
case uses `--steps 1024 --samples 5 --require-monitor-signal`.

Phase traces include current/peak host RSS and stage snapshots include all eight
GPU allocator peaks. Peak counters are cumulative; nested phase times must not
be added. The harness validates finite state through bounded local slabs outside
stepping timing, avoiding a whole-domain validation workspace.

## Public 15B CUDA result

**Preparation passed; execution failed.** This is not a 15B simulation pass.
The unmodified preparation revision `b673256` reached these stages:

| Stage | Time | Host peak RSS | GPU live data / rank | GPU peak / rank |
| --- | ---: | ---: | ---: | ---: |
| Public preparation and placement | 984.69 s | 113.02 GiB | 64.607 GiB | 64.672 GiB |
| Lower | 0.87 s | 113.02 GiB | 64.607 GiB | 64.672 GiB |
| Executable compilation | 3.11 s | 113.02 GiB | 64.607 GiB | 64.672 GiB |
| First 32-step execution | Failed | — | — | — |

All eight ranks have the same live/peak allocator values. Automatic CUDA policy
selected the capacity schedule. The compiled executable still requests
**14.026 GiB of temporary workspace per rank**, beyond the available headroom
in the 71.27 GiB allocator pool. Execution raises `RESOURCE_EXHAUSTED` before
completing a segment; there is no 15B GCUPS result from this session.

The prior public preparation attempt exceeded 32.5 minutes and sampled 1.26 TiB
host RSS without reaching compilation. This run now completes preparation in
16.41 minutes with 113.02 GiB peak RSS. This is a cross-session diagnostic
comparison, not a controlled preparation-speed benchmark. The dense immutable
input copy explains most of the current peak. Source/monitor planning takes
255.66 s; coefficient generation/placement takes 461.12 s. Uniformity checks and
other preparation remain additional CPU work. Nested phase durations overlap.

The per-case 900-second watchdog was extended within the unchanged $40 session
cap by pausing only the waiting coordinator, not the benchmark process. Exact
pause/resume times are in `raw/coordinator-extension.json`; the actual process
exit code was retained. No solver behavior or numerical criterion changed.

## Public 15B JAX result

**Incomplete because of the budget deadline**, not a recorded GPU OOM. The case
ran for 955.02 s and was terminated while generating/placing coefficients. Host
peak was 113.02 GiB; executable compilation and execution were not reached.
The last open phase is retained in its JSONL trace. There is no final capacity
JSON or JAX 15B throughput result. The controller records return code 124, and
`coordinator-jax-extension.json` records the budget termination explicitly.

The source/monitor phase took 342.71 s. Small source-workspace probes and the
source-fix hardware tests overlapped this case, so its preparation timing is not
a controlled comparison against the CUDA preparation time. They did not change
its source checkout: both full 15B cases tested solver revision `b673256`.

## Source workspace diagnosis and fix

A separate compile-only 512×128×256, eight-GPU capacity-schedule probe isolates
sources and monitors. These tests ran while the JAX capacity case was in host
preparation and are **not throughput measurements**.

| Probe | Original workspace bytes/rank | Linear source-update workspace bytes/rank |
| --- | ---: | ---: |
| Mode source + monitors | 31,773,768 | 14,601,032 |
| Monitors only | 14,601,024 | 14,601,024 |
| Mode source only | 29,832,216 | 12,659,480 |
| Neither | 12,659,472 | 12,659,472 |

Optimized HLO shows full-field transposes surrounding multidimensional thin-plane
source updates. Source injection adds 17,172,744 bytes on this probe, approximately
two complete local fields. This explains the size-dependent workspace absent from
the earlier source-free prepared fixture; monitor accumulation is much smaller.

Commit `8d1529d` changes local source patches to flat-index scatter additions,
keeping native row-major storage. Each patch has unique indices; overlapping
sources are still applied in their original sequence. A multidimensional fallback
preserves indexing for shards larger than the signed 32-bit flat-index limit.
On the probe, source-specific extra workspace falls to **eight bytes**.

The change passes **all eight eight-H100 full-state/continuation tests**, with
unchanged tolerances, plus **six local tests**, including crossing patches on all
three partition axes, overlapping patches, clamped origins and waveform indices.
The 18 paired throughput comparisons earlier in this report predate this source
change. **A fresh 15B execution, propagated spectra and paired throughput checks
of this final source change remain outstanding.** The small probe does not prove
15B execution or justify extrapolating GCUPS.

## Remaining scope

Dense geometry rasterization, necessary CPML extrusion copies, fallback
full-tensor/x64/sponge preparation, optional whole-volume diagnostics and
nondivisible public-state cropping remain separate capacity limits. Pure JAX
execution workspace must be evaluated separately from successful preparation.

## Resource lifecycle and remaining acceptance

Owned pod `gdm1u0x1hohyum` was created at 13:08:01.377 UTC and deleted with HTTP
204, followed by a confirming GET 404 at **14:29:38.295 UTC**. All **165** final
remote evidence files were frozen, copied and SHA-256 verified before deletion;
`raw/SHA256SUMS.json` is the integrity manifest. Independent deletion watchdogs
were stopped only after confirming absence. No owned resource remains running.

Elapsed-time compute estimate is $37.9783 at $27.92/hour; allowing $0.05 for disk
charges gives a conservative **$38.03 of the newly approved $40**. The provider
billing query currently reports only $9.1685 and is incomplete. This estimate is
separate from the prior $25 allowance; see `accounting.json`.

The requested additional $15 has not been approved or spent. The next run should
use `8d1529d` and execute the public 15B CUDA modal case for 1,024×5 continuing
steps with `--require-monitor-signal`, then verify final-source-change spectra
and matched throughput. JAX's remaining workspace/capacity must still be tested
separately. Neither the new source fix nor the preparation fix establishes 15B
realistic execution until that full-size rerun succeeds.
