# Streamed CUDA and JAX scaling

Historical measurements supporting PR #287. The change improves distributed
stepping, modal accumulation, continuation, and shard-local memory preparation;
it removes `cuda_hopper`. Use `cuda_streamed` or JAX. The native CUDA extension
must be rebuilt for **ABI 21 / version 0.21.0**.

## Results and limits

Eight H100 SXM 80 GB GPUs, FP32 CPML, 80 nm grid, 12-cell CPML, a mode source,
and two 101-frequency mode monitors. These are distinct experiments on the
revisions recorded in their archived reports, not a new run of the cleanup
commit. Environment details and exact source/native hashes are in the archive.

| Experiment | CUDA | JAX | Timing boundary |
| --- | ---: | ---: | --- |
| 2.097B cells, `(640,640,5120)` | 151.57 GCUPS | 73.92 GCUPS | Median of five synchronized warm 256-step samples |
| 1.074B cells, `(128,1024,8192)` | 135.50 GCUPS | 66.76 GCUPS | Same warm stepping protocol |
| Finite-pulse S-bend, `(128,512,1024)` | 82.68 s | 95.17 s | Single fresh-process completion trials, excluding final NPZ serialization |
| Capacity sweep, 4.607B cells, `(832,832,6656)` | 157.66 GCUPS | Not measured | Warm stepping; 58.82 GiB maximum per-GPU peak live allocation |
| Public preparation and execution, 15B cells, `(10000,1250,1200)` | 155.56 GCUPS | Not validated | Median of four warm 1,024-step segments, after an initial segment |

Shapes are `(z,y,x)`. Short warm samples measure stepping throughput and may end
before light reaches distant monitors. They do not establish converged optical
completion or universal scaling. CUDA met the earlier 150 GCUPS target on the
2.097B case; JAX did not meet 75 GCUPS. The capacity sweep used the same native
binary, but its later 15B follow-up includes additional preparation/source fixes.

The S-bend runs converged after 16,384 steps (2.398 ps). Completion includes
construction, compilation, continuation, convergence checks, and final mode
decomposition. The optimized CUDA result is compared with the earlier JAX
baseline; these are single trials, not latency distributions. Convergence
requires three checkpoints with field-norm/peak below `1e-6` and raw complex-DFT
relative L2 change below `1e-4`. Field norm is a decay proxy, not physical energy.
No spatial mesh-convergence claim is made.

The 15B public CUDA case completed preparation in 969.20 s and 5,120 steps in
494.964 s, with finite state and signal at both monitors. Host peak was 113.02
GiB; each GPU peaked at 64.672 GiB of JAX allocations. This demonstrates capacity
and propagation, not optical convergence or full-size backend parity. Pure-JAX
15B execution and final-source-change paired performance remain unvalidated.
An additional propagated-spectrum comparison was not rerun after that source
fix; its full-state/continuation tests did pass.

Correctness comparisons retain `rtol=3e-5` and
`atol=max(1e-12, 2e-6*max(abs(reference array)))`. The earlier completion
comparisons passed, including the optimized CUDA run. The 80 nm propagated
CUDA/JAX comparison passed. **The separate 64 nm refinement failed raw-DFT
pointwise parity**, although both flux spectra passed; its timing is not a
correctness acceptance result. Hardware counters were unavailable. Allocator
live/peak measurements, reserved pools, and NVML residency are distinct.

## Maintained benchmark tools

Run from the repository root in an environment with BeamZ dependencies and,
for CUDA, the matching rebuilt extension. Set `PYTHONPATH=.:scripts` for both
package and direct-script imports. Select visible GPUs before starting Python.
Keep generated output under ignored `benchmarks/results/`.

| Purpose | Tools in `scripts/` |
| --- | --- |
| Strong/weak scaling | `benchmark_modal_scaling.py`, `benchmark_modal_stepping.py`, `benchmark_h100_backends.py` |
| Summarize scaling runs | `summarize_modal_scaling.py` |
| Full optical completion | `benchmark_device_completion.py` |
| Public preparation and compilation/capacity | `benchmark_public_preparation.py`, `benchmark_compile_capacity.py` |
| Propagated numerical parity | `validate_modal_scaling.py`, `compare_modal_scaling.py` |

These extend the existing `benchmark_h100.py` and
`benchmark_cuda_realistic.py` harnesses. Scaling workers require H100s; preparation
can also run on CPU. The capacity probe defaults to a synthetic prepared fixture;
use `--workload modal` to include public material/source/monitor preparation.

Preview a small scaling schedule without running GPU work (use a fresh output
directory each time); remove `--dry-run` to execute on matching hardware:

```bash
PYTHONPATH=.:scripts python scripts/benchmark_modal_scaling.py \
  --counts 1 2 --local-sizes 128 --cubes 256 --frequencies 101 \
  --stepping-only --host-setup --dry-run \
  --output benchmarks/results/scaling/baseline
```

After real runs, summarize their results:

```bash
python scripts/summarize_modal_scaling.py benchmarks/results/scaling \
  --output benchmarks/results/scaling.csv
```

Run each completion backend in a separate process with the same visible devices,
shape, step limit, and checkpoint interval, then compare the saved artifacts:

```bash
PYTHONPATH=.:scripts python scripts/benchmark_device_completion.py \
  --backend jax --devices 1 --output benchmarks/results/completion-jax.json
PYTHONPATH=.:scripts python scripts/benchmark_device_completion.py \
  --backend cuda_streamed --devices 1 --output benchmarks/results/completion-cuda.json
python scripts/compare_modal_scaling.py benchmarks/results/completion-jax.json \
  benchmarks/results/completion-cuda.json --output benchmarks/results/parity.json
```

For a smaller preparation measurement on two simulated CPU devices:

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  PYTHONPATH=.:scripts python scripts/benchmark_public_preparation.py \
  --shape 64 64 128 --devices 2 --backend jax \
  --output benchmarks/results/preparation.json
```

These examples exercise maintained tools; reproduce historical numbers using
the exact workload, hardware, environment, and revision in the original report.
Automated correctness contracts remain in `tests/`; this report is not a test.

## Archived evidence

[Download the PR #287 evidence archive](https://github.com/beamzorg/beamz/releases/tag/evidence-pr287-20260929).
It preserves all 727 tracked files under `benchmarks/reports/` and `scripts/` at
commit `b6fd79dc8ad5a73640e8069004ea09f4a4e78d34`, including superseded reports,
experiment controllers, plots, telemetry, compiler dumps, and raw results.
`ARCHIVE_MANIFEST.json` contains a SHA-256 for every archived file. The uploaded
archive was downloaded and verified before removing evidence from this tree.

File: `pr287-benchmark-evidence.tar.gz` (13,550,605 bytes).
SHA-256: `7a62a1959038960c848c4d07262b98d0bddea1b3c245615c5a5c8fd740fd2e9e`.
Verify with the release's `SHA256SUMS` using `sha256sum -c SHA256SUMS`.

Start with `H100_DEVICE_COMPLETION.md` for stepping/completion,
`H100_CAPACITY_STUDY.md` for the size sweep, and `PUBLIC_PREPARATION.md` plus
`h100-source15/README.md` for the final 15B result, under `benchmarks/reports/`
inside the archive. Original relative links and script paths are preserved.
The archive does not include the previously local-only 249 MB and 10.7 MB raw
archives mentioned in historical reports. No claim is made that those are
available from this release. The evidence tag also preserves the complete
source revision for old experiment scripts.

Future benchmark runs should write generated files to `benchmarks/results/` and
publish durable evidence separately; commit only concise reports and selected
small summary tables. This cleanup reduces the PR diff, not existing Git history.
