# Public 15B CUDA source-layout follow-up

Status: **passed: full public 15B CUDA preparation, compilation and 5,120-step
execution on eight H100s, with finite final state and signal at both mode
monitors.** Warm throughput is **155.56 GCUPS**. This is the separately approved
$15 follow-up to [the $40 preparation session](../h100-prep40/README.md).

The previous public 15B CUDA run completed preparation and compilation, then
failed to allocate 14.026 GiB/rank of temporary workspace. Its mode-source patch
updates caused full-field layout conversions. Commit `8d1529d` replaced these
with flat-index local additions, removing the source-specific volume workspace
in a smaller compiler probe. Eight real eight-H100 parity/continuation tests and
six local source tests passed before this full-size run.

## Protocol

- Revision `a2750b7`: source fix plus optimized-HLO evidence capture; same native
  CUDA binary as the preceding session (SHA-256 begins `6753641e1a682751`).
- Secure CA-MTL-1 host, eight H100 SXM 80 GB GPUs, $27.92/hour. No concurrent
  benchmark workloads. Exact topology, packages and binary/source hashes in raw
  evidence. A short identical-load check precedes the capacity test.
- Ordinary public `ModalWorkload.build()` and `Simulation.compile()` path:
  `(z,y,x)=(10000,1250,1200)`, exactly 15B cells, 80 nm resolution, physical
  x/y/z extent 96/100/800 µm, binary waveguide materials, 12-cell FP32 CPML,
  TE mode source and two 101-frequency mode monitors.
- CPU setup followed by direct sharded placement; automatic CUDA capacity
  scheduling; donated continuation state; 90% JAX pool;
  CUDA autotuning off. No reduced precision, source replacement or monitor removal.
- Compile one 1,024-step segment and execute it five times (5,120 steps), keeping
  the same state and coefficients. Time segments individually, excluding setup,
  executable compilation and final validation. Report warm GCUPS from segments
  two through five, with all raw segment times retained.
- Require finite final state, positive DFT weights and nonzero DFT amplitudes at
  both monitors. This demonstrates propagated signal, not optical convergence
  or full-size parity against another backend. Smaller full-state and propagated
  reference tests are documented in the preceding report.

Reproduce with the environment and CLI in the preceding report, changing to:

```bash
python scripts/benchmark_compile_capacity.py --workload modal \
  --shape 10000 1250 1200 --devices 8 --axis z --backend cuda_streamed \
  --frequencies 101 --steps 1024 --samples 5 --require-monitor-signal \
  --output result.json
```

## Measured result

| Measurement | Result |
| --- | ---: |
| Logical cells | 15,000,000,000 |
| Resolution | 80 nm |
| Public preparation and device placement | 969.20 s (16.15 min) |
| Executable lowering / compilation | 0.88 s / 3.14 s |
| First 1,024-step segment | 100.002 s; 153.60 GCUPS |
| Four warm 1,024-step segments | 98.735 / 98.742 / 98.742 / 98.742 s |
| Warm median throughput | **155.5566 GCUPS** |
| Warm runtime coefficient of variation | 0.00298% |
| Total 5,120-step execution | **494.964 s (8 min 15 s)** |
| Bounded final-state/monitor validation, outside stepping timing | 152.45 s |
| Host process peak | **113.02 GiB** |
| Live arrays on every H100 | **64.607 GiB** |
| Peak JAX allocation on every H100 | **64.672 GiB** |
| Compiled temporary workspace per rank | **42,100,552 bytes (40.15 MiB)** |

The earlier failing full public case requested 15,060,558,664 temporary bytes per
rank (14.026 GiB). The source-layout fix removes approximately 14 GiB of field
layout workspace without changing the stored physical fields or coefficients.
The run automatically selects the CUDA capacity schedule. Allocator statistics
are distinct from the 90% pool reservation and CUDA/NCCL memory reported by
`nvidia-smi`; every rank's snapshots are retained.

Both mode monitors have 101 frequencies and 200 sampled spatial points. Their
raw accumulated DFT peaks are 404,856,384 and 418,185,120, respectively, and all
DFT weights are positive. These raw amplitudes are not normalized transmission
coefficients. The final step counter is exactly 5,120; all checked field, CPML
and monitor state is finite. The physical simulated duration is approximately
0.749 ps at this grid's timestep. This establishes repeated execution and signal
propagation, not optical convergence.

The new allocation regression test also **passes on eight H100s** (48.61 s of
pytest time). It compares the compiled workspace with and without sources and
requires the source-specific increase to remain smaller than one full local
field. Its original failing implementation exceeds that bound by approximately
two local fields in the retained ablation. The test is committed as `0c12eb4`;
it does not change the solver measured here.

An additional fresh small-domain propagated-spectrum comparison was skipped
because the remaining budget window was insufficient. The final source fix
already passed eight-H100 full-state/continuation tests and six local tests in
the preceding session. Do not describe the older propagated-spectrum check as a
rerun of this source change.

Run `python benchmarks/reports/h100-source15/summarize.py` to rebuild the numerical
summary from the raw case. All segment samples and optimized HLO are retained.

## Scope

The full 15B baseline cannot execute this public workload, so a paired full-size
throughput comparison is unavailable. The 18 earlier baseline/fixed process pairs
validate preparation revision `b673256`; they predate the source-layout change.
The earlier source-free prepared fixture is not a matched realistic baseline.
Pure JAX's full 15B execution remains unvalidated separately.

## Budget and resource lifecycle

Pod `2hd25htbjwodh4` was created at **14:33:32.780 UTC** and deleted with HTTP 204,
followed by a confirming GET 404 at **15:04:33.555 UTC**. The independent deadline
watchdog also confirmed its absence. All **21** final remote evidence files were
frozen and SHA-256 verified before deletion; `raw/SHA256SUMS.json` records them.
No resource from either of these two sessions remains running.

Elapsed-time compute estimate is **$14.4313** at $27.92/hour. Allowing $0.05 for
storage gives a conservative **$14.49 of the additional $15**. Provider billing
has not posted records yet. Together with the preceding session's $38.03 estimate,
these two allowances used at most **$52.52 of $55**. Accounting is estimated,
not a final provider invoice; see `accounting.json`.
