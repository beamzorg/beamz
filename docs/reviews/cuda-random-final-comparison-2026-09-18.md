# Final random-domain comparison with origin/main — 2026-09-18

The branch was faster in all four cases, by **6.6–19.1%**, without calibration.
In each case the slower branch process still beat the faster main process.
Throughput ranged from **7.09 to 8.73 GCUPS**, so consistent 9 GCUPS remains
unachieved. Fields and monitors passed the existing numerical tolerance in all
four cases; CPML auxiliary-state failures remain in three cases.

Four previously unmeasured domain shapes were sampled before running benchmarks,
using seed `2026091807`. Sampling was stratified into short-x, long-y, flat-z and
irregular domains, accepting 16–24 Mi logical cells. This is a small prospective
check of the frozen predictor, not a representative random sample of all devices
or a guarantee for every simulation. No selector thresholds or kernels were
changed in response to these results.

Each case used FP32, lossless binary or smooth material, exactly 12 CPML cells
on all faces, one mode source, two compact mode monitors and three frequencies.
Each executable advanced 256 steps. Main/branch/branch/main ran sequentially in
fresh processes, each with four warmups and nine synchronized timed executions.
Rates below average the two process medians. Logical cell counts exclude padding;
internal storage conversions are timed. Setup, JIT compilation, mode solving and
result validation are excluded from these warm executable throughputs.

## Performance

| Domain (z,y,x) | Cells (million) | Material | Main GCUPS | Branch GCUPS | Change | Internal axes |
|---|---:|---|---:|---:|---:|---|
| 1126×297×65 | 21.74 | binary | 7.118 | 8.411 | +18.2% | 120 |
| 104×1083×180 | 20.27 | smooth | 5.950 | 7.088 | +19.1% | 012 |
| 110×202×866 | 19.24 | binary | 8.188 | 8.728 | +6.6% | 012 |
| 325×203×312 | 20.58 | smooth | 7.062 | 8.343 | +18.1% | 012 |

Process medians (main repeats; branch repeats):

- short_x: 7.122, 7.113; 8.322, 8.501 GCUPS.
- long_y: 5.865, 6.036; 7.088, 7.089 GCUPS.
- flat_z: 8.187, 8.188; 8.728, 8.728 GCUPS.
- irregular: 7.062, 7.062; 8.342, 8.344 GCUPS.

All branch processes used the default geometry predictor with **zero calibration
time** and shell 64×4. Only short-x rotated; the other layouts retained canonical
storage. The long-y case's estimated queue-work reduction was 19.3%, below the
25% policy margin, so it remained unchanged. This comparison does not establish
whether a different layout would improve that case: no candidate sweep was run.

## Numerical check

Initial states, material coefficient arrays and compiled source arrays matched
exactly between revisions for every process. Every final-state leaf was finite.
Validation used the same existing tolerance, without relaxing it:
`abs(error) <= max(3e-6, 1e-6 * reference_leaf_peak) + 3e-5 * abs(reference)`.

| Domain case | Fields pass | Monitors pass | CPML entries outside tolerance (each repeat) | Max CPML error / leaf peak |
|---|---|---|---|---:|
| short_x | True | True | [16, 16] | 4.31e-06 |
| long_y | True | True | [716, 716] | 3.62e-06 |
| flat_z | True | True | [476, 476] | 3.02e-06 |
| irregular | True | True | [0, 0] | 2.37e-06 |

These are main-versus-branch comparisons. CPML failures must not be described as
complete numerical equivalence; related discrepancies were already present
before predictive layout selection. Prior seeded selector tests establish
branch-internal parity for their tested cases, not universal parity with main.

## Provenance and execution

Compared fetched `origin/main` `3dc23ed55eb5fb22f9539c8dd69bfb918e3c0c84` against the current local branch at base `b0a5ff32d70755aa3c6f5d20e8ae9a5fdfe4ee7b` with its uncommitted changes captured in the solver snapshot. Main advanced from the earlier comparison only through `uv.lock` changes;
solver/native sources are identical to `c5fe0d88`, so its existing compiled
extension was reused. Both checkouts used the same CUDA/JAX Python environment.
Native builds target SM86, Release, without fast math. Per-run JSON records
extension, worker, workload and input hashes. The runner verified that solver
sources stayed unchanged throughout the comparison.

Peak sampled temperature: **69°C**. Minimum free VRAM: **14.59 GiB**. Power limit: **370 W**, unchanged. GPU utilization reached **100%**. Existing desktop/T3 processes were left running. Nothing was pushed.

[Raw timings, numerical validation, telemetry and source provenance](rtx3090-2026-09-18-random-final/).
[Machine-readable aggregate](rtx3090-2026-09-18-random-final/analysis.json).
Reproduce with `scripts/benchmark_cuda_random_comparison.py`, passing an isolated
main checkout, a fresh output directory and a fresh reference-state directory.
