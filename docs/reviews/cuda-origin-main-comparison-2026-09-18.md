# Matched origin/main versus local CUDA branch — 2026-09-18

Compared fetched `origin/main` commit `c5fe0d8833ac6539241ae9d3c98445e5a7305786` with the local working tree based on `b0a5ff32`. These are fresh matched measurements, not a comparison against historical documentation.

Both use RTX3090, CUDA 13.3.73, SM86 Release builds with fast math off, the same Python/JAX environment, FP32 fields and auxiliary state, lossless material, CPML12 on all faces, one solved mode source, two compact mode monitors and three frequencies. All domains contain approximately 16.8 million logical cells. No padding or lossy workloads were used.

Each fresh process runs 256 steps, four warmups and nine synchronized timed executions. Cases use forward/reverse process order: main/default/layout/layout/default/main for narrow, main/default/default/main otherwise. The reported value is the mean of two process medians. Conversion costs are included; setup, compilation, mode solving, validation and host result hashing are excluded. These are warm compiled-executable rates, not public API or cold-start rates. No confidence interval is inferred from two process repeats.

| Physical domain (z,y,x) | Material | Main default | Branch default | Change | Branch opt-in layout | Change vs main |
|---|---|---:|---:|---:|---:|---:|
| 1024×256×64 | binary | 7.230 | 6.648 | -8.1% | 8.367 | +15.7% |
| 128×256×512 | binary | 8.231 | 8.566 | +4.1% | — | — |
| 257×193×341 | smooth | 6.965 | 8.244 | +18.4% | — | — |

All throughput values are logical GCUPS. The opt-in narrow variant uses `BEAMZ_CUDA_STORAGE_AXES=120` and `BEAMZ_CUDA_CPML_CORE_FUSION=0`. Defaults have all `BEAMZ_CUDA_*` overrides removed. The main checkout and native build are isolated; the branch extension was never replaced.

## Numerical comparison

Every run has identical initial-state, compiled-coefficient and compiled-source fingerprints for its physical case. Every final state is finite. All four narrow branch runs (default and cyclic storage) have identical complete-state hashes.

Branch versus main is not bitwise identical. The existing hardware-test tolerance was evaluated without relaxation: rtol=3e-5 and atol=max(3e-6, 1e-6 times reference leaf peak). Some CPML entries exceed it. This comparison therefore does not establish full numerical equivalence to main. The branch contains earlier changes to floating-point operation ordering; their contribution is plausible, but was not isolated in this study.

| Case / branch configuration (first repeat) | Max field error / leaf peak | Max CPML error / leaf peak | Entries failing tolerance |
|---|---:|---:|---:|
| narrow / branch_default | 1.23e-06 | 7.86e-06 | 72 |
| narrow / branch_layout | 1.23e-06 | 7.86e-06 | 72 |
| wide / branch_default | 9.64e-07 | 8.29e-06 | 108 |
| irregular / branch_default | 8.07e-07 | 2.18e-06 | 0 |

## Provenance and operational limits

The GPU power limit remained 370 W; peak observed temperature was 69 C and minimum free GPU memory was 19008 MiB. GPU processes ran sequentially with a 2-GiB headroom and 85-C stop guard. No solver changes, pushes or default changes were made for this comparison.

[Raw samples, full-state validation, input fingerprints, compiler flags, telemetry and archived benchmark/source files](rtx3090-2026-09-18-origin-main-comparison/). `analysis.json` records all process medians and every tolerance failure. An initial diagnostic run stopped at the first tolerance failure; its log is retained and its timing is excluded from the table. The completed runs record failures rather than silently relaxing tolerances.

No >=9-GCUPS guarantee, arbitrary-domain guarantee, or speedup outside these three workloads is established.

The archived worker is the exact unformatted version used for the completed comparison. The reusable worker in `scripts/benchmark_cuda_revision.py` was import-sorted and formatted afterward; lint passes. No timed workload or solver behavior was changed by that cleanup.
