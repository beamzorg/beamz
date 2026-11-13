# CUDA tracing and nanoparticle benchmark artifacts

Current work uses **12 CPML cells per face at every resolution**:

- [Four-hour engineering log](../cuda-autoresearch-2026-11-11.md): corrected
  controls, correctness regressions, candidate decisions and measurement limits.
- [CPML12 baseline](cpml12-baseline.md): corrected nanoparticle timings.
- [Custom tracing](custom-tracing.md): implemented compiler/proofs, synthetic
  evidence, reproduction commands and the next scheduling architecture.
- [Direct native core validation](direct-native-core-validation.md): frozen
  CUDA source, independent oracle, guards and sanitizer scope.

Generated JSON, charts, probe sources and logs listed here are local artifacts
under `.cache/cuda-autoresearch-2026-11-11/reports/`; raw run/build directories
remain under the same dated cache. This follows the repository evidence-index
convention. Narrative summaries and reproduction tools remain in source.

## Historical first-pass artifacts

See the [engineering plan](../cuda-tracing-architecture-plan-2026-11-11.md)
for interpretation, next experiments and acceptance gates. The reports listed
below describe the earlier first pass, before the four-hour optimization run.

The benchmark convention has since been corrected to **12 CPML cells per face
at every resolution**. Existing measured reports below used the earlier 80 nm
convention and remain unchanged for provenance. They must be rerun before being
used as the baseline for future optimization. The runner now enforces CPML12;
see the revised plan above for the next steps.

- `mie-sweep-plan.json`: all five requested resolutions, physical parameters,
  timesteps and a six-field memory lower bound; no allocation or timing claim.
- `mie-cached-component-baseline.json`: six complete 30 fs measurements at
  5/4/2.5 nm using the available cached ABI-30 component. Every result records
  its source, benchmark and binary hash separately.
- `mie-source-build-baseline.json`: the authoritative six-case replay using
  the precise SM86 component freshly built from this checkout. All cases pass;
  all three mstep comparisons have zero measured maximum leaf-relative error.
- `mie-large-source-build.json`: additional 1280 nm, 256-cubed control at
  5 nm, with unchanged physical PML/sphere/duration. Fresh native build,
  three executable samples, no public-call samples; both schedules pass.
- `mie-baseline.html`: offline chart of the physical nanoparticle sweep.
  Uses the fresh-source report. Only successful measured points are shown;
  selectable timing metrics.
- `mie-geometry.json`: eligible mstep cells with and without source exclusions.
- `mie-cached-profile-kernel-shares.json`, `mie-cached-profile-status.json`,
  `mie-cached-perfetto.json.gz`: separate warmed 5 nm mstep profiling replay.
  Event-duration shares are not unprofiled wall-time shares. The compressed
  trace is inspectable in Perfetto; its original temporary path is metadata.
- `native-build-info.json`: precise SM86 native component built from this
  checkout using CMake/Ninja and CUDA 13.3; independent from the cached wheel.
- `cell-trace-depth4.json`: deterministic symbolic Yee cell graph, generated
  arithmetic, and exact depth-four initial-field dependency bounds/counts.
- `traced-rolling32-report.json`, `traced-rolling32.cu`: generated synthetic
  common-extent periodic CUDA probe, seven samples, with an independent scalar
  C++ solver. Its normalized coefficients, lack of PML/source/monitor effects,
  and host-driven submission make it a compiler experiment rather than a
  production throughput claim. Depth zero means the unblocked sweep control.
- `traced-memcheck.txt`, `traced-racecheck.txt`, `traced-synccheck.txt`: successful
  Compute Sanitizer checks of a small ragged periodic probe. Instrumented
  runtimes in these logs are excluded from performance interpretation.

Raw reports contain local paths for exact provenance; binaries and compiler
build directories stay outside versioned source. Use the commands in the plan
to regenerate artifacts on another machine. The full 2 and 1 nm resolution
cases are configured but unmeasured in this historical first pass.
