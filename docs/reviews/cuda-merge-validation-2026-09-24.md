# CUDA PR #224 merge validation — 2026-09-24

Merged main `639a1127` into PR head `88142066`; solver merge commit `085387d3`.
Main advanced to `aa26e533` during testing with a `CONTRIBUTING.md`
acknowledgment and CPML sharding tests; both updates were also merged. Its solver
and notebook sources are identical to the tested main revision.

Validation used an RTX 3090, freshly built native extensions, and isolated
checkouts. The user's existing working trees and uncommitted edits were preserved.

## Merge resolution

Preserved main's sharded CUDA target, launch geometry, and hardware tests alongside
PR #224's streaming graph attributes, storage layouts, and performance tests.
Bumped the combined native contract to ABI 19 / component 0.19.0 and regenerated
both bindings, so pre-merge extensions are rejected. Automatic layout selection
now excludes sharded programs; the eligibility regression test covers that gate.

## Performance

No material throughput regression against the pre-merge PR was observed: changes
are −0.03% to −0.05%. Every final-state leaf matches the pre-merge PR bitwise
in every repeated case, including CPML state and monitor accumulators.

| Domain (z,y,x) | Pre-merge GCUPS | Merged GCUPS | Change |
|---|---:|---:|---:|
| 1126 × 297 × 65 | 8.639 | 8.636 | -0.03% |
| 104 × 1083 × 180 | 7.089 | 7.086 | -0.05% |
| 110 × 202 × 866 | 8.707 | 8.703 | -0.05% |
| 325 × 203 × 312 | 8.341 | 8.337 | -0.05% |

These are warm executable rates, averaged from two process medians per revision.
They do not measure startup latency or establish performance on other hardware.

Against current main, the merged PR is **6.2–18.3% faster** in these cases.

| Domain (z,y,x) | Main GCUPS | Merged GCUPS | Change | CPML entries outside tolerance per run |
|---|---:|---:|---:|---:|
| 1126 × 297 × 65 | 7.315 | 8.634 | +18.03% | 16 |
| 104 × 1083 × 180 | 6.071 | 7.084 | +16.69% | 716 |
| 110 × 202 × 866 | 8.190 | 8.701 | +6.24% | 476 |
| 325 × 203 × 312 | 7.047 | 8.337 | +18.29% | 0 |

**The existing CPML parity blocker against main remains.** Fields and monitors pass
in all cases, but CPML auxiliary-state failures reproduce the original report's
16 / 716 / 476 / 0 counts. The benchmark tolerance remains exactly
`max(3e-6, 1e-6 * reference_leaf_peak) + 3e-5 * abs(reference)`.
No tolerances were relaxed. Bitwise agreement against the pre-merge PR establishes
that the merge did not introduce these differences; it does not establish complete
numerical equivalence with main.

Both 16-process comparisons completed, verified identical initial states,
coefficients, and source arrays, and checked that solver sources did not change.
Peak sampled GPU temperature was 71°C, minimum free VRAM was 15,049 MiB, and the
370 W power limit was unchanged. Desktop processes were left running.

## Modal sources notebook

Executed every original code cell in `examples/notebooks/modal_sources_monitors.ipynb`
on current main, the original PR, and the merged solver. Explicitly selected
`cuda_streamed`, disabled docs test mode, and used the notebook's full settings:
17 monitor frequencies, three candidate modes, seven broadband source profiles,
and 12,605 steps per simulation. The straight-waveguide grid is 180 × 77 × 56
in notebook (x,y,z) order. Ran the single-profile, broadband, and junction cases.

An appended export cell collects **unnormalized** flux, complex modal amplitudes,
mode-monitor flux, and complex Ey samples for all three simulations, plus effective
indices and source/monitor frequencies: 15 arrays total. This avoids accepting
normalization-to-one as evidence of equivalence. Comparison uses
`abs(error) <= 1e-6 * reference_array_peak + 3e-5 * abs(reference)`;
no unit-dependent absolute floor is applied to the raw spectra.

**All 15 exported arrays are bitwise identical to the original PR.** Against
current main, all 15 pass the stated tolerance; the maximum peak-relative error
is 2.0524e-6 (broadband Ey). The maximum flux error is 1.5445e-7 of the relevant
spectrum's peak, and the maximum modal-amplitude error is 1.1224e-7 of its peak.
Effective indices and frequency arrays match main bitwise. Every notebook
assertion passed. These results cover this full tutorial, not every simulation.

Executed notebooks and compressed raw arrays are retained locally under the
validation checkout's `.cache/pr224/notebook-{before,merged,main-final}/`.
The tracked JSON summaries contain provenance and per-array errors.

## Other checks and limits

- CPU unit, contract, and kernel suites: **1,112 passed, 267 skipped**. Optional
  mesh/GDS packages are absent in this environment; skips are not passing evidence.
- Latest-main CPML sharding additions: **9 passed** (six unit cases and
  three simulated multi-device CPU cases).
- Focused CUDA/sharding/backend/CPML CPU suite: **106 passed**; the updated layout
  eligibility suite was also rerun (**22 passed**).
- CUDA hardware suite: **382 passed, 26 skipped** in 912.65 seconds. All 49
  storage-layout cases, all six tuning cases, and 327 backend cases passed.
  The 24 multi-GPU cases and two H100 cases were skipped for unavailable hardware.
- Full Ruff lint/format, Vulture, ABI regeneration check, and simulation-package
  Pyright passed.
- Full-project Pyright reports the same seven GDSFactory attribute errors in
  unchanged `beamz/design/gds.py` recorded by the original PR.
- Only one GPU is available. Multi-GPU CUDA hardware and H100 performance remain
  unvalidated; the native CPU sharding contracts are covered by the CPU suite.

The first exploratory notebook run used a stale raster binary and was discarded.
After rebuilding, the first export attempt used an incorrect field accessor;
all notebook cells passed, and the complete notebook was rerun with the corrected
exporter. Only the successful runs with rebuilt native code are used here.

## Reproduction and evidence

Use the same Python/JAX environment for all checkouts, with each revision's own
native extensions. Build Release with fast math disabled. The measurements use
CUDA 13.3.73 / GNU 16.1.1; the GPU executes SM86 code. Main's portable architecture
list was retained; the PR and merged builds targeted SM86.

```sh
python scripts/validate_modal_notebook.py --root /path/to/checkout --output /fresh/notebook-output
python scripts/benchmark_cuda_random_comparison.py \
  --main-root /path/to/reference-checkout \
  --output /fresh/benchmark-output --reference-dir /fresh/reference-states \
  --build-description 'Describe the actual native builds here'
```

`--main-root` can point to either current main or the pre-merge PR. The runner
labels that reference `main` and the executing checkout `branch`. It retains
four warmups, nine synchronized 256-step samples per fresh process, and
reference/merged/merged/reference order for each of the original four shapes.
Timing excludes setup/JIT/mode solving and includes internal storage conversion.
All initial states, material coefficients, and source arrays must match exactly.

[Environment and extension hashes](pr224-merge-2026-09-24/environment.json).
[Machine-readable comparison](pr224-merge-2026-09-24/summary.json).
[CPU and static-check logs](pr224-merge-2026-09-24/).
[Pre-merge comparison records](rtx3090-2026-09-24-merge-before/).
[Current-main comparison records](rtx3090-2026-09-24-merge-main/).

## CI cleanup before merging

The first GitHub run exposed three CI integration issues: the optional native
module was imported statically in the tuning identity helper, hardware and unit
tuning files shared an unqualified pytest module name, and eight review-document
links pointed outside the MkDocs documentation tree. The helper now uses the
same dynamic import mechanism as backend discovery; the hardware test file has
a unique name; source links are pinned to the validated GitHub revision.

After these changes, full test collection succeeds, all 22 tuning unit tests and
six tuning hardware tests pass, and both executable documentation tests pass.
Full Ruff, Pyright, Vulture, and the strict MkDocs build pass in an isolated
environment with the CI dependencies installed. The seven GDSFactory type errors
from the earlier GPU environment do not occur with those dependencies installed.
These changes do not alter the CUDA kernels, default geometry selection, or
notebook calculations measured above.

The complete CPU evidence suite with CI dependencies passed: **1,822 passed,
1 skipped, 1 xfailed**. Its changed-line coverage exposed a gap in CPU coverage
of the storage adapter and explicit calibration control flow. Eighteen new CPU
cases now check geometry, packed material tails, ragged monitor lanes, source
coordinates, state restoration, invalid calibration results, balanced timing,
cache reuse and invalidation, and headroom aborts. Device compilation and timing
are deterministic test doubles in the calibration tests; these complement the
actual CUDA parity tests above.

All **40 focused storage/tuning tests pass**, including the existing 22 tuning
cases. Combined with the complete evidence run, changed-line coverage is
**90% (522/575 lines)** against the required 85%, storage adapter statement
coverage is 100%, and overall statement/branch coverage is **85.58%**. All six
coverage policy floors pass. Ruff, Pyright, and Vulture also pass. No production
numerical code changed for this coverage improvement. Final GitHub CI independently
reruns the complete suite. The latest main update only adds documentation diff
attributes in `.gitattributes`.

[Focused CPU test log](pr224-merge-2026-09-24/cpu-calibration-storage-tests.log).
[Changed-line coverage](pr224-merge-2026-09-24/changed-line-coverage.log).
