# Merging interior and CPML work within two-step tiles

Investigation stopped at the user's request. The tested merged-y prototypes improve the complete CPML pair
but have not beaten the ordinary cyclic-storage schedule or established the
>=9 logical GCUPS target. FP32 and ordinary stepping remain the defaults.
All native prototypes in this follow-up are isolated under `.cache/perf/`.

## Evidence so far

Every timing in the initial studies below is the mean of two fresh-process medians. Each process
uses four warmups and nine timed 256-step executions of the identical physical
1024x256x64 simulation, storage order 120, lossless binary material, FP32 fields
and CPML state, 12-cell CPML, one mode source, two compact mode monitors and
three frequencies. Conversions are timed; compilation/setup are excluded.
Variant order is balanced and reversed. Between-study drift is visible, so
compare against the controls in the same row.

| Experiment | Ordinary control | Retained CPML pair | Prototype |
|---|---:|---:|---:|
| One tile across all stored y | 8.311 | 6.847 | 6.008 |
| Two balanced y halves | 8.502 | 7.070 | 7.733 |
| Two halves with three-plane H ring | 8.320 | 6.948 | 7.662 |

The whole-y tile uses 15x66x16 cells, 1,024 threads and 96,720 dynamic shared bytes.
Its shared psi rows are compacted to the 24 physical absorber coordinates.
It combines the former interior and y-face owners, but only one block fits per
SM. It is slower and is not a candidate for production. Its trace attributes
72.7% of device-kernel time to the merged kernel. The reported 66.7% occupancy
is a launch-resource estimate, not a measured achieved-occupancy counter.

The half-y version uses 15x33x16 cells and 672 threads. Each tile contains at most
one absorber face, permitting 12 shared psi rows, 49,608 shared bytes and two
resident blocks on SM86. It improves the paired control by 9.4%, but remains
below ordinary stepping. It currently applies only to max staggered stored-y
extents 63..66 and otherwise falls back to the retained oriented pair.

The next prototype reuses a three-plane H ring for H1 and H2, retaining two E1
planes. During each wave, E1(N) consumes H1(N-1) before H2(N-1) overwrites that
cell. E2 then consumes H2(N-1) and H2(N-2); H1(N+1) can reuse the dead N-2 plane
on the next wave. XY offsets preserve physical coordinates. This permits a
16x33x16 tile with 704 threads and 46,848 shared bytes. The measured speedup over
the simpler half-y layout is not established. Dense FP32 reports a 24-byte
compiler stack frame; the packed FP32 variant has none.

Longer z sweeps preserve this shared footprint but did not help:

| Same-build variant | GCUPS |
|---|---:|
| Ordinary | 8.628 |
| Merged y, z16 | 7.829 |
| Merged y, z32 | 7.755 |
| Merged y, z64 | 7.424 |

Less halo work alone is insufficient. These measurements do not identify the
exact hardware stall mix; performance counters are unavailable.

## Correctness and reproducibility

Every large-run final state leaf matches its physical ordinary reference
bitwise. Seeded exact-state tests passed for whole-y (24 cases), half-y (48),
three-plane-ring (48), and longer z sweeps (24). They cover FP32/BF16 psi,
lossless binary/smooth materials, mode and ragged field monitors, and 33-step
runs plus two-step continuation. The half-y studies cover widths 63/64/65;
long-z tests span multiple 64-cell z tiles and partial tails. The whole-y and
half-y studies also cover source-only and source-free graphs. Each prototype
passed native memory checking and race checking with zero errors/hazards.
These sanitizer fixtures do not certify the whole Python stack.

Raw timings, complete-state hashes, binary/source fingerprints, candidate
patches against the retained source and test/sanitizer logs are in:

- [whole-y study](rtx3090-2026-09-18-cpml-full-y/)
- [half-y study](rtx3090-2026-09-18-cpml-half-y/)
- [three-plane H ring](rtx3090-2026-09-18-cpml-half-y-ring/)
- [longer z sweeps](rtx3090-2026-09-18-cpml-half-y-long-z/)

No power setting was changed and nothing was pushed. The subsequent edge/corner trial passed 70 seeded checks and native memory/race
checking, but produced no reliable throughput gain. Its matched means were:
ordinary 8.356, previous merged tile 7.568, enlarged edges 7.465, enlarged edges
with row padding32 7.395, and with row padding64 7.224 GCUPS. All ten large-run
states matched exactly. These edge and padding choices are not retained.
[Edge/padding evidence](rtx3090-2026-09-18-cpml-oriented-edges/) includes patches,
raw samples and validation logs. The completed general repeated-y study below uses the retained edges.


## Final general-domain comparison

The generalized merged kernel repeats 33-cell y tiles and uses the three-plane
H ring. It applies to 12-cell CPML when all staggered y extents are at least 46
and the maximum is at least 47; smaller domains fall back to the retained pair.
This study used seven variants in 14 balanced timing rounds, three warmups per
variant and 256 steps per execution. Each case was compared within one process.
All cases used FP32, CPML12, a mode source, two compact mode monitors and three
frequencies. Storage conversions are included; setup and compilation are not.
Dimensions below follow benchmark order (z, y, x).

| Physical domain / material / storage axes | Best ordinary | Retained CPML pair | General merged pair |
|---|---:|---:|---:|
| 1024x256x64 / binary / 120 | 8.666 | 7.036 | 7.751 |
| 128x256x512 / binary / 012 | 8.734 | 7.614 | 7.965 |
| 257x193x341 / smooth / 012 | 8.290 | 7.248 | 7.345 |

Values are logical GCUPS. The best ordinary variant was the queue schedule for
the narrow and irregular cases and automatic fusion for the wide case. Input
alignment and row padding did not improve the merged kernel. Every one of the
21 variant/case final states matched its reference bitwise. The generalized
candidate passed 64 seeded tests, 32 padded-layout tests and native memory/race
checks with zero reported errors/hazards.
[Raw general-domain results](rtx3090-2026-09-18-cpml-general-y-aligned/).

A trace of the merged half-y ring attributed 64.4% of device-kernel time to the
merged interior/y-face kernel and 12.8% to yz edges; compact monitors were about
0.3%. Shared-memory/register limits and halo work remain optimization targets,
but the exact hardware stall mix has not been established.

The final private candidate adds an interior-only path to middle y tiles and
changes dense FP32 thread count. It passed 96 seeded checks (64 general plus 32
padded), but was neither benchmarked nor run under the sanitizer before the
stop request. Its sanitizer executable was only compiled. Dense FP32 still
reports a 24-byte stack frame. No performance improvement is claimed for it.
Its source and logs remain under `.cache/perf/cpml-general-y-fast-*`.

None of this follow-up's private prototypes was promoted to production. The
installed extension was restored atomically to the retained y-face build and
its SHA256 was verified against that build. FP32 and ordinary stepping remain
defaults. Cyclic storage remains opt-in. Consistent >=9 GCUPS is not achieved.
No further experiments were run after the stop request; nothing was pushed.
