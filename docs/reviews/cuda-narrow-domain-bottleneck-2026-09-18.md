# Why 1024x256x64 is slower, and the route toward 8.4 GCUPS

Update: [same-physics cyclic storage](cuda-cyclic-storage-study-2026-09-18.md)
is now implemented as an explicit opt-in. A controlled narrow-domain run
measured **6.75 -> 8.38 GCUPS** with fusion disabled in both layouts and every
state value identical. Automatic-fusion repeats measured roughly 8.2–8.3.
The implementation preserves FP32 and 12-cell CPML. Consistent >=9 GCUPS and
a winning complete-CPML temporal schedule remain unachieved. The investigation
below records the evidence leading to this result.

The benchmark ordering is z,y,x; x is the contiguous storage dimension.
The established FP32 CUDA schedule measures about 6.8 GCUPS for this case.
The evidence identifies extra boundary work and inefficient short-x execution.
It does not establish the exact hardware stall mechanism: performance counters
are unavailable, and the ordinary narrow-domain queue combines core and shell
work in one kernel, so its total time cannot be attributed solely to CPML.

## Confirmed implementation costs

1. CPML affects roughly 44.69% of cells at 1024x256x64, versus 29.82% at
   128x256x512. These fractions are `1-product(N-24)/product(N)`, ignoring small
   staggered-extent adjustments. More cells carry auxiliary recurrence work.
2. The 64-cell contiguous direction leaves 40 interior cells after 12 cells
   on each side. The ordinary core queue uses a 64-column mapping. Only
   approximately 40/64 mapped x slots perform interior updates. Its x-face
   strip has 24–25 columns, also mapped into a default 64-column shell tile.
   These are useful-slot ratios, not measured SM occupancy or bandwidth.
3. `FuseCpmlCore` rejects x widths below 384 in automatic dispatch. The narrow
   case therefore uses separate H/E phase queues; the wide case can use a
   shared-memory H/E interior. That threshold preserves measured performance:
   blindly forcing the existing fused tile on narrow grids is not a fix.
4. The earlier equal-volume axis-permutation study ranged from 6.80 GCUPS at
   1024x256x64 to 8.23 at 1024x64x256 and 8.54 at 256x64x1024. CPML volume is
   the same for these permutations. This supports an orientation/layout cost
   beyond the physical boundary fraction. Those were separate benchmark
   setups; a storage-only permutation of identical physics still needs testing.
5. Compact mode-source and monitor work is small in the relevant profiles.
   GPU utilization reached 100% in the large studies; insufficient total cell
   count does not explain this case's penalty.

## Focused next implementation

Reaching 8.4 from 6.8 requires 23.5% more throughput, equivalent to about 19.0%
less elapsed time. The existing shell-tile improvement of roughly 5% is useful
but insufficient.

First, test narrow-interior thread mappings and H/E fusion tiles sized for the
post-CPML width, rather than assigning 64 columns to a 40-cell interior. Keep
exactly 12 physical CPML cells and handle staggered tails explicitly. A
40-column core trial and a 32-wide shell trial directly address this case;
adaptive choices must subsequently cover widths 63/64/65 and other arbitrary
widths rather than hard-coding one benchmark. Revalidate source/monitor timing,
logical cell accounting and complete-state parity before using a new dispatch.

Second, test selecting an internal storage axis with a longer extent, while
preserving physical axes, vector components, material/source placement,
absorber profiles and monitor results. This is a larger change than padding:
current x-unit-stride assumptions occur throughout indexing and compiled IO.
It must be measured on an identical physical simulation, not inferred from
rotated benchmark setups. Simple padding alone did not recover the gap.

Continue evaluating the complete two-step CPML path separately. The retained
384-thread x-face refinement improved that experimental narrow case from
5.34 to 5.83 GCUPS, but does not improve the 6.8-GCUPS default path. Its progress
must not be counted as closing the default-path gap. The final >=9 GCUPS goal
remains unmet.

## Narrow-core mapping experiment

A follow-up trial changes only the ordinary core queue's thread-to-cell mapping:
40 columns for post-CPML interior widths 33–40, 48 for widths 41–48, and the
established 64-column mapping otherwise. Partial final thread rows are masked
explicitly to preserve unique write ownership. The shell remains 64x4, and
neither arithmetic nor H/E scheduling changes. This isolates the potential
benefit of reducing unused core thread slots; it does not reduce the amount
of field data that H and E exchange through global memory.

Nine seeded hardware checks passed for physical widths 63/64/65, all three
shell mappings, a mode source, two field monitors, 33 steps and two-step
continuation. The ordinary queue matched the fused control bitwise for every
state leaf, and matched JAX within the existing tolerance.

The performance comparison uses frozen baseline/candidate native binaries,
fresh processes in baseline/candidate/candidate/baseline order, four warmups,
nine timed samples per process and 256 steps. Each large case has FP32
12-cell CPML, binary lossless material, a mode source and two three-frequency
compact mode monitors. Final complete-state hashes are compared across binaries.
Raw measurements and binary fingerprints are in
[the narrow-core experiment directory](rtx3090-2026-09-18-cpml-narrow-core/).

Results below are the mean of the two fresh-process GCUPS medians for each
binary, not a pooled confidence interval:

| Shape (z,y,x) | Established queue | Narrow-core trial | Change |
|---|---:|---:|---:|
| 1024x256x63 | 6.711 | 6.752 | +0.61% |
| 1024x256x64 | 6.746 | 6.743 | -0.06% |
| 1024x256x65 | 6.672 | 6.597 | -1.14% |
| 128x256x512 | 8.419 | 8.428 | +0.10% |

All 16 runs had identical final complete-state hashes within each shape.
The x=64 baseline process medians themselves ranged from 6.643 to 6.850, so
small changes should not be treated as reliable improvements. Peak GPU
temperature was 72 C and minimum free memory was 18,940 MiB; power settings
were unchanged. The trial offers no useful measured speedup and was reverted
from production source and the installed native binary. Its source and binary
remain under `.cache/perf/cpml-narrow-core/`; the nine correctness tests remain.

This result weakens the idea that eliminating masked core slots alone will
close the gap. It leaves reducing H/E field traffic through a narrow-specific
fusion kernel, improving the CPML shell mapping, and testing storage-axis
selection as the concrete next experiments. Neither the exact hardware stall
mix nor an 8.4-GCUPS narrow-domain implementation has been established.

## Follow-up: narrow-specific fusion

The subsequent [fusion study](cuda-narrow-fusion-study-2026-09-18.md) tested
20x12x16, 40x6x16 and 16x16x16 H/E tiles on widths 63/64/65 and a developed
smooth-material case. None beat the ordinary 32x8 shell. A further per-tile
source rejection check passed correctness but did not rescue performance.
Both prototypes were left out of production. Internal storage-axis selection
is now the stronger next experiment; the same-physics validation remains to
be implemented.
