# Narrow-domain H/E fusion experiment

All trials use FP32 and 12-cell CPML. No production source or dispatch
change from these trials was retained; the established native binary has been
restored. The consistent >=9 logical GCUPS goal remains unmet.

## Question and method

After 40/48-column ordinary-core repacking failed to improve throughput, this
experiment tested field reuse through H/E fusion with tile dimensions that
better fit a narrow post-CPML interior. This is a one-step fused control,
not a new implementation of complete two-step CPML.

Three private native specializations replace the existing core-tile selectors:
`32x8x8` selects **20x12x16**, `64x4x8` selects **40x6x16**, and `32x4x8`
selects **16x16x16**. Those selector aliases apply only to the isolated prototype;
they are not new public controls. Tile dimensions are x,y,z, whereas domain
shapes below use z,y,x. Each fused variant uses the existing 32x8 CPML shell.
The ordinary controls disable core fusion and select either 64x4 or 32x8 shells.

Each case uses the same input state, lossless material, solved mode source,
two compact mode monitors, three frequencies, and 256 synchronized timesteps.
Ten timing rounds use a balanced cyclic order followed by its reverse, after
four warmups per variant. Setup, compilation and correctness host copies are
excluded. Logical-cell accounting excludes halos and redundant work. The
large domains contain roughly 16.5–17.0 million logical cells.

The binary-material cases use (1024,256,63), (1024,256,64), and (1024,256,65).
The smooth-material case uses (1024,256,64), after 1,024 preconditioning steps.
Its first attempt failed before timing because the prototype omitted `presteps`
from the simulation-construction options. That setup error was fixed and the
case rerun; the failed log remains alongside the completed measurements.

## Fusion results

| Case | Ordinary 64x4 | Ordinary 32x8 | Fused 20x12x16 | Fused 40x6x16 | Fused 16x16x16 |
|---|---:|---:|---:|---:|---:|
| narrow63 | 6.777 | 7.061 | 6.130 | 6.535 | 5.094 |
| narrow64 | 6.630 | 6.909 | 6.189 | 6.571 | 5.441 |
| narrow65 | 6.577 | 6.977 | 5.273 | 6.043 | 5.530 |
| developed_smooth | 6.509 | 6.901 | 6.105 | 6.675 | 5.489 |

All values are median logical GCUPS. None of the fusion tiles beats the ordinary
32x8 shell control. The shell control gains approximately 4–6% over 64x4 in
these cases. This reinforces a measured shell benefit but does not establish
an automatic rule for all arbitrary widths or close the remaining gap.

## Sparse-source check experiment

The older one-step fused kernel checks source bounds per staged H component.
The two-step implementation already rejects nonintersecting source regions
once per tile. A second private build applies that strategy to the one-step
fused kernel, conservatively including its backward halo. It changes neither
source chronology nor accumulation order. The best 40x6x16 specialization uses
39 registers/thread versus 35 without this check, and adds 16 bytes of static
shared memory; neither build reports a stack or local-memory allocation. These
are compiler resource counts, not measured occupancy or stall counters.

The repeated narrow64 comparison produced:

| Ordinary 64x4 | Ordinary 32x8 | Fused 20x12x16 | Fused 40x6x16 | Fused 16x16x16 |
|---:|---:|---:|---:|---:|
| 6.626 | 6.986 | 6.075 | 6.607 | 5.322 |

The source check does not rescue fusion relative to its same-run ordinary
control. Its absolute values versus the earlier build are from separate runs;
they should not be interpreted as a precise isolated source-check speedup.

## Verification and disposition

- The first build passed 27 seeded hardware cases: three core tiles, widths
  63/64/65, all three shell layouts, binary/smooth lossless materials, mode
  sources, field monitors, 33 steps and two-step continuation. Every CUDA state
  leaf matched the ordinary queue bitwise, with the existing JAX tolerance.
- The source-check build passed the same 27 cases and six rotated-source cases
  covering all three source normals and both material layouts. The rotated
  checks use the first prototype core tile; the other two have the seeded
  width/continuation coverage above.
- Every large timed variant matched its case's ordinary control bitwise for
  the complete final state, including CPML and monitor accumulators.
- No sanitizer or full-release validation was added for these rejected trials.
- Maximum observed GPU temperature was 77.0 C;
  minimum free memory was 14868.0 MiB. Power settings were unchanged.

The next meaningful experiment is an internal storage-axis permutation that
keeps the physical simulation identical while giving the update kernels a
longer contiguous dimension. Earlier physically rotated setups are suggestive,
but do not validate that transformation. It must preserve staggered components,
CPML term ownership, material packing, source placement and all monitor outputs.
Padding or smaller tile dimensions alone have not supplied the required gain.

Raw timing samples, native hashes, exact-state digests, telemetry, the probe,
and the two prototype patches are in
[the fusion directory](rtx3090-2026-09-18-narrow-fusion/) and
[the source-check directory](rtx3090-2026-09-18-narrow-fusion-source-cull/).
The tested first binary and source checkpoint remain under
`.cache/perf/cpml-narrow-fusion-v1/`; the second is in
`.cache/perf/cpml-narrow-fusion-build/` and its adjacent source directory.
