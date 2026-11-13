# Plane-retirement experiment plan (implemented and tested)

Status, Nov12–13 UTC: the two-hour continuation implemented this design,
proved and tested both 36- and 18-barrier consumers, and measured complete
512³ prefixes. Both are correct and compile without stack/spills, but are
slower than the previous temporal path. The selected improvement instead uses
support and NoIO material specialization. See the
[completed study](cuda-plane-retirement-study-2026-11-12.md) for measurements,
proof scope, rejected alternatives and the next architectural constraints.
The original feasibility plan follows for comparison with the actual result.

The four-hour run produced a correct native temporal compiler and a measurable
large-domain crossover. It did not establish several-fold acceleration. The next
experiment should reduce the live field frontier, rather than repeat larger
register arrays: the tested full-paper FP32 layouts spilled despite constant
coefficients and full-cache support proofs.

At the time of this plan, the design was unimplemented. Its resource arithmetic
below is a feasibility argument; the linked study contains the subsequently
completed lifetime proof, actual resource diagnostics and performance result.

## A concrete storage target on RTX3090

Use cached ZYX16×16×32, owned15×15×31,512 threads. Each thread owns one XY
position and16 fixed Z slots. Keep only H permanently in private registers:
3×16=48 floats/thread. Store one complete three-component E/H exchange frame
in shared memory:3×8192×4=98,304 bytes. A three-component225-point X-face stage
needs2,700 bytes; frontier bases and a phase word need68 bytes. Total101,072
bytes, below the101,376-byte per-block opt-in limit (the actual driver must
confirm available dynamic shared memory/reservation). The time-window words
would add8 bytes.

Retire old E a plane at a time. Three small vectors hold the saved last plane,
the current old-E/result vector, and the previous new-E result. That targets
48+9=57 field registers/thread, instead of the96 field registers in the rejected
512-thread full-register experiment. This leaves a plausible metadata budget
under the128-register/block limit; compilation must verify it, with the actual
native source/CPML/observation environment. Register estimates do not prove fit.

## Candidate schedule and the dependency that makes it possible

H updates read old E and only their own old H. E updates read current H and
only their own old E. No E update reads a neighboring E. That last fact allows
old E to be saved only immediately before its shared plane is overwritten.

1. Compute all current owned H into private H, using the old E shared frame.
   Import current H ghosts before publishing H; old-E readers must retire first.
2. Save the old E values of the last cache plane and publish its current H.
   This seeds the wrapped negative-Z dependency needed by the first E plane.
3. For each Z plane, save its own old-E vector and publish current H for that
   plane. A barrier makes current H available to every XY neighbor.
4. Compute that plane's E, including unchanged CPML, injection and observation.
   A barrier retires H readers. Publish the preceding E result into the plane
   whose H is now dead; hold the current E result for the next iteration.
5. At the final plane, publish the remaining E results and restore current E
   ghosts. Publish the next version only after a final barrier.

Private H must remain intact through the whole E sweep: it is the next
iteration's old H. It cannot be discarded after its E plane consumes it.
The last shared H plane must likewise survive until the last E plane computes,
even though the first E plane has already used it as its negative-Z neighbor.
These are essential constraints for the independent token proof.

Current E ghost values need not occupy permanent private slots. E consumes only
owned old-E centers. Its current ghosts can be reconstructed from the immutable
upwind trails after H readers retire. The2,700-byte stage may first batch H
imports, then batch E ghosts after a barrier has retired every H-stage reader.
The exact placement needs a cross-warp lifetime proof; same-thread reasoning
alone is insufficient.

This schedule adds per-plane CTA barriers. It trades synchronization and a
larger shared carveout for a much smaller private live frontier. The benchmark
must decide the tradeoff; no speedup should be inferred from storage fit alone.
The shared allocation admits only one512-thread CTA, or16 active warps, versus
32 warps in the current Fast1024-thread consumer. Less redundant boundary work
can therefore come with less latency hiding; this is a central risk to measure.

## What the trace compiler should generate

The existing scalar operator should remain the physics source. Extend scheduling
and storage allocation around it:

- Derive old-E last-use and current-H last-use edges from the actual Yee reads.
- Bind each logical field version to private H, the shared family frame, or one
  of the three short E vectors. Emit copies only when a live edge crosses a
  storage lifetime.
- Preserve post-H/post-E injection and raw observation clocks, including all
  split multiply/add rounding. CPML state retains its proved physical owner.
- Reuse the verified support certificates, active time windows, immutable
  uniform-material classes and concurrent same-rank graph branches.
- Make shared-stage reuse and warp participation explicit proof obligations.

The steady interior field exchange for owned15×15×31 is exactly1,217 imported
and1,095 exported six-field points per update. At FP32 this is
`24*(1217+1095)/6975 = 7.9553 bytes/Yee-update`, before startup/final scatter,
materials, CPML, sources, DFT, padding and memory transaction overhead. The
current owned7×15×31 geometry uses11.6203 bytes on the same logical accounting.
Neither number is measured DRAM bandwidth or an end-to-end speedup prediction.

## Stop gates before promotion

First prove slot/version/last-use ownership with adversarial warp ordering and
negative controls. Then compile the actual consumer and reject unacceptable
spills. Run seeded and nonzero CPML cases, ragged walls, source/observer overlaps,
odd tails, late starts, split continuation, reset and replay against an ordinary
CUDA reference, comparing all30 leaves bytewise. Complete bounded memory, race
and synchronization checks. Only then run matched fresh-process large-domain
and full30fs Mie timing, with cold setup reported separately.

A successful mechanism can remain an opt-in research path until complete
workloads win. The same-method fallback must preserve the existing feature
coverage; smaller simulations should retain the measured D2 path.
