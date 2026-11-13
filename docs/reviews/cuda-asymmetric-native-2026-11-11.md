# Complete asymmetric moving-cache CUDA consumer

This is an opt-in research port, with no production dispatcher or installed
binary change. Its current source is `scripts/cuda_asymmetric_native.cuh`;
`scripts/patch_cuda_native_moving.py` assembles it into an immutable native build.
The native session and workspace implementation live in `cuda/src/native_moving.*`
and the two `scripts/cuda_native_moving_*.inc` templates.

The kernel owns a transformed 5 × 15 × 31 volume and retains a 6 × 16 × 32 cache.
Hx/Hz/Ex/Ez stay in permanent shared storage; Hy/Ey stay in fixed thread-local
register slots. Each version first advances H, imports the current six-field low
boundary, then advances E. Three CTA barriers enforce the old-E, import, and
current-E/publication lifetimes. A 600-byte temporary batches the low-X imports
into contiguous logical points. Its private Hy/Ey values are retired after the
import barrier. This follows the numerically and sanitizer-validated field-only
low-X schedule; it does not use the rejected two-barrier early-import variant.

The point update is generated from the cell IR, rather than recoded in the
consumer: `scripts/asymmetric_native_cell_generated.cuh` implements
`AdvanceNativeCell<Phase, Component>`. The environment supplies cached reads,
canonical native boundary differences, packed coefficients, local CPML state,
and post-update injection. CPML psi uses ordinary coherent global loads/stores
and remains in place. The local-term IDs are H0…H5 followed by E0…E5. Profile and
psi operations occur only when the generated active-term predicate succeeds.

Each rank runs a source/CPML/monitor/boundary-free bulk specialization followed
by a general specialization. Same-rank tiles are independent. Four frontier
slots retain the three preceding ranks; epoch-qualified owner tags reject stale
producers across macroblocks, reset, and replay. Epoch wrap sets an error.
Startup field loads retain actual staggered support clipping in both classes.

A macroblock uses local versions 1…depth. Sources use the frozen native chunk
clock plus `macro_offset + version - 1`; raw sparse monitor samples use row
`version - 1`. The native runner reduces those samples with the unchanged DFT
kernels in chronological order at the same chunk-local offsets. It advances
clocks once at the chunk boundary. Remainders use ordinary native steps at their
explicit offsets, without restarting the DFT phase at a macro boundary.

Initial admission is deliberately narrow: depths 4 and 16; Cartesian FP32;
uniform 12-cell CPML; scalar H; homogeneous byte-packed or block-packed lossless
E; canonical staggered fields with dense pitches; no PEC, tensor material, or
pre-E injection.
Nonempty source groups must be coincident multi-source groups with matching
starts, so the split multiply/add rounding matches the existing IO kernels.
Field input/final banks must be distinct; psi input/output pointers must alias.
The runner captures coefficient/source/observation class facts privately at
upload. Unsupported packets fail before capture/allocation where possible.

Input/final aliasing is not enabled. Although the moving startup/final geometry
has an independent proof in `scripts/prove_cuda_moving_inplace.py`, a future
alias mode must also remove incompatible pointer promises, account for any late
initial observer, and avoid allocating the unused final bank. Rebinding pointers
alone would not establish a capacity improvement.

The first full-physics binary has passed the initial numerical gates below.
Further performance paths remain experimental. Field-only speedups and the generated functor's independent host-oracle checks
are not full-engine performance or race evidence. Promotion requires exact all
30 state leaves (fields, psi, DFT, clocks), seeded/nonzero starts, reset/replay,
odd continuations, material/source class controls, and bounded CUDA memory,
race, and synchronization checks, followed by matched fresh-process timing
against the strongest admitted D2 baseline.

## First complete-physics gate

The immutable `moving-native-v3` binary (SHA256
`5e3d0d025dc5ecec5fddcb3324752bf4f2186923fde46157099350df5b3ad4ed`)
passes the first complete-state campaign. This includes seeded Mie and
rectangular cases at depth four, 35-step and 259-step cases at depth sixteen,
reset/repeated preserving and owned calls, split/odd continuations, changed
sources/materials/monitor indices, and nonzero clock origins. References use
the independent ordinary native Program path for changed operational metadata,
and exported independent JAX references where available. The first stale
changed-monitor reference is retained explicitly; changing gather indices while
reusing the old D2 observation dependency plan was not a valid oracle.

The exact records and command provenance are under
`.cache/cuda-engine-autoresearch-2026-11-11-2032/moving-native-v3/`, especially
`small-gates-summary.json`. These numerical gates do not substitute for the
pending device memory/race/synchronization checks or matched performance.

All sixteen depth/layout/coefficient-class/bulk specializations compile without
stack or register spills. Bulk uses 59–64 registers; generic uses 82. Static text
is substantial: bulk has approximately 4,900 instruction sites, generic 14,800,
from fixed-slot unrolling. These are static counts, not executed instructions or
an instruction-cache-miss measurement. `asymmetric-resources.json` and
`asymmetric-sass.json` preserve the detailed evidence. Further cache expansion
must account for this code-size cost as well as frontier bytes and occupancy.

## First performance diagnosis

A bounded same-module preserving screen with matched 256-step chunks found no
speedup: guide512 was 8.371 s versus 5.611 s for D2 segment64, guide256 was
2.113 s versus 0.918 s, and Mie272 was 1.903 s versus 0.660 s. These are initial
screens, not balanced confidence estimates. Full-state and repeated-call parity
passed. CUPTI attributed guide512 5.165 s to the general body and 3.188 s to
bulk; Mie272 was 1.659 s general and 0.224 s bulk. DFT and epoch kernels were
small. A field-only improvement did not transfer automatically to the complete
operator.

The separate scalar-queue candidate is reproduced with
`scripts/patch_cuda_native_scalar_queue.py`. It rotates both private families
after every physical Z-slot iteration, including excluded/invalid slots. Six
rotations restore physical orientation before imports and publication. The
independent proof and source review are in
`scripts/prove_cuda_native_scalar_queue.py` and its focused tests. The candidate
retains the same arithmetic and barriers and passes the complete numerical
gates. Its generic static text falls from approximately 14,800 to 5,760 sites,
while registers rise from 82 to 112 without spills. Bulk remains 63 registers.
Initial screens (guide512 8.436 s, Mie272 1.884 s) show no clear performance win.
This does not isolate instruction-cache effects: rotations and address liveness
also changed. It is not selected over v3.

The next isolated lowering is emitted by
`scripts/patch_cuda_native_import_packets.py`. Under canonical staggered
admission, it forms one six-bit support mask and one transformed
neighbor/surface/base for six trail loads. The bulk import mask is constant63
only after the strict existing support certificate; initial cache clipping is
unchanged. Invalid components remain positive zero, and a missing producer sets
the diagnostic only when some component is valid. No extra persistent state,
shared storage, workspace, or cell arithmetic is introduced. The initial v5
build uses v3 as parent; independent source/support proofs precede GPU gates.

## Isolated import and policy results

The import-packet v5 binary passes 52 seeded small-state cases plus the complete
first/repeat large-case comparisons. Its initial same-module screens improve the
moving algorithm from 8.371 to 7.224 s on guide512 and from 1.903 to 1.705 s on
Mie272. Strong D2 remains faster at 5.611 and 0.650 s. Bulk text falls from 4,968
to 3,256 static instruction sites and its register use falls 63 to 61; generic
remains 82 registers without spills. Records are under `moving-import-packets-v5`.

Six separate IO policy classes preserve exact state but regress Mie: guide512
is 7.113 s, Mie272 3.236 s. The natural nonbulk register counts are 76 (NoIO),
79 (ObserveOnly), 72 (SourceOnly/BothIO), and 82 (full generic); all still occupy
one CTA. Splitting small same-rank cohorts can therefore add underfilled waves
without an occupancy gain. The parallel policy prototype retains specialized
bodies but forks disjoint same-rank classes onto persistent auxiliary streams,
then joins main before advancing rank. It is a separate experiment; source
review is not a timing or sanitizer pass. The native runtime owns stream/event
creation, graph cache identity, and cleanup, including errors before a join.

UniformE scalar placement has compile-only resource artifacts. Capturing three
common bit-exact E source values removes per-point code decoding, but natural
register counts do not fall (bulk 61 to 63, NoIO 76 to 77; other classes unchanged).
A single NoIO minimum-two-CTA diagnostic reaches 64 registers with a 16-byte
spill frame and 36/76 static spill-store/load bytes. These are static compiler
metrics, not measured DRAM traffic. No UniformE=true physics dispatcher exists
in those resource artifacts; actual use needs the independently checked host
tile certificates and numerical gates.

The live base consumer additionally saturates the operational epoch at
UINT32_MAX. Clearing a run diagnostic cannot revive wrapped generations. The
host-compiled test extracts the actual primitive and verifies saturation and
repeated failures; existing frozen performance artifacts retain their earlier
body for provenance. This changes no ordinary field or clock arithmetic.

## Bounded full-paper register-budget study

The larger 15 × 15 × 31 owned / 16 × 16 × 32 cache was tested as a field-only,
compile-first architecture, without changing the production solver. Warp halves
pair neighboring Z parities; X neighbors within a 16-lane half use shuffles and
cross-half Y-component neighbors use a narrow shared edge. Other components use
a two-component exchange frame. Physical slot stores and all-lane shuffle
participation were reviewed against an independent route proof.

All four resource variants were rejected before GPU execution:

| Variant | Threads | Private field scalars/thread | Registers | Spill frame | Static stores/loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guarded paired-Z | 1024 | 48 | 64 | 200 B | 1052 / 820 B |
| Full-cache Interior | 1024 | 48 | 64 | 184 B | 732 / 744 B |
| Two Y rows, Interior, low-only | 512 | 96 | 128 | 224 B | 1088 / 900 B |
| Same, speculative ghost H | 512 | 96 | 128 | 280 B | 1272 / 1108 B |

The Interior class erases update physical coordinates/validity, initial support
checks, import support checks, and final validity checks only under the complete
cache-envelope certificate. Its pre-allocation PTX has no local declarations or
local loads/stores: the field arrays are scalarized, so the final stack reflects
physical register pressure rather than an escaped dynamic array. Splitting the
same CTA state over fewer threads did not solve allocation. The final speculative
H experiment is legal only for pure field arithmetic: ghost H results are
unconditionally overwritten by imports before H publication; E retains its
owned predicate. It worsened this compiler result and was not pursued.

These static spill bytes are instruction-site totals, not measured DRAM bytes.
The copied resource binaries explicitly refuse ordinary execution because host
Interior partitioning is not integrated. Exact manifests are under
`.cache/autoresearch-4h-2026-11-11/register1024-zpairs-paper-v1`,
`register1024-zpairs-interior-resource-v2`,
`register512-zpairs-y2-interior-resource-v3`, and
`register512-zpairs-y2-speculative-h-resource-v4`. The lower-state eight-plane
cache remains the measured field-only architecture; this study establishes a
resource limit, not a simulation speedup.

The copied native7 active-time-window prototype prunes versions whose entire
owned box lies outside every canonical component support. For maximum support
extents M, origin O and owned extents B, a valid owned point necessarily has
`max(1, max(O-M+1)) <= n <= min(Depth, min(O+B-1))`. These are conservative union
bounds, so mixed staggered corners may still compute invalid points. Empty tiles
publish their epoch tag before returning; their unwritten trails cannot be read
by a physically valid import. For a late first version, the previous owned box
is completely invalid. Only its low ghost union is reconstructed, from strictly
lower-rank producers at the exact previous version, after a base-table barrier.
The original initial-bank startup remains for first=1 and the Fast body retains
the full-depth path. Final writes still require actual n=Depth; source/DFT and
mutable psi updates use actual versions. The two bounds occupy 8 additional
shared bytes rather than persistent registers. Six integer/provenance tests,
including skipped-endpoint and missing-late-ghost witnesses, pass. The actual
native build passed 76 small complete-state gates and 13 nonzero-clock cases,
including seeded 35-step and 259-step depth-16 macros. Sanitizer and timing
gates remain separate requirements.
The immutable `moving-time-window-v12` build has 48 variants with zero stack or
spills. The Fast body remains 64 registers and 2264 static instruction sites;
the selected generic policy bodies use 121–128 registers. Startup reconstruction
adds 1000 static instruction sites to selected generic bodies, so a spill-free
build alone establishes no speedup. Its manifest records that only the copied
consumer differs from native7 v11.

Complete-state first/repeat screens remain slower than the strong D2 control:
Guide512 takes 5.75125 s versus 5.61082 s, and Mie272 takes 0.976721 s versus
0.657786 s.
These modestly improve the same moving algorithm's v11 screens, but do not
justify a default backend change. The visited-volume reduction is not an engine
speedup prediction.

At the larger Mie512 prefix, v12 takes 3.350644 s versus same-module D2 3.357060 s,
with first/repeat all 30 leaves exact and immutable host inputs. The 0.19%
difference is a screen tie; it establishes a near crossover, not a promotion.

The subsequent Fast-only UniformE build v13 retains the time window. Its copied
rectangle case, with source waveforms extended by repeating the last sample
from 7 to 35, passed fresh ordinary-Program reference and moving reset/repeat
comparisons under memcheck and synccheck. Both tools report zero errors, and
host inputs remain immutable. Instrumentation includes all launched Asymmetric
kernels, with no API-error filtering. The rectangle has zero certified UniformE
bulk tiles, so this validates the general time-window/CPML/source/DFT path,
not the new UniformE true branch. Racecheck was terminated at the remaining
44.9-second hard deadline without a completed report; it is incomplete, not a
pass. Exact module SHA, helper transform, commands, tool logs and scope are in
`window-fast-uniform-v13-sanitizers/summary.json` under the engine research cache.

A separate completed check closes the UniformE true-branch memory/synchronization
coverage: the unchanged small-Mie seeded fixture, 7 steps at depth 4, reports 72
certified UniformE bulk tiles. Memcheck and synccheck both report zero errors,
with two fresh ordinary-Program/reset-repeat all 30 comparisons passing, immutable
host inputs and JAX absent. The actual filter is
`kns=AsymmetricILi4ELi1ELb1`, selecting both depth-4 byte-layout Fast H-class
wrappers; there is no launch-count limit or API-error filtering. Both use exact
v13 SHA `c2240384e6e3621e30c0d531333e39b4bd60704143cd36736d3f2de47d67b6c4`.
Commands, scope, positive branch count and logs are archived in
`fast-uniform-v13-true-sanitizers/summary.json`.

Subsequent bounded racechecks complete for the same positive Fast case and for
the unchanged rectangle at 7 steps/depth 4. Each reports zero hazards, errors and
warnings, with fresh ordinary-Program/reset-repeat all 30 comparisons, immutable
host inputs and JAX absent. The Fast filter remains
`kns=AsymmetricILi4ELi1ELb1` and has 72 certified tiles. The rectangle checks all
Asymmetric launches and has 9 certified tiles. Its canonical window is full-range
on every tile: maximum supports (65,97,129), geometry (7,15,31), zero late-start
or early-end tiles at depth 4. Thus depth-16 general late reconstruction and early
termination were still untested by that check, despite completed
memcheck/synccheck and numerical/dependency proofs. Exact depth-4 race evidence is in
`fast-and-general-v13-racechecks/summary.json`. No suppressions, API filters or
launch-count limits are used.

The final bounded rectangle check completes depth-16 general race coverage at
17 steps: zero hazards, errors and warnings, with fresh ordinary-Program and
moving reset/repeat all 30 comparisons. It uses the existing copied waveform
extension helper, repeating the last sample 7→17 for both paths, and filter
`kns=AsymmetricILi16ELi1ELb0`. One full moving macro executes before the ordinary
tail. Independent geometry identifies 125 late-start tiles and 125 early-end
tiles (15 overlap); all have the boundary flag and therefore execute matched
general wrappers. The original 35-step timeout remains preserved and incomplete.
Exact final commands, result, coverage counts and logs are in
`general-depth16-v13-racecheck/{summary,geometry-coverage}.json`.

The host-only UniformE 75% coverage policy is frozen as research source and has
18 actual-C++/NumPy tests. It uses nonzero original coarse masks as the
denominator: Guide 448/480 and Mie 98/120 both exceed 75%, so it would disable
neither actual workload. No performance benefit is claimed. The previously
measured full 30 fs Mie result for unguarded v2 is positive, reinforcing that short
prefix timing alone cannot justify discarding the class.
