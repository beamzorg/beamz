# Four-hour tracing/CUDA research run

The later [two-hour continuation](cuda-plane-retirement-study-2026-11-12.md)
confirms another 1.12665× throughput improvement over the temporal build below:
155.604→138.112 s on the same full 512³ Mie fixture. Its new support/NoIO
specialization is approximately 1.99× the earlier pinned-main default CUDA
measurement. The study below remains the historical four-hour record.

Started 2026-11-11 20:32:20 UTC; authorized deadline 2026-11-12 00:32:20 UTC.
The bounded research run is complete. We achieved a measurable large-domain
temporal-blocking win, **not the requested several-fold acceleration**. Results
below distinguish complete-engine measurements from isolated field-only
mechanisms. No installed binary or
production default has been replaced. All GPU work is local on the RTX3090,
with its existing 250W cap; no RunPod work is authorized or performed.

## Direct stock-main comparison, 2026-11-12

A subsequent user-requested comparison freshly fetched `origin/main` at
`d5282752db8b258ebbaa5b1b7927026223116b74`, built its untouched CUDA component,
and reran the temporal candidate on the same full 512³ Mie fixture. On main,
`backend="cuda"` defaults to the **streamed** schedule; `cuda_mstep` is an
explicit selection. The earlier 12.1% research result below is not a
stock-main speedup.

| Complete 512³ Mie, 12,347 updates | First execution | Warmed execution |
|---|---:|---:|
| Stock main default CUDA / streamed |274.193 s|275.308 s|
| Stock main explicit cuda_mstep, default 16×32/whole |194.797 s|195.666 s|
| New native moving temporal depth 16 |165.510 s|155.693 s|

The direct measured ratio is **1.7683× throughput**, or **43.45% less warmed
execution time** against default CUDA on main. Both first and repeated solves
match all 30 saved reference leaves bytewise, and all host inputs remain
unchanged. This is one fresh process and one timed repeat per variant, not a
balanced multi-process confidence estimate or a claim across other workloads.
FP32, CPML12, sources, materials, six-face DFT, clocks, preserving ownership and
the 48×256+59 step schedule are identical. These are prepared solver execution
times; scene preparation, compilation, upload and readback are excluded.

Against main's explicit two-step `cuda_mstep` backend with its default 16×32 tile
and whole columns, the temporal build is **1.2567× throughput**, or 20.43% less
execution time. Its first and repeated solves also match all 30 leaves bytewise.
Thus 1.77× is the gain over main's public default CUDA schedule; 1.26× is the gain
over its explicitly selected Mstep schedule. Neither ratio is the earlier
1.12× comparison against an already optimized research control.

The stock build changes only the architecture target to SM86 rather than
building for four architectures. Release mode and fast-math-off match main's
package defaults. All 195 tracked CUDA/BeamZ source files were checked against
their Git blobs and remain unmodified. A CPU-only check invokes main's actual
lowering helpers and proves the external runner's input/output shapes, aliases
and default schedule attributes; main also regenerates all five Mstep plan
arrays byte-for-byte. The installed component is not replaced.

Raw reports, exact source snapshot, build log, frozen runners, input hashes and
the strict comparator result are under
`.cache/main-vs-temporal-2026-11-12/`. In particular see
`full-main-streamed-vs-temporal-comparison.json`, `stock-source-proof.json`,
`stock-lowering-proof.json` and `provenance.json`.
`summary.json` contains all three fresh runs with module/worker/report hashes;
both `full-main-*-vs-temporal-comparison.json` records pass the strict comparator.
An initial external runner unrolled all 49 calls and requested 147 GiB of
workspace. That attempt is rejected and archived as
`full-main-streamed-unrolled-OOM.*`; the final runner uses a rolled XLA chunk
loop, consistent with main's public scan. This was an adapter allocation
artifact, not a measured failure of stock main's default runtime.

## Final measured results

All rows include the complete CUDA runtime workload, FP32, CPML12, sources,
materials, monitors, chronological DFT, tails, clocks and preserving ownership.
Every first and repeated solve matches all 30 reference state leaves bytewise;
input arrays remain unchanged. Times are warmed execution, excluding upload,
graph construction/certification and bounded readback. They do not include
Python scene preparation or compilation.

| Workload | Strong starting D2 | Improved UniformE D2 | Traced moving v13 | Evidence |
|---|---:|---:|---:|---|
| Guide512³,512 updates |5.610s|5.108s|5.751s v12 screen; v13 not retimed|Three fresh alternating D2 pairs; median paired gain1.0981×|
| Mie272³,680nm cube,30fs/6560 updates |16.900s|16.259s|Not measured full-pulse|One fresh process pair; D2 gain1.0395×|
| **Mie512³,680nm cube,30fs/12347 updates** |**174.513s**|**170.698s**|**155.619s**|One fresh process per variant; first solve plus timed repeat each|

The large full Mie case contains 134,217,728 Yee cells and uses a 1.328125 nm
mesh, 100 nm-radius index 2 sphere, 500 nm source and six-face DFT. The moving
compiler is **1.1214× faster than the strong starting D2**, a 10.83% execution
time reduction. It is **1.0969× faster than the new improved D2 candidate**.
Counting one complete six-field Yee timestep as one cell update, throughput
is 9.496, 9.708 and 10.649 billion cell updates per second respectively.
The complete workload ran through the entire physical pulse; these numbers
are not extrapolated from a short prefix. The one-process comparisons are
observations, not a confidence interval or proof of universal improvement.
The guide D2 gain is more strongly replicated: paired speedups1.098349,
1.097992 and1.098113. Smaller scenes should retain D2 based on these results.

Full512 setup/first-call/warmed seconds respectively:

| Runtime | Setup | First execution | Warmed execution | Resident payload |
|---|---:|---:|---:|---:|
| Starting D2 |4.656|174.407|174.513|12.434GB|
| UniformE D2 v2 |3.027|169.730|170.698|12.434GB|
| Moving v13 |3.460|165.832|155.619|13.768GB|

The moving first execution includes material certification and graph creation;
it does not deliver the same gain as its warmed run. Certification also uses
542MB of temporary host prefix storage plus private material snapshots. Peak
process host RSS was13.08GB. Resident payload is not total allocator/driver
memory. Graph specialization and constructor overhead therefore matter for
single-shot jobs. Setup timings here are single observations, not optimized
comparative claims.

Raw full 512³ packets, readbacks, frozen worker and three reports are in
`.cache/autoresearch-4h-2026-11-11/final-mie512-full/`. Both
`uniform-comparison.json` and `temporal-vs-uniform-comparison.json` passed the
strict complete-physics comparator. Full272 evidence is in
`final-full30fs-mie272/`; replicated guide evidence is in `uniform-confirm-v2/`.
The interactive result chart is in
[cuda-autoresearch-four-hour-2026-11-11.html](cuda-autoresearch-four-hour-2026-11-11.html).

## What the tracing compiler now does

The final native v13 emits the existing canonical cell operator, rather than
maintaining a second handwritten physics formula. The trace exposes component
reads, field versions, material decoding, CPML, injection and observation.
Scheduling and storage lowering then produce a diagonal moving cache with
owned7×15×31 and cached8×16×32 cells. All six fields live in fixed private
slots; a three-component shared exchange frame handles cross-thread reads.
Coalesced low-X imports and four barriers replace the original shared-heavy
consumer. Fast tiles use1024 threads and 64 registers; general tiles use 512
threads and 121–128 registers. All 48 kernels have zero stack/spills and use
51,748 shared bytes. Temporal depth 16 carries directional face trails across
ranked tiles, avoiding repeated full-volume field loads/stores.

The compiler proves immutable support and material facts before selecting a
specialization. Exact coefficient-bit certificates remove repeated material
decoding on eligible tiles; arbitrary values retain the original decoder.
Active time windows avoid empty startup/retirement work. Independent same-rank
policies run on graph branches, with explicit joins before later ranks, DFT,
epoch changes or clock advancement. Raw detector samples use direct sparse
lookup and retain chronological DFT arithmetic. Source multiply/add rounding,
CPML memories and physical state ownership remain unchanged.

The benchmark runtime imports **no JAX**. JAX remains in scene preparation and
independent reference export. Thus removing JAX from execution alone does not
explain the remaining limit: the measured bottlenecks include register/shared
capacity, instructions, synchronization, boundary/source/monitor work and tile
startup/retirement. The tracing improvement is the ability to prove and lower
these facts while keeping one physics definition.
The current frontend traces BeamZ's canonical Yee formulas with explicitly
supported material/source environments. General tracing of arbitrary Python or
PyTorch cell programs remains future work.

The relevant implementation entry points are the
[canonical native cell generator](../../scripts/generate_cuda_native_cell.py),
[geometry IR](../../scripts/cuda_native_geometry_ir.py),
[register-frame lowering](../../scripts/patch_cuda_native_register_frame.py),
[active time-window lowering](../../scripts/patch_cuda_native_time_window.py),
[Fast UniformE lowering](../../scripts/patch_cuda_native_fast_uniform_e.py),
and [parallel policy graph lowering](../../scripts/patch_cuda_native_parallel_policies.py).
The improved D2 material path is in
[patch_cuda_register_material.py](../../scripts/patch_cuda_register_material.py).
These patchers assemble isolated research copies; they do not install a new
component. Frozen assembled sources, build manifests and exact binary hashes
remain with each candidate under `.cache/`.

## Validation and remaining limits

The complete moving path passes 97 seeded/material/clock numerical cases,
including replay, reset, odd tails, nonzero clocks and true UniformE branches,
plus both full 512³ first/repeat reference comparisons. Root's final focused CPU
sweep passes 56 tests; other native/export suites pass separately. The D2
candidate passes 42 ownership/continuation cases plus six changed-H/PEC cases;
its actual constructor certificate is checked against an independent oracle.

General-path memcheck and synccheck pass with zero errors. An additional
depth 4 small-Mie check passes both tools on the actual Fast UniformE branch:
72 certified bulk tiles, unchanged source/CPML/DFT, ordinary-reference and
reset/repeat all 30 bytewise checks. It instruments both Fast H-class wrappers
without a launch-count limit or API-error filtering. Additional bounded
depth 4 racechecks pass with zero hazards/errors/warnings on both that Fast
true branch and a rectangular all-Asymmetric case, retaining exact first and
reset/repeat state. Evidence is in `fast-uniform-v13-true-sanitizers/` and
`fast-and-general-v13-racechecks/` under the native research cache. The earlier
general-path 35-step depth 16 racecheck timed out; that attempt is preserved.
A subsequent 17-step depth 16 general racecheck completes with zero hazards,
errors or warnings and exact ordinary-reference/reset-repeat state. It runs
one complete 16-step macro and one ordinary tail, covering 125 late-start and
125 early-end tiles, including 15 with both, as checked independently from the
actual geometry. Its source extension from 7 to 17 samples is identical for
ordinary reference and candidate. Evidence is in
`general-depth16-v13-racecheck/{summary,geometry-coverage}.json`. This closes
the specific active-window race-check gap within that bounded case.
Sanitizer coverage remains bounded to the stated cases, not all scenes or
instantiations. Broader feature coverage and balanced full-workload performance
validation remain promotion gates. No production
default or installed binary is replaced. Native admission remains narrower
than BeamZ's full feature set: FP32 Cartesian CPML12, scalar H and supported
lossless packed E/source forms; unsupported scenes retain their current path.

The next concrete architecture experiment is
[E-plane retirement](cuda-plane-retirement-next-experiment.md): derive the
last-use frontier from the trace, retain H privately and stream old/new E
planes through one shared family. It targets the paper's 15×15×31 owned tile
without the spills seen in our full-register FP32 builds. This is an
unimplemented schedule requiring a dependency proof and full consumer resource
check. It trades more barriers for less live storage; no speedup is promised.
Logical steady interior field traffic would fall from 11.6203 to 7.9553 bytes
per Yee update before material/CPML/observer/startup costs. That arithmetic
does not predict several-fold complete-engine acceleration.

The sections below preserve the chronological experiments, including failed
variants. Their historical geometry and interim results are not the final v13.

## Measurement contract

Use FP32 fields and exactly 12 CPML cells at every resolution. Complete numerical
gates compare all 30 state leaves bytewise: six fields, twelve CPML memories,
DFT state, and clocks. First and repeated calls must both pass, with matched
ownership, chunk schedule, source, material, observer, and timing semantics.
Changed-physics cases receive new ordinary CUDA Program references. Exported
independent JAX references are also used, but JAX is absent from the benchmark
runtime interpreter. Compilation, fixture upload, and bounded readback are
reported separately from warmed execution.

The strong guide control is512³,512 updates, an 80 nm mesh, binary waveguide,
mode source, two mode monitors with three frequencies, CPML12, preserving
ownership, two256-step chunks, and D2 Z-segment64. The short Mie screen is272³
with 256 updates. The full Mie case is a 680 nm cube,2.5nm mesh,6560 updates/30fs,
CPML12, and the same nanoparticle/source/monitor physics as the released-FDTDX
comparison. Short-screen timings do not establish full-pulse gains.

## Retained complete-engine improvement

`register-adaptive-compact-v1` uses one lane per warp-rounded XY coordinate,
a uniform device coefficient-class proof with the original arbitrary-scalar
fallback, and a compact CPML edge metadata view with the original fallback.
The default32×16 core uses40 registers, no stack/spills,39,336 shared bytes,
and two resident CTAs. Device class selection requires lossless scalar H decay
and three bit-identical finite H source coefficients. It does not constrain
other packets at the API boundary.

Three fresh paired guide processes show a median paired speedup of1.062521×
over the strong D2 control (observed range1.060429–1.065619). Representative
medians are5.6053s versus5.2755s. Two fresh short-Mie pairs show1.035906×.
A separate full6560-update Mie screen is16.8613s versus16.7002s: essentially
a tie, not a robust improvement. Artifacts and exact frozen worker sources:
`.cache/autoresearch-4h-2026-11-11/adaptive-confirm-v1/` and
`.cache/autoresearch-4h-2026-11-11/full30fs-mie272/`.

Seeded rectangular/Mie ownership, reset, replay, split continuation and odd-tail
gates are exact. Six changed-physics coefficient cases, including anisotropic
source scaling, mixed/uniform loss, signed zeros, and PEC intersections, also
match fresh ordinary CUDA references in all 30 leaves.

## Complete tracing-driven moving cache

The compiler emits the actual canonical point formula through
`AdvanceNativeCell<Phase,Component>`. Its environment supplies cache reads,
packed materials, native boundary differences, local CPML recurrence, and
post-update sources. In the initial v3, the owned box is5×15×31, cached6×16×32.
Hx/Hz/Ex/Ez occupy shared memory; Hy/Ey retain six fixed private slots each.
A four-rank frontier carries directional face trails. CPML psi initially stays
in place in global memory. Sources retain exact split multiply/add rounding;
raw monitor samples are published at the correct version, then reduced by the
existing chronological DFT kernels. Unsupported cases fail explicitly.

The first complete v3 port is correct but slower than strong CUDA: guide512
8.371s versus5.611s, short Mie1.903s versus0.660s. General tiles dominate:
61.8% of guide GPU time and87.2% of Mie GPU time. This is why a field-only
win does not establish a full-engine speedup. Full details and admission limits
are in `cuda-asymmetric-native-2026-11-11.md`.

Isolated lowerings, each numerically gated:

| Candidate | Guide512 | Short Mie272 | Interpretation |
|---|---:|---:|---|
| v3 complete moving |8.371s|1.903s|Correct, slower than default CUDA|
| Scalar Z queue |8.436s|1.884s|Rejected; smaller static text alone did not help|
| Six-field import packets |7.224s|1.705s|Retain as better research parent|
| Sparse direct observation, v3 parent |8.288s|1.313s|Real Mie hotspot; guide inconclusive|
| Canonical geometry, import parent |7.187s|1.696s|Sub1% differences inconclusive|

These are exploratory screens, not fresh paired confidence estimates. They
have different parents as stated and should not be treated as an additive
speedup ladder. Six-policy dispatch, uniform material facts, and stationary
CPML storage are being evaluated separately before composition.

## Tracing facts beyond arithmetic

A useful trace must expose both cell operations and immutable execution facts.
The region compiler proves source/observer intersection, component support,
CPML axis membership, version clocks, and material classes. On guide512,
24,974 of25,044 general tiles have no source or observer. On Mie, distinct
observer-only, source-only, and combined interior regions justify separate
kernel policies. Their omission of operations is conditional on exact
certificates; arbitrary scenes retain the general consumer.

`scripts/cuda_material_tile_facts.py` proves per-component coefficient-bit
uniformity over the full transformed trajectory box, clipped to actual
staggered supports. It treats aliased table codes as equal only when bits
match, preserves signed-zero distinctions, and matches the decoder's invalid
byte-code positive-zero semantics. A summed-volume table makes box queries
constant-time after a bounded per-component pass. All three E coefficients
are uniform on 99.657% of real guide updates and 95.002% of real Mie updates.
These are coverage facts, not measured speedups. A second candidate applies
an analogous full-column certificate to the strong adaptive D2 path using
private uploaded metadata; caller arrays and the packet ABI are unchanged.

The stationary CPML proof exploits the fact that a physical point retains its
cache slot while one tile owns it. A new owner lazily loads the last global
psi state; the previous owner publishes it on the high-face union or terminal
version. Only single-axis, IO-free tiles are initially eligible, retaining
four psi scalars per point and using the original global path elsewhere.
Logical scalar-access counts predict substantial psi traffic reductions;
GPU resources, races, and complete performance remain separate gates.

## First-principles capacity experiments

The field-only coalesced low-X cache is exact and sanitizer-clean. Fresh paired
measurements show about1.064× over matched D2 at512³ and1.247× at640³. It
omits CPML, sources, materials, and DFT, so these are mechanism results only.

Widening the cache to12×16×32 lowers steady logical field traffic from14.369
to9.121 bytes per Yee update, but needs99,692 shared bytes and128 registers,
leaving one CTA. It remains exact at tiny,512³ and640³ sizes, yet loses:
189.0ms versus154.4ms at512³ and330.5ms versus313.4ms at640³. The larger
cache is rejected. Traffic reduction without a viable instruction/occupancy
budget is insufficient. A1024-thread, one-family shared cache is a separate
bounded resource experiment, not an accepted solver.

## Paper relationship

The [Lu et al. systolic paper](https://arxiv.org/html/2502.20610v1) describes
moving cached subdomains along the space-time diagonal and exchanging face
trails. Its reported setup uses16-bit fields on H100, material/conductivity
fields, adiabatic absorption, and no sources or outputs. Our experiments
preserve FP32, CPML12, real injection and DFT. The paper's5.5× comparison is
against its naive implementation; it does not predict5.5× over BeamZ's
already temporally blocked D2 backend. The directional dependency argument
is implemented and proved here, while its complete performance remains an
engineering question.

## Reproduction and evidence

Research binaries, source hashes, exact build commands, frozen workers,
numerical gates and raw timings are under
`.cache/autoresearch-4h-2026-11-11/` and
`.cache/cuda-engine-autoresearch-2026-11-11-2032/`.
`cuda_research_compare.py` rejects mismatched packet physics, ownership,
schedules, host mutation, or non-bitwise first/repeat state before comparing
times. `analyze_cuda_sass.py` records static instruction sites with binary
identity; static code size is not a dynamic execution/cache-miss measurement.

For example, reproduce the complete large moving run from the repository root
using the frozen worker and existing prepared fixture/binary:

```sh
.cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python \
  .cache/autoresearch-4h-2026-11-11/final-mie512-full/worker.py \
  --fixture .cache/autoresearch-4h-2026-11-11/final-mie512-full/fixture-seg64.json \
  --backend native --method moving --moving-depth 16 --ownership preserve \
  --module .cache/cuda-engine-autoresearch-2026-11-11-2032/moving-fast-uniform-v13/build/_cuda_native.cpython-311-x86_64-linux-gnu.so \
  --warmups 0 --samples 1 --output /tmp/beamz-moving512-reproduction.json
```

The worker performs an initial complete validated solve before its timed
repeat. Do not run a second GPU workload concurrently. The compact machine
readable result/provenance index is
`.cache/autoresearch-4h-2026-11-11/final-results.json`. Timings above refer to
specific binary/report hashes, not whichever source is later rebuilt.

The several-fold complete-engine target was not achieved. Final complete
measurements and their limits are recorded at the beginning of this report.

## Update at23:18 UTC: material facts and parallel graph scheduling

The strong D2 uniform-material candidate (`register-uniform-material-v2`) now
has three fresh alternating pairs against the same strong D2 segment64 control.
All packet/ownership/schedule/physics hashes and all 30 first/repeat leaves agree.
Paired gains are1.098349,1.097992 and1.098113; median1.098113×. Median execution
is approximately5.610s versus5.109s.448 of480 active coarse XY columns admit the
conservative full-column constant-material proof. Setup increases from roughly
1.76s to2.10s because certification runs once during upload. This is a warmed
execution gain, not an equivalent first-result improvement. A short-Mie screen
shows no improvement (.62791s versus the adaptive parent's .62183s), so no
universal material-class performance claim or default promotion is made.
`uniform-confirm-v2/` preserves the six fresh processes, comparator records,
frozen worker and source/binary hashes. The actual constructor's C++ certificate
also passes five independent NumPy-oracle cases including signed zeros, aliased
codes, invalid byte codes and halo-only material changes.

Serial six-policy dispatch is numerically exact but regresses Mie because each
rank has too few tiles per class to fill82 SMs. The new graph compiler forks
same-rank policies onto persistent nonblocking streams and joins every branch
before the next rank, DFT, epoch or clock update. Distinct same-rank tiles never
own the same physical psi point at any version: componentwise owner coordinates
can only increase, and changing tiles strictly increases rank. Trails, final
fields and raw detector keys likewise have unique writers. The complete gates
exercise actual macros, replay, host-metadata mutation and nonzero clocks.

Parallel-policy v8 screens at6.399s guide and1.492s short Mie, recovering the
serial policy's7.113s/3.236s. CUPTI confirms max6 concurrent kernels; Mie kernel
interval union is1.4825s versus4.419s summed durations. Overlapped sums must
never be presented as wall-time shares. Composing sparse direct observation
with that graph (v9) yields6.401s guide and1.011s short Mie. All76 small gates,
13 clock cases and large first/repeat all 30 leaves are exact. Strong D2 still
wins (5.599s/.650s); the several-fold complete-engine target remains unmet.

The guarded CPML allocator now derives interference from actual traced
recurrences. Immutable one-axis activation guards allow twelve logical psi
states to share four private scalar slots; without that fact, twelve are
required. H/E histories remain separate. Coloring cuts static wrapper text
from21,848 to13,496 instruction sites compared with three separate axis bodies.
The colored wrapper uses127 registers without spills and unchanged49,820B
shared memory; it remains one CTA. Actual numerical integration passed, but
regressed to6.513s guide and1.0285s short Mie versus its6.401s/1.011s parent;
it was rejected for performance. Coloring is a storage allocation transformation,
not an arithmetic change or a claim about dynamic instruction/cache costs.

A new1024-thread cache retains24 private field scalars per thread and uses a
single shared family. With coalesced low-X imports it passes tiny,512³ and640³
field-only state/guard/repeat checks and screens at126.2ms/222.8ms versus matched
D2's157.3ms/316.2ms (about1.25×/1.42×). It is a mechanism improvement, not a
complete-engine result. A separate high-X export variant initially left valid
outputs unwritten; finite output poisoning identified that distinction from
propagated NaNs. An explicit geometric retirement predicate fixed the numerical
gates, but high export is slower (131.4ms/233.1ms), so it is rejected. All failed
artifacts remain labeled and are excluded from accepted performance results.

The full paper-sized FP32 cache uses paired Z lanes, two shared components and
a narrow shared Y-edge frame. The independent routing proof checks32,768
neighbor routes and rejects wrong parity/half-frame mappings. Its first resource
build spills; a full-cache interior certificate removes clipping but still
spills. PTX contains no local arrays before ptxas: this is genuine allocation
pressure, not array escape. No such spilling paper-sized binary has run on GPU.
