# Large-domain CUDA design from the dependency graph

The strongest architectural correction is a phase-correct, **persistent physical
cache**, not merely a deeper overlapping tile. Stage incoming boundary values
once, retain them for the following timestep, and update each retained physical
cell in place. A circular `(B+1)` cache supplies that mechanism while avoiding
the old-own-field gather/register retirement in our first moving-box probe.

Everything below separates paper statements, original derivation, source-level
payload predictions and measured evidence. Predicted payload is **not measured
DRAM traffic** and is not an engine speedup prediction. CPML/source/DFT execution
must still beat the best complete production D2 control.

## Paper audit: what is actually specified

The [paper](https://arxiv.org/pdf/2502.20610) was read through its 3D implementation
and Figures 3–7, including the drawings. Figures 3/4 show rectangular blocking's
cyclic dependency and halo remedy; Figure 5 changes ownership to a diamond.
Figure 6 flattens its history into a sequential diagonal bar. Figure 7 instead
moves a same-time horizontal bar, permitting phase-parallel interior updates
and lower-dimensional boundary communication. Section 4 specifies 31×15×15
updates within 32×16×16 cached cells and approximately 1280 face cells per
iteration. It reports 5.45 fewer global operations and roughly 4 FLOP/B, says
structural-field transport doubles operations and 16-bit representations halve
bytes. Its test excludes sources/output fields and uses adiabatic absorption,
not PML. Abstract/results state approximately 0.15 TCUPS; the introduction says
0.18. The paper does not provide enough code, timestep/domain, SKU or timing
detail to reproduce the discrepancy or a matched BeamZ comparison.

That is the paper summary. The following equations and code audit are our own
derivation, with implementation assumptions exposed.

## Figures 6/7 versus our actual code

Our fixed-u ownership, `u=p+n`, already makes a horizontal physical-time block
move upwind. It is therefore a Figure 7-style dependency schedule, rather than
the mixed-time sequential bar in Figure 6. The important implementation gap is
where values reside between phases/timesteps:

- The original `Moving` stores only owned cells in shared memory. `Fetch`
  reaches into global trails for each nonlocal operand. Its fixed-u shared
  slots represent different physical cells on the next iteration, requiring
  all old own H/E values to be gathered before any H overwrite.
- The broad `FaceCachedMoving` adds a padded cache. It imports all old low-halo
  values once, then replaces H's pure low faces with **current** H before E.
  This removes derivative-time global addressing, but retains own-field GPR
  retirement and reloads old boundary values on every iteration.
- A physical circular cache retains the intersection in the same physical
  slots. Imported current boundary values become resident old values next time.
  It removes that repeated old-halo import and the shifted own-field gathers.

The paper does not specify register mapping, ring layout or exact face packing.
Its cache dimensions support this last interpretation; they do not prove a
particular CUDA implementation or register count.

## Exact interior geometry and operand counts

Let owned dimensions be `B=(z,y,x)`, volume `V=z*y*x`, summed face area
`A=zy+zx+yx`, and summed edge length `e=z+y+x`. Define

```
S = V - (z-1)(y-1)(x-1) = A-e+1       outgoing owned high-face union
P = (z+1)(y+1)(x+1)                   padded physical cache volume
L = P-V = A+e+1                       incoming low-face union
```

Edges/corners are included once. An interior producer must publish all six
fields on `S`: its old-own-field recurrence is consumed from `u-ones` by later
boxes. Current-H curls require only tangential H components on pure low faces.
Those pure planes have their other coordinates owned, so they are disjoint;
their exact total is `2A` scalars, not `2A-e`.

For the original callback schedule, per tile/full timestep:

```
H-phase requested nonlocal reads = 9S + 4A - 2e
E-phase requested nonlocal reads = 3S + 2A
total requested reads            = 12S + 6A - 2e
unique incoming field/version/p  = 6S + (4e-6) + 2A
outgoing stores                  = 6S
```

The `4e-6` term is old-E upper edge strips needed by positive H derivatives,
beyond the shifted old-center box. H/E phases share `3S` old-E inputs. Compiler
CSE/cache reuse can eliminate repeated requests; unique inputs per consumer
are also not DRAM reads, because consumers may share cache lines.

| Owned B, z/y/x | V | S | L | Requested reads | Unique incoming | H phase requested/unique | E phase requested/unique |
|---|---:|---:|---:|---:|---:|---:|---:|
|4/8/32|1024|373|461|6884|3240|4933/2408|1951/1951|
|8/8/16|1024|289|353|5324|2496|3817/1856|1507/1507|
|15/15/31|6975|1095|1217|19948|9118|14353/6808|5595/5595|

These formulas are cross-checked by enumerating the actual Yee trace. They
describe the algorithm's requested scalar values, not post-compiler loads.

## What face staging changes

Ignoring cold initialization, final scatter and physical walls, field-only FP32
payload per useful full Yee update is:

| Owned B | Original requests+stores | Broad old 6L+current 3A+stores | Tangential current 2A | Circular current 6L+stores |
|---|---:|---:|---:|---:|
|4/8/32|35.633 B|24.422 B|22.797 B|19.547 B|
|8/8/16|27.570 B|18.797 B|17.547 B|15.047 B|
|15/15/31|15.207 B|9.942 B|9.280 B|7.955 B|

The broad cache's 4/8/32 reads are `6*461+3*416=4014` floats versus 6884 callback
requests. It overfetches the original algorithm's 3240 unique incoming values.
Therefore a speed gain cannot automatically be attributed to fewer DRAM bytes:
coalescing, less address/control work and compiler behavior matter. Tangential
H staging saves one component per face without changing the old cache's shape.
Circular retention adds reuse across timestep boundaries, reducing imports to
`6L` current values and retaining old own values locally.

## Physical-slot invariant and the three-barrier schedule

For fixed owned-u lower corner `a`, at step n:

```
Q(n)   = [a-n,   a-n+B-1]       owned physical cells
P(n-1) = [a-n,   a-n+B]         previous physical cache
P(n)   = [a-n-1, a-n+B-1]       new physical cache
P(n-1) intersection P(n) = Q(n)
```

All intervals are componentwise. Map a physical coordinate to a stable slot
modulo `(B+1)`, relative to the initial cache origin. A rolling nonnegative head
implements this without C++ signed remainder on negative wall coordinates.
Retained physical cells do not move in storage.

1. Update H on `Q(n)` in place. Its old own H and E at p/positive neighbors all
   lie in `P(n-1)`. H writes do not change any E input. **H barrier**.
2. Import all six current-version values on `P(n)\Q(n)`. Their circular slots
   replace the old cache's outgoing high slots, which H has finished reading.
   Material values for new physical cells can be loaded here. **Import barrier**.
3. Update E on `Q(n)` in place. Old own E remains resident; H at p/negative
   neighbors is current throughout `P(n)`. **E barrier**, then publish `S`.

No extra publication barrier is required only under a precise read footprint:
publication **loads**, not merely stores, must be gated to the outgoing high
union. `Q(n+1)` excludes that union. Both sets lie in `P(n)`, whose slot mapping
is injective, so next H writes cannot alias previous publication reads. The
next H barrier retires those reads before imports recycle the old high slots.
Final whole-output reads are allowed on the last step, where no next H exists.
An unconditional all-owned load before testing `Surface` violates this proof.

`scripts/prove_cuda_circular_cache.py` checks physical coordinates, versions,
cache slots, lower-owner imports and publication disjointness in 18 irregular,
wall-crossing cases. Early import, stale H(n-1), and all-owned publication reads
all fail. This is a dependency/liveness proof, not a CUDA racecheck substitute.

## Reconstructing 5.45, bytes and the performance discrepancy

The paper's geometric ratio is directly reproducible:

```
6975 / (32*16 + 32*16 + 16*16) = 6975/1280 = 5.44921875
```

Counting six FP32 fields, a read and a write of those face cells costs
`2*1280*6*4/6975 = 8.8086 B/update`. At 36 illustrative FLOP/update, that is
`4.087 FLOP/B`. This arithmetic explains the approximate 4 under a stated
load+store convention. It is not measured DRAM traffic.

The introductory naive count is 36 scalar accesses/update, or 144 B, whereas
reading/writing a perfectly reused six-field cell is 12 accesses, or 48 B. Thus
the 16.35 ratio `144/8.8086` combines implicit 3× cell-local reuse with 5.45× face
geometry. Comparing 0.25 FLOP/B directly against 4 while calling the whole change
5.45× mixes those reference levels. It need not be a mathematical contradiction,
but it is not a single consistently measured memory-operation reduction.

If transported payload really doubles while *all transported values* halve in
size, the two factors cancel in bytes. If only E/H are FP16 but six material
values remain FP32, the above face payload becomes 13.213 B, not 8.809. The paper
does not pin down that storage layout. Our model uses six material channels as
an explicit scenario, not a claim about unpublished code. Immutable material
can instead be loaded only on arrival and need not be stored in trails. Three
packed lossless E codes are a different scenario again.

The approximate 0.15 result and 5.5× statement should not be replaced by the
introductory 0.18. Neither has a disclosed matched FP32/CPML12/source/six-DFT
comparison. The paper also writes 60×10^9 FLOP/s and 4×10^9 B/s in its hardware
example; NVIDIA's [H100 datasheet](https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306)
lists 60 TFLOP/s and 3.9 TB/s for NVL. The ratio survives that common 1000× unit
factor; absolute throughput units do not.

## FP32 resource feasibility on SM86

[NVIDIA's Ampere guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/) gives
64K 32-bit registers/SM, 100 KiB shared/SM, 99 KiB shared/block and 48 resident warps.
Shared allocations above 48 KiB require dynamic opt-in. Paper geometry should
not be copied blindly into a shared-memory FP32 kernel:

| Owned B | Padded cells | Six FP32 fields | +six FP32 material channels | +three byte codes instead |
|---|---:|---:|---:|---:|
|4/8/32|1485|35,640 B|71,280 B|40,095 B|
|8/8/16|1377|33,048 B|66,096 B|37,179 B|
|15/15/31|8192|196,608 B|393,216 B|221,184 B|
|7/7/15|1024|24,576 B|49,152 B|27,648 B|

The paper-sized six-field FP32 cache exceeds the per-block shared limit. Six
fields plus six FP32 material channels exceed even total registers+shared,
before temporaries. A mixed register/cache decomposition is another design,
not an automatic port. The original H100 precision/footprint is materially
different. Balanced 8/8/16 keeps 1024 owned cells while reducing face payload
approximately 23% versus 4/8/32 and uses less padded storage.

At 256 threads, two resident CTAs require roughly ≤128 registers/thread *and*
enough shared storage. A 66 KiB material-rich block limits shared residency to
one regardless of register count. Full twelve-psi FP32 storage adds 48 bytes
per cached cell: 8/8/16 fields+psi alone are 99,144 B, leaving almost no per-block
space. Real packed CPML supports, component staggering, coefficients and source/
observation metadata require explicit admission and compiled measurements.

A paper-pattern SM86 candidate is owned 7/7/15 with power-of-two cache 8/8/16.
Its 735 updates use 24 KiB for six FP32 fields; adding all 12 logical psi arrays and
three byte codes gives 75 KiB before metadata. This fits the nominal 99 KiB block
budget, unlike a direct FP32 copy of the paper geometry. Its circular field
payload is 16.980 B/update, so it trades some surface efficiency for cheap mask
rotation and a complete-physics storage budget. The earlier prototype's `(q-1)&(B-1)`
incoming indexing is invalid for such non-power-of-two owned dimensions: explicit
`q==0 ? B-1 : q-1` is required. The parametric prototype headers now use that
conditional and guarded partial-thread tails. This section is a resource model,
not a measurement of the new geometry. The independent geometry proof covers
12 tile/depth configurations through T64 and rejects the old owned mask,
unguarded tails, wrong E-neighbor sign and missing rolling-head updates.

### Half-shared/register circular state

A larger FP32 cache can distribute storage across registers and shared memory.
Assign each fixed cache slot `tid+j*threads` to one thread; retain all three H
and E components for that slot in fixed, unrolled register variables. Only one
opposite family occupies shared memory at a time. No thread reads another's
registers.

The schedule is H update; current low-face imports into the evicted register
slots; barrier 1 retiring Eshared reads; all H registers to shared; barrier 2;
E update; barrier 3 retiring Hshared reads; all E registers to shared; barrier 4.
Owned old E is disjoint from imports. Outgoing reads use the owning registers,
so no cross-thread publication retirement is necessary. Full H loops must
precede register imports if old-E register neighbors are used.

| Cache / owned | Threads | State registers/thread | One-family shared | Resident target |
|---|---:|---:|---:|---|
|16³ / 15³|256|96|49,152 B|two CTAs if total registers ≤128|
|16×16×32 / 15×15×31|512|96|98,304 B|one CTA if total registers ≤128|

The two targets both supply 16 resident warps when successful. Paper geometry
has 6975 owned cells versus 6750 across two 15³ blocks. Shared bytes plus the 64-byte
base table require dynamic shared opt-in. State alone leaves only 32 registers
per thread for addressing, coefficients and arithmetic; local spills can erase
the advantage. This is a compiler/resource gate, not an occupancy promise.

Opposite-family centers are same-thread registers. Z neighbors are also
same-thread fixed slots when the cache XY plane equals the thread count.
Completing all H updates before imports preserves old E for H's positive Z
reads; all H registers are current before E's negative Z reads. Thus source-level
opposite shared requests decrease 24→12→8 floats/update for all-shared, local
centers, then local centers+Z, before compiler CSE. X/Y neighbors remain shared.
The 16³/512 alternative has two cache planes per thread group; its Z neighbors
change thread owners and must retain shared reads. The CPU test rejects that
local-Z admission.
`prove_cuda_circular_half_shared.py` checks both target geometries and a small
4³ reference, with early-overwrite, stale-import, cross-thread-register and
wrong-Z-direction failures. Fifteen positive cases cover three sharing modes;
five negative controls fail. It does not prove that CUDA arrays scalarize or fit.

At N512, finite circular field payload for T8/T16 is 16.610/13.733 B/update for
15³ and 14.769/12.015 for 15×15×31. These retain cold/final scatter and invalid
publication costs. They exclude CPML/material/source/DFT execution and cannot
be treated as a measured full-engine improvement.

### Paper-shaped warp X and two-component shared frame

For cache 16×16×32 and 512 threads, every warp is one complete cache X row;
each fixed register index j is one Z plane. Canonical Yee X derivatives use
only opposite-family components 1/2: H prefetches old Ey/Ez at `(lane+1)&31`,
and E prefetches current Hy/Hz at `(lane-1)&31`. Both shuffles must execute for
**all 32 lanes**, at every j, before any owned or physical-validity predicate.
Evicted/ghost lanes still provide operands to active neighbors. Circular wrap
is correct because the register slots retain physical coordinates modulo 32.
This proof admits X=32 only; a width-16 mapping needs a separate subgroup proof.

Centers and Z remain same-thread registers. Only Y derivatives remain shared,
and they use components 0/2. The shared frame can therefore hold two components,
not three. Complete H loops still precede imports; the four existing retirement/
publication barriers remain. H shuffles see old E; E shuffles see current H after
imports. E updates do not change those H register sources. Each thread publishes
its outgoing values from its own registers.

The source-level opposite shared-load count becomes 24→8→4 scalars per useful
interior update for all-shared, register centers+Z, then warp X. Per-step frame
writes fall from 6P to 4P. At P=8192, V=6975, the last two designs request 104,952
versus 60,668 shared scalar reads+writes per step before CSE. Warp X adds 1024
warp shuffle instructions, or 32,768 lane transfers, per step, including ghost/
nonowned participation. It does not change global field/trail payload.

The frame is 65,536 bytes; eight base addresses and the phase counter bring the
prototype to 65,604 bytes. State remains at least 96 FP32 scalars per thread.
One 512-thread CTA is still the shared-residency limit on SM86; total registers
must fit the per-SM budget, and shuffle temporaries/addressing may spill. These
are operation/resource predictions, not a speed or DRAM-traffic measurement.

`prove_cuda_circular_warp_cache.py` checks nine T2/4/8 cases with low, interior
and high owners. Tags cover physical walls, invalid lanes, incoming/evicted
slots, version clocks and register ownership. Five negatives reject divergent
full-mask shuffles, incorrect wrap direction, stale old-E/current-H operands,
and omission of a required Y component. The focused test also verifies exactly
four shared and four shuffled useful operands per interior update. CUDA
convergence and the compiler lowering still need independent GPU checks.


The actual SM86 compiler diagnostic **rejected this lowering before GPU use**.
For T4/8/16/32 it reached the 128-register launch cap, with a 184-byte stack and
ptxas reports of 916 spill-store / 976 spill-load bytes. An isolated per-j
`__syncwarp` variant still reported 896/964 bytes and the same stack. Those are
compiler resource diagnostics, not measured dynamic DRAM traffic. The original
T8 SASS contains 64 unpredicated shuffle instructions, matching the 16 j slots ×
two operands × two phases in the loop body; correct collective placement did
not solve register pressure. The hypothesis reduced shared requests but failed
its resource gate, so no speed claim or GPU timing follows.

Exact commands, source/header/binary hashes and all eight resource rows are in
`.cache/cuda-large-domains-2026-11-11/register-warp-resource-rejection.json`;
flags include SM86, O3, relocatable device code, 512 threads, one minimum CTA,
phase metadata and the warp-neighbor specialization. Independent token proof
and an implementable register schedule are separate requirements.

### Asymmetric permanent shared storage

A different storage trade keeps Hx/Hz/Ex/Ez permanently shared, with only Hy/Ey
in registers. Cache 6×16×32, owned 5×15×31 and 512 threads give six fixed slots per
thread: 12 retained register scalars rather than 96. Opposite y-component X
neighbors use unconditional full-warp shuffles; their Z neighbors stay in the
same thread at `(j±1) modulo 6`. Canonical Y derivatives never consume component
y. All other operands read the permanent shared fields. Modulo 6 is essential;
`&5` is incorrect. X still wraps at 32, and ghost/evicted lanes participate.

The schedule is H updates in place; B1; import all six current low-face fields;
B2; E updates in place; B3; publish outgoing high-face values. There is no shared
family-copy phase. B1 retires old-E reads before imports recycle the old high
slots. B2 publishes H/import values before E. B3 retires E stencils and publishes
E before observations. Gated outgoing shared-load slots are disjoint from the
next H write slots. Next B1 retires those publication readers before imports
reuse their slots. Loading every owned point before testing the surface breaks
this proof; only the final step may load whole final outputs.

The phase counter has a separate interwarp lifetime: initialize 1; H reads n;
B1 retires **all** H readers; thread 0 writes n+1 during imports; B2 publishes it;
E reads counter−1=n; B3 retires all E readers. The next counter write is after
next B1, which requires every warp to have crossed this B3. An end-E counter
write before B3 can overtake a slow warp's E read; updating before B1 can likewise
overtake an H read. Field storage disjointness does not make metadata races safe.

There are V=2325 owned cells, P=3072 cached cells, L=747 incoming and S=645 outgoing
cells. Four permanent shared fields use 49,152 bytes; bases/counter bring this to
49,220 bytes, requiring dynamic shared opt-in. Two CTAs plus NVIDIA's 1 KiB/block
reserve consume 100,488 of 102,400 shared bytes/SM. Two 512-thread CTAs would supply
32 warps if total registers fit ≤64/thread; this compares with 24 warps for a
three-CTA 256-thread B7 control, whose actual residency depends on its compiled
resources. Twelve retained state scalars leave substantially more temporary
budget than the full-paper all-register design, but compilation is still a gate.

The trade increases opposite shared reads to 16 useful scalars/update, plus four
own-component shared reads; it lowers retained registers and eliminates frame
copies. Shared writes are 4(V+L)=4P per step. None of this changes global field
payload for this geometry: steady payload is 14.369 B/update. Finite predictions
include initialization, final scatter and invalid zero trail stores:

| N | T | Field B/update | Visited/valid | Four-rank trail MiB |
|---|---:|---:|---:|---:|
|512|8|21.873|1.079|281.1|
|512|16|19.119|1.186|612.2|
|640|8|21.754|1.063|436.5|
|640|16|18.746|1.131|914.6|

`prove_cuda_circular_asymmetric.py` passes nine low/interior/high-owner T2/8/16
cases and rejects seven controls: early imports, stale H, divergent shuffles,
wrong modulo-6 wrapping, all-owned publication loads, early counter update and
missing E counter subtraction. The phase proof uses adversarial warp ordering.
These are token/lifetime proofs and requested-payload estimates; compile fit,
CUDA correctness and complete-physics performance remain separate requirements.


The corrected two-CTA prototype passed the CPU compiler gate: all T4/8/16/32
specializations use 64 registers, zero stack/spills and 49,220 shared bytes.
Frozen provenance is in
`.cache/cuda-large-domains-2026-11-11/asymmetric-stable-two-ctas-v2/manifest.json`
(binary SHA `61dc6650c8db934d12f0e8b0e520aef33151ace61ac046542e4be4867ad76cfd`).
This establishes compiler feasibility only. The phase-counter draft was corrected
before GPU use; later GPU correctness/performance must be reported separately.
The numeric predictions above are reproducible with
`model_cuda_systolic_traffic.py --sizes 512 640 --steps 8 16`; the `asymmetric`
geometry is in the catalog, and `asymmetric-catalog-model.json` records this run.

## Finite-domain numeric predictions

`scripts/model_cuda_systolic_traffic.py` factors exact rectangular intervals over
the complete grid. It includes six staggered supports, version-zero loads,
final field scatter and the current code's unconditional zero-filled trail
publication for invalid physical slots. Four-rank reuse changes allocation,
**not the number of publication stores**. These T8 predictions use 4/8/32:

| N | Visited/valid field slots | Requested original GB | Broad cache GB | Circular cache GB | Invalid zero-publication GB |
|---|---:|---:|---:|---:|---:|
|256|1.244|6.314|4.394|3.919|0.288|
|384|1.159|20.938|14.464|12.863|0.633|
|512|1.118|49.203|33.866|30.075|1.112|
|640|1.094|95.606|65.664|58.261|1.726|

GB means 10^9 bytes. Compiler CSE, L1/L2 residency, 32-byte sectors, writes and
metadata/spills can move actual DRAM bytes in either direction. The model's
optional 700 GB/s assumption yields only an all-payload-transfer illustration:
at N256, 9.02/5.60 ms for original/circular; at N640, 136.58/83.23 ms. These are **not
runtime predictions**; barriers, instruction throughput and full physics remain.

| N | Direct trail GiB | Four-rank trail GiB | Four field banks+ring GiB | Two banks+ring GiB |
|---|---:|---:|---:|---:|
|256|1.367|0.082|1.599|0.840|
|384|4.292|0.173|5.276|2.724|
|512|9.803|0.299|12.370|6.335|
|640|18.721|0.459|24.007|12.233|

GiB means 2^30 bytes. The current four-bank comparison harness cannot safely
admit 640 on 24 GiB once reserve/context is included. That is not a two-bank solver
capacity bound. Reusing a reference workspace after its oracle is read back is
a testable harness improvement. CPML12 packed psi, codes, sources, DFTs and
staging must be included in real admission; the model reports separate psi/code
scenarios without pretending they implement those features.

## Owned in-place global field bank: an independent anti-dependency proof

An owned moving solver can potentially alias initial and final field storage.
Let P be a producer box and C a distinct same/later-rank box. There must be an
axis with `C_index > P_index`. Along that axis, for T≥1:

```
producer final maximum = P_origin+B-1-T
                      ≤ C_origin-2
                      < C_origin-1 = consumer initial minimum
```

Thus P's final writes cannot overwrite any initial cache value needed by C.
Earlier ranks have already finished. The same box must retire **all** version-zero
global reads before its own final scatter. Staggered supports and physical-wall
clipping only shrink the sets. Injective component/physical storage is assumed.
For a larger negative initial halo h, the sufficient condition becomes T≥h;
arbitrary custom read signatures cannot inherit the one-cell admission blindly.

`scripts/prove_cuda_moving_inplace.py` checks 12 rectangular/staggered T1/2/4
cases, all producer-versus-future-consumer read/write sets, reverse same-rank
box order and an aliased global bank with version tags. Every initial read sees
version 0; every final field value is written exactly once at T. A T1/two-cell
initial halo and a wrong final-coordinate shift both fail. This is not yet a
CUDA alias experiment or a permission to mutate immutable caller inputs.

Immutable caller/preserve snapshots, concurrent external initial readers,
material/monitor buffer aliasing and late version-zero reads remain excluded.
Deferred observations require separate chronological raw staging. At N640/T8,
one owned field bank plus the ring is about 6.346 GiB, compared with 12.233 GiB for
two banks; actual CPML/coefficient/monitor allocation still has to be admitted.

The separate [packed CPML derivation](cuda-moving-cpml-inplace-2026-11-11.md)
proves a stronger auxiliary-state result: each physical psi point has a
monotone chain of owners, so local recurrences can remain in a single mutable
packed global bank. No dense twelve-psi cache or psi trails are required.
Ordinary mutable loads/stores and the existing phase/rank barriers are essential;
read-only psi caching and historical/nonlocal psi consumers are not admitted.

## Bounded larger-case recommendation

For N512, paper-scaled 7/7/15 T8/T16/T32 gives field payload
24.780/21.648/20.558 B/update, startup+final 7.196/3.598/1.799 B/update, visited
ratios 1.075/1.135/1.229 and ring 392/816/1722 MiB. Balanced 8/8/16 gives
22.778/19.712/18.672 B/update and visits 1.094/1.161/1.268. T32 saves only about 5%
payload against T16 while approximately doubling ring storage and adding 8–9%
visited work. Three controlled larger cases suffice initially: N512 paper-scaled
T8, the same T16, and balanced T16. If half-shared resource admission succeeds,
its larger geometry should replace a case rather than create a broad grid search.

## Reproduction and decision

```sh
python scripts/model_cuda_systolic_traffic.py --sizes 256 384 512 640 --steps 8 \
  --output .cache/cuda-large-domain-first-principles-2026-11-11/model.json
python scripts/prove_cuda_circular_cache.py
python scripts/prove_cuda_circular_geometry.py
python scripts/prove_cuda_moving_cpml_inplace.py
python scripts/prove_cuda_moving_inplace.py
python scripts/prove_cuda_circular_half_shared.py
python scripts/prove_cuda_circular_warp_cache.py
python scripts/prove_cuda_circular_asymmetric.py
/home/quentinwach/.cache/beamz-pr233-ci/bin/python -m pytest -q \
  tests/unit/test_cuda_systolic_cost_model.py
```

The CPU tests cross-check formulas against trace enumeration, finite-domain
factoring against independent brute force, circular negative controls and the
owned in-place anti-dependency condition.
All 13 tests pass; Ruff passes. The model JSON records source URLs and the audited CUDA
SHA. `model-depths.json` provides analytic T8/16/32 cases, not a tuning result.
No GPU was used for this analysis and no production/kernel source was edited.

The concrete engineering order is: phase-correct broad faces as a controlled
baseline; circular physical retention with the proved publication read footprint;
balanced FP32 geometry and immutable material arrival; then packed CPML12,
absolute source clocks and raw observation staging with chronological DFT.
Measure exact state parity, resources, useful work, requested versus measured
traffic and complete D2 performance at every stage. Increasing depth alone is
not the proposed correction.
