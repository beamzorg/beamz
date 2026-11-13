# Packed global CPML state with moving-box ownership

Standard CPML auxiliary state can remain in one mutable packed global bank
while fields use a moving cache. It does not require dense on-chip psi arrays,
psi halos or versioned psi trails. This follows from the **local recurrence and
unique ownership**, not from the field-only speed result. It is a proposed
full-physics integration with an independent CPU ordering proof; no production
or CUDA implementation was changed for this analysis.

## Ordering argument

For a physical point p, derivative term t and field family F, standard CPML has

```
psi[F,t](p,n) = b[F,t](p)*psi[F,t](p,n-1) + a[F,t](p)*D[F,t](p,n)
```

The corrected derivative/field consumes the newly computed psi immediately.
There is no spatial psi derivative or foreign-point psi reader. H and E have
separate auxiliary banks and phases; their field-gradient inputs still obey
the existing Yee schedule and require field caches/trails.

The owner is `O(p,n)=floor((p+n)/B)`, componentwise. Increasing n increments
each owner coordinate by zero or one. Therefore:

1. A point's owner sequence is componentwise nondecreasing.
2. Once it leaves a box it never returns; a box owns a contiguous time interval
   for that point.
3. Distinct successive owners have strictly increasing rank. Distinct boxes
   of the same rank cannot update the same physical auxiliary word at *any*
   versions.

An owner processes its local versions in increasing order and leaves global
psi at the last version it owns. The next owner launches at a later rank and
reads exactly n−1. Within one owner, the next timestep may assign the point to
a different thread, so CTA memory ordering is required. Existing H/E barriers
provide this; across owners, ordered rank-kernel completion provides it.

Unlike overlapping temporal tiles, this schedule never recomputes the same
auxiliary point/term/version in multiple CTAs. That is why the global bank can
be updated directly instead of freezing an input bank for private halo histories.

## Actual staggered packed faces

For each family/component, use its true Yee dimensions. For term axis k and
logical stop N_k, physical CPML support is the union
`[0,12)` and `[N_k-12,N_k)`. The packed coordinate is

```
low:  packed = p[k]
high: packed = 12 + p[k] - (N_k-12)
```

The two sides occupy disjoint packed intervals 0..11 and 12..23. Other coordinates
retain their component's true extents and descriptor pitches. Each derivative
term and family has disjoint storage. Edges/corners update several **different**
terms, not the same word. E's extra terminal nodes change its logical stops;
using a single unstaggered N for all terms is not admitted.

Physical support is fixed for p even though its moving box crosses a CPML
interface. Inactive points have no packed recurrence to transfer. Dense and
padded-pitch mappings are injective on their physical support; allocated padding
is excluded from physical updates. The proof cases use disjoint low/high supports.

## CUDA and arithmetic contract

- Load old psi through an ordinary **mutable** global pointer. Do not use
  `__ldg`, a texture/read-only path or a false const/restrict alias assumption:
  the same word changes within the kernel.
- Compute new psi once and use that register value in its corrected derivative.
  Preserve the existing multiply/FMA order, signs and BF16 round-trip nodes
  where applicable. Storing psi must not cause a second expression to reread
  the newly stored value as if it were old psi.
- H and E auxiliary banks remain separate. Complete each phase's global stores
  before the existing CTA phase barrier; keep every lane at the barriers,
  including partial-tail lanes.
- Preserve same-stream rank ordering, including when replayed as a graph.
  This proof does not allow simultaneously executing dependent rank frontiers.
- There is no need for atomic updates under these ownership conditions.
  Distinct same-rank writers address different packed words.

CUDA's [synchronization documentation](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#synchronization-functions)
specifies block visibility for prior global/shared accesses at a CTA barrier.
It also distinguishes read-only cached loads. These guarantees do not replace
rank ordering or the physical ownership proof. Shared-memory racecheck alone
does not establish absence of global data races.

## Independent oracle and failure controls

`scripts/prove_cuda_moving_cpml_inplace.py` constructs actual physical low/high
slabs and packed indices for all twelve H/E derivative arrays. Each word carries
its physical point and latest global timestep. It groups updates by owner,
local timestep and H/E phase, checking every read against n−1 before advancing.
An independent per-point rank map rejects distinct same-rank ownership.

Positive cases cover dense/padded pitches and actual CPML12 on 25×27×29 with
staggered component extents. That case has 217,344 distinct packed words. A 2+3
continuation performs 1,086,720 ordered updates and 188,248 owner transitions,
with no same-rank conflict. Every word ends at the final global clock.

Reverse ranks, reverse local time, overlapping low/high packed coordinates and
aliased H/E banks all fail. The test is an integer/tag ordering oracle, not a
numerical replacement for native FP32/BF16 field/psi parity.

The local-consumer condition is essential. A hypothetical stencil that updates
psi(q, n) and subsequently asks for psi(q, n−1) at another point in the same
phase has already destroyed that old value. Deferred historical psi observers
likewise require a snapshot/trail. Standard CPML has neither dependency.

## Cost and admission

For twelve packed derivative arrays, CPML12 has asymptotic state size
`12*24*N²*4 = 1152*N² bytes` per bank. The exact staggered calculation gives:

| N | One psi bank MiB | Requested FP32 psi read+write B/basecell-step |
|---|---:|---:|
|256|72.281|9.035|
|384|162.422|6.016|
|512|288.563|4.509|
|640|450.703|3.606|

The traffic column counts one load and store per active word per step. It is
not measured DRAM: repeated owner-local updates may hit cache, while sectors,
metadata and profile loads add costs. Crucially, the auxiliary cost is O(N²),
and its bytes per volume update decrease as 1/N. It avoids adding twelve dense
psi channels to every interior tile's register/shared resource budget.

The model's owned one-field-bank + one-psi-bank + three byte E-code arrays +
four-rank field trail is approximately 7.521 GiB at N640/T8, excluding monitors,
sources, metadata and context. This is a layout estimate, not an implemented
full-solver allocation or admission result. Immutable caller/preserve snapshots
can require extra banks; they are separate from the working-state requirement.

## Integration gates

Use the real BeamZ packed term descriptors, profile samples, field staggering
and metallic boundary behavior. Keep 12 cells at every resolution. Do not replace
CPML by the paper's adiabatic absorber. Start with existing FP32 source and
observation contracts; capture raw field samples when computed and replay DFT
updates chronologically. External historical psi readers are not admitted.

Before promotion, compare all fields, twelve psi arrays, sources, DFTs and clocks
against the frozen full native reference, including nonzero initial state,
odd/split continuations and physical corners. Validate global index bounds and
source-level ordering in addition to bounded CUDA sanitizers. Then measure the
complete workload against the best D2 path; the field-only control is insufficient.

```sh
python scripts/prove_cuda_moving_cpml_inplace.py
python scripts/model_cuda_systolic_traffic.py --output /tmp/traffic-model.json
```

Raw proof output is
`.cache/cuda-large-domain-first-principles-2026-11-11/cpml-inplace-proof.json`.
