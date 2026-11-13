# Private physical CPML state in an owned moving tile

This is a proved storage proposal and an isolated compiler resource prototype.
It is not an enabled engine path or a measured speedup. The current complete
moving engine is still slower than native D2 on the tested complete workloads.

The initial region class is exactly `flags in {2,4,8,34,36,40}`: no source or
observation, exactly one CPML axis, optional physical boundary handling. Keep
the real component `Valid` predicate and CPML12 active predicate. Only two H
and two E recurrences use that axis, so four FP32 states per physical cache
point suffice. Other recurrences retain the existing global fallback.

For componentwise ownership `owner(p,n)=floor((p+n)/B)`, a physical point's
owners are monotone. Distinct owners therefore have strictly increasing rank.
A point on an owner's high union is its last owned version in that tile.
Publishing its next ψ immediately from a private scalar cannot race another
same-rank owner; later-rank kernels wait for the complete producer kernel.

An eager cache can load ψ0 at startup and ψn on incoming low faces. For every
startup point in `[origin−1,origin+B−1]`, `owner(p,1)>=tile`, so earlier ranks
have not overwritten ψ0. For a low-face point at version n, each low axis has
`p+n=origin−1`; `owner(p,n+1)>=tile`. Its version-n producer is on its high
surface, and no earlier rank can have advanced ψ beyond n before the import.

The better lowering loads only on the first **owned** use: `n==1` or any local
owned coordinate is zero. At n>1 the preceding owner has published ψ(n−1)
on its high union. Subsequent owned versions use the same physical thread/slot.
This removes auxiliary startup/halo loads and auxiliary low-X staging entirely.
Store the new value privately; publish globally only on the owned high union
or at the terminal version. Final publication covers every valid active point.

Packed addressing is term-specific: derivative-axis extent is `N+Phase`,
coordinate `<12` maps directly, and the high band maps to `coordinate+24−extent`.
Other coordinates and dense packed-slab pitches remain unchanged. Each term
has separate mutable storage. Inactive or invalid slots must not evaluate a
private-state load, packed pointer, or publication store. No historical/spatial
ψ readers or per-step auxiliary observers are admitted. Ordinary mutable global
loads are required; read-only cached loads are inappropriate.

Private scalar publication can occur during the H/E recurrence store: there
are no shared ψ readers and no extra CTA barrier is needed. Shared field
publication retains its existing high-union footprint and synchronization.
Immutable caller snapshots and continuation/chunk boundaries retain their
existing lifetimes; each completed macro publishes a complete final ψ slab.

The cache 6×16×32 source requirement is 24 extra private floats per thread.
That is not a ptxas register count. The resource-only copied header passes four
scalar references per unrolled j, never a whole-array pointer. The first object
keeps runtime dispatch uncached: fixed-axis resource entries use 127 registers,
without stack/spills. A second object uses the stronger single-axis active fact
and a CTA-uniform NoIO switch; both selected entries and the wrapper use 121
registers without stack/spills, with unchanged 49,820-byte shared payload. It
passes Parameters by const reference from the grid-constant wrapper. Multi-axis
or zero masks retain the generic body. Both objects remain hardware-unvalidated.

`cuda_predicated_state_storage.py` derives a stronger register allocation from
the actual guarded recurrence IR. The twelve states form three disjoint
four-state interference cliques under the one-axis predicate. Four colors are
optimal; without that predicate twelve are needed. H terms use colors
`[0,0,1,0,1,1]`, E terms the same plus 2. H/E histories cannot share colors
because both survive the phase boundary. Logical state/profile IDs and global
arrays stay distinct. A third copied lowering uses one colored single-axis body
and a generic multi-axis fallback, avoiding three separate inlined axis bodies.
The compiler rejects multi-axis activation and an axis changing during a point's
retained lifetime; real validity/active predicates remain in the generated body.
This object compiled at 127 registers without stack/spills. Its NoIO wrapper has
13,496 static instructions and 711 branch sites, versus 21,848/1,261 for the
three-axis wrapper. The selected colored body alone has 6,944 instructions.
These are static counts; no instruction-cache or runtime improvement is inferred.

The colored lowering was subsequently composed with the frozen sparse-observation
and parallel-policy parent as native v10. Actual depth-4/16 macros passed all 76
small-state checks, 13 nonzero-clock checks, and all 30 leaves on both large
workloads, including repeated execution. The measured screen nevertheless lost:
Guide512 took 6.5133 s versus the parent's 6.4012 s; Mie272 took 1.02855 s versus
1.0113 s. The candidate used at most 127 registers without spills. This is a
correct, hardware-validated storage transformation, but it is rejected as a speed
promotion. Reducing logical auxiliary traffic did not compensate for its added
private state and control flow in these measurements. Frozen native-v10 logs and
manifests remain under `.cache/cuda-engine-autoresearch-2026-11-11-2032/`.

The CPU coverage model checks the existing native flag histograms and coarse
valid-component counts exactly. At T16:

|Workload|Eligible tiles|Valid component updates covered|Active ψ updates covered|Selected ψ payload: direct → cached|
|---|---:|---:|---:|---:|
|Guide512|21,372|22.25%|82.91%|8.028 → 2.537 GB|
|Mie272|4,332|30.29%|56.42%|1.544 → 0.498 GB|

Counts refer to one macro and include ragged supports, all four selected
recurrences, first-owned loads and high/final stores. They predict reductions of
56.71% and 38.23% of total logical ψ payload, respectively. They are not measured
DRAM traffic, cache misses, timing gains, or an estimate of whole-engine speedup.

`prove_cuda_stationary_psi_cache.py` exhausts every relative startup/incoming
boundary class and checks sampled physical integer recurrences using actual
staggered packed keys. Missing H/E publication, missing first-owned refresh,
wrong E extent, halo publication, and future-version publication fail.
`analyze_cuda_stationary_psi_coverage.py` separately counts the full workload.
Artifacts are `stationary-psi-{cache-proof,coverage}.json` under
`.cache/autoresearch-4h-2026-11-11/`.

```sh
python scripts/prove_cuda_stationary_psi_cache.py
/home/quentinwach/.cache/beamz-pr233-ci/bin/python -m pytest -q \
  tests/unit/test_cuda_stationary_psi_cache.py \
  tests/unit/test_cuda_stationary_psi_coverage.py
```
