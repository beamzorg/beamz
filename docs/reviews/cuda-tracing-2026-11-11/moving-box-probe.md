# Moving-box CUDA mechanism probe

This is a standalone **field-only** experiment. It has no CPML, sources, DFTs,
material arrays or native BeamZ API overhead. Its plain CUDA reference is a
limited, simple solver, **not the best production D2 engine**. These results do
not establish an engine speedup. The complete-physics overlapping D4 experiment
remains slower than production D2 and is disabled by default.

## Implemented communication

At field version n, assign the physical cell to u=(z+n,y+n,x+n). Each CTA owns a
fixed 4×8×32 box in u and executes T=4/8/16 versions. All six fields occupy 24 KiB
of shared storage; 256 threads own four u-cells each. This moves the physical
subdomain rather than recomputing overlapping halos.

Each version prefetches old H/E at u−(1,1,1) into registers, then synchronizes.
H consumes old E at u−ones or u−ones+axis; a publication barrier follows H.
E consumes current H at u or u−axis; another barrier follows E. Old E remains
in registers. The next pre-overwrite barrier also protects previous publication
loads. Every valid physical field update is computed once. Invalid visited
slots are predicated; visited slots are reported separately from valid updates.

Boxes execute in increasing sum of their three box indices, with one ordinary
CUDA launch per rank. Same-rank boxes are independent. A producer completes all
T versions before a higher-rank consumer launches. There are no spinning CTAs,
persistent scheduling or residency-dependent progress assumptions.

Cross-box reads use the producer's high-face union. It has
S=1024−3×7×31=373 cells, including its edges/corners. The direct prototype stores
`[tile,version,component,surface]`. Version-zero reads use immutable initial
fields. Six different Yee component extents and zero exterior values are used;
this is a normalized uniform-coefficient problem, not BeamZ's full boundary
or material model. Generated canonical Yee arithmetic is compared bitwise
against separate explicit physical-coordinate CUDA equations.

Relative addressing is the selected lowering. Because every transformed read
offset is 0 or−1, local power-of-two masks and a lower-neighbor linear tile delta
replace repeated global integer division. An independent integer oracle checks
206,471 addresses and the high-face union's bijection.

## Evidence

All six fields, including logical padding and outer field canaries, match the
plain CUDA reference bitwise for N=17/33/128 with T=4/8/16. Direct relative also
passes N=256 at all three depths. There are zero invalid trail-address/rank
reads. Fields start from deterministic nonzero values; missing data is not
masked by an all-zero initial state.

On the local RTX3090, the initial N256 three-sample rejection/confirmation
screen gives:

| Depth | Plain CUDA ms | Direct moving ms | Plain/moving |
|---|---:|---:|---:|
|4|32.3835|14.9309|2.169|
|8|64.7795|29.6938|2.182|
|16|129.7050|59.0715|2.196|

Two fresh processes per variant in ABBA order confirm N256/T8 ratios 2.160 and
2.138. At N128, two-process confirmations give roughly 1.35–1.38 against the
same simple control. This supports investigating larger domains, but neither
the control nor this field-only workload substitutes for a full native D2
comparison. Within each process, plain kernels are timed first; these are
bounded screens rather than confidence intervals. CUDA-event timing excludes
allocation, initial-data copies and validation readback.

Direct relative uses 210 registers/thread, no local storage/spills and 24 KiB
shared. The generic initial implementation uses 254 registers and 96 B local
storage. The relative lowering improves the mechanism without changing its
communication graph. Two rejected resource experiments are archived: splitting
components over 768 threads reduces registers to 80 but causes 448 B spill stores
and 808 B spill loads/thread; forcing two resident 256-thread CTAs causes 320 B
stores/348 B loads. Neither spilled layout is used for the performance result.

## Bounded trail ring

A separate four-rank ring uses an immutable rank-local slot map. Its address is
`[(rank%4)*max_frontier+slot[tile],version,component,surface]`. All cross-box rank
gaps are 1..3, so a same-stream rank launch finishes the last consumer before
rank r+4 reuses the epoch. Each slot has an explicit producer-owner tag; every
cross-box read verifies the expected producer. This detects stale reuse without
assuming old data happens to be harmless.

Ring4 passes N17/33 across all three depths and N256/T8 bitwise. The deliberate
two-epoch negative control rejects with 146,593 stale-owner reads and 224,922
field mismatches. Nonzero random initial state is retained. Unsupported epoch
counts fail compilation; two epochs require an explicitly named unsafe-control
macro/CLI flag and cannot be admitted as a valid benchmark.

At N256/T8, the ring reduces trail storage from 1,468,271,232 to 87,657,984 bytes
(16.75×), and admitted total GPU payload from 3.098 to 1.717 GB. It loses time:
fresh-process medians 47.865/48.113ms versus direct 30.001/30.303ms. Its 254
registers and 176 B local storage erase part of the communication gain. The direct
relative implementation remains selected. The ring is a validated bounded-memory
mechanism, not a performance promotion.

Direct and ring implementations each pass memcheck, synccheck and racecheck on
N17/T8 with zero errors/warnings. The kernel filter is `kns=Moving`. Sanitizer
times are excluded from performance results. **The historical timing binaries did not initialize/check trail guard padding**;
canary parity in those binaries covers the six field allocations.
Sampled memcheck and explicit owner/rank/surface checks provide separate trail
evidence and do not imply exhaustive memory-safety coverage. The final host-only
revision adds initialized trail guard bands and checks output/owner counters
again after timed repetitions. That final host/admission build uses 213 registers
without spills. Its direct/ring small gates pass, as do the final N256 direct
and hoisted-ring repetitions, including field/trail guards and zero accumulated
bad-owner reads. These checks are separate from historical performance binaries.

## Reproduction and provenance

```sh
python scripts/probe_cuda_moving_box.py --nvcc /path/to/nvcc \
  --output-dir /tmp/moving-direct
# Explicit GPU execution, after choosing an otherwise idle GPU:
/tmp/moving-direct/probe 256 8 5
python scripts/probe_cuda_moving_box.py --nvcc /path/to/nvcc \
  --trail-ring 4 --output-dir /tmp/moving-ring
# Optional measured bounded-memory lowering:
# add --hoist-ring-bases
python scripts/prove_cuda_moving_box_address.py
python scripts/prove_cuda_systolic_schedule.py
python scripts/prove_cuda_systolic_ring.py
```

Compilation is the wrapper's default; GPU execution requires `--run N T` or
running the binary explicitly. Each build copies its source/generated cell
header and records their hashes, compiler/version/flags and binary hash.
N≤256 is admitted only when `cudaMemGetInfo` covers exact field/trail payload
plus a 256 MiB reserve. Direct N256/T16 needs approximately 4.743 GB; this growth
motivates the bounded ring despite its present register penalty.

Raw artifacts are under `.cache/cuda-autoresearch-2026-11-11/`:
`moving-box-v1`, `moving-box-v4-relative`, `moving-box-final-relative`,
`moving-box-ring4-guarded`, `moving-box-ring2-guarded-negative` and
`moving-box-summary.json`. Final source is `scripts/cuda_moving_box_probe.cu`;
`moving-box-final-direct-repro` records the final source/direct compile;
`moving-box-ring4-hoisted` contains the hoisted-ring source, final checks and
`balanced-final-256.json`. Final source SHA256 is
`bd90a9c6484b1561795a5233a4031d8c78cae4cb94f7f34aa67f990bc2f07631`.
Rejected resource-only builds have their own compiler logs/manifests. No
production Python/native kernel, CMake default or installed component is changed.

The CPU CPML scheduling model covers logical full-support recurrence values,
not packed psi slab storage or physical CPML admission. A real next backend must
integrate packed CPML12 ownership, dynamic physical masks, sources and exact DFT
timing/order, then beat the complete native D2 workload. This probe supplies a
working communication mechanism and bounds for that work, not those features.

An optional final lowering precomputes the eight lower-neighbor trail bases
and producer-owner checks once per CTA into 64 bytes of shared storage. It was
frozen at 17:42 and compiles to 196 registers with zero local storage/spills. It
passes all six N17/33 depth gates and three sanitizers. N256/T8 fresh-process
medians 37.001/37.003ms improve the old ring but still lose to direct 29.585/29.870ms.
Both remain compared against a simple plain solver, not native D2. Owner
prevalidation covers all geometric in-grid predecessors, so its deliberately
unsafe E2 rejection is conservative; the earlier per-read E2/value mismatch is
the stronger pending-reader witness. Pruning whole-box inactive version intervals
remains unimplemented and needs a proof that first-active reads cannot consume
uninitialized same-box state.
