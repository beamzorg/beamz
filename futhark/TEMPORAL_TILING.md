# Temporal tiling for the Futhark backend — design notes

**Files (2026-10-06):** the experiment harnesses named below (`tiling/bare*.fut`,
`checker.py`, `shapes.py`, `cost.fut`, `dct.fut`, `compress_study.py`,
`lean.fut`, `gen.py`, `bench.py`, the probes, and the CPML tile harnesses and
patches) were removed once their results were written up here. The tracked
ones are in git history: `git show 9832ae7e:futhark/tiling/<file>`. Kept:
`tiling/compare_temporal.py` (exactness check), `tiling/core_bench.fut` (core
kernel harness) and `tiling/fc.sh` (builds it with the Futhark checkout).

**Status (2026-10-06, evening):** both remaining "obvious" gaps were
measurement artefacts. This GPU streams 294 GB/s (copy) and 361 GB/s (read),
not 420, so the core kernel moves about 70 B per cell and pass against about
60 B its reads and writes need, not 2× the minimum. Edge cells are not 2×
slower than plain stepping either: the edge machinery costs about 1.3× plain
per cell, and the rest is that the edge is mostly CPML cells, which cost more
in any scheme. On this GPU the K = 2 design is within about 25% of its limits
in both kernels. See [Calibration](#calibration-how-far-from-the-limits-2026-10-06-evening).

**Status (2026-10-06, later):** the lean CPML kernel is integrated in a
split state (`tiling/split_state.patch`, not applied): bit-exact on every
`compare_temporal.py` case, but slower than the committed tiling, 2.47 vs
4.48 GCUPS with one wall and 1.85 vs 2.91 with all walls at 128×256×512.
In production the CPML kernel runs at 0.8–0.9 ns per cell-step, not the
harness's 0.4–0.5, and the split state's lookups make the core kernel 28%
slower. Even at harness speed it would only tie with one wall. CPML tiles
do not pay for the target, large domains with a thin CPML shell (and the
patch fails at 256×512×512, not investigated). See
[CPML tiles in production](#cpml-tiles-in-production-a-split-state-2026-10-06);
current next steps are in [Where to resume](#where-to-resume-2026-10-06).

**Status (2026-10-06):** a lean CPML tile kernel (`tiling/cpml_bench.fut`,
not in production yet) is bit-exact and runs CPML regions at 0.34–0.39 ns
per cell-step with 16×16 or 8×32 jobs, against 0.53–0.77 for plain stepping
and 0.5–0.75 for the production edge kernels. It is limited by register
pressure (one block per SM), not by halo bandwidth: the memory controller is
43% busy and removing all halo traffic saves 6–17%. A checkerboard would gain
at most about 15% on core tiles and about 25% on CPML tiles. See
[Lean CPML tiles](#lean-cpml-tiles-halo-and-registers-2026-10-06).

**Status (2026-10-05, night):** the core kernel turned out to be bound by
DRAM bandwidth, much of it re-reading neighbours' halo. Tiles are now 16×32
with core cells chosen per 8×16 quadrant: tiled 4.48 GCUPS at 128×256×512
and 4.46 at 256×512×512 with one CPML wall (were 4.26 and 4.15), 2.91 and
3.31 with all walls (2.64 and 3.02), still bit-identical; the smallest shapes
got slower. See [Core kernel: bandwidth, not latency](#core-kernel-bandwidth-not-latency-2026-10-05-night).

**Status (2026-10-05, evening):** the production path
(`BEAMZ_FUTHARK_TEMPORAL=2`) uses the core-plus-plain-edges design and is
bit-identical to plain stepping on every deterministic case. Since the memory
work it needs *less* device memory than plain stepping (59 vs 71 B/cell at
256×512×512, was 155), so the large shape no longer pages: one wall 4.14 vs
2.98 GCUPS plain (was 1.31), all walls 3.05 vs 2.67. With CPML on all six
walls of the small shapes it only ties: the edge there is about 36% of the
cells, mostly CPML. See [Memory per cell](#memory-per-cell-2026-10-05-evening)
and [Production integration, 2026-10-05](#production-integration-2026-10-05).
Since 2026-10-05 the backend builds with the patched Futhark checkout
(`../futhark`) instead of 0.27.1 plus generated-code patches, and the core
kernel's window rings hold one plane per component (the interleaved layout
lost 5% on the newer compiler): tiled 4.18 GCUPS at 128×256×512 and 3.93 at
256×512×512, one wall. See FUTHARK_ISSUES.md, "Moving to the checkout", and
[CPU, 2026-10-05](#cpu-2026-10-05) for the CPU.
Earlier sections are kept as history; the 2026-10-04 status and next steps
are superseded. The time-skewed checkerboard (a model only, so far) is exact
with about K/2 levels of border history and a K/2-cell halo, at roughly the
synchronous version's redundant work; see that section. Priorities were
re-ranked after reading two LRnLA (diamond tiling) papers, for large domains
where CPML is a thin shell; see [Related work and re-ranking](#related-work-lrnla-diamond-tiling-and-re-ranking-2026-10-05).

## Motivation

A CPU comparison on a Ryzen 9 7950X (FDTD-Bench study
`benchmarks/20261003-zapfdtd-beamz-cpu/`) ran the same 5.7–14.8M-cell
silicon-block grid (32-cell CPML, one point source, no monitors). Results are
median GCUPS of the stepping loop only:

| Solver | GCUPS |
|---|---:|
| ZapFDTD (native, 16 threads) | 3.09–3.10 |
| BeamZ Futhark `multicore` (32 threads) | 0.144–0.172 |
| BeamZ JAX CPU | 0.113–0.118 |

ZapFDTD's lead comes mostly from *temporal blocking*: a cache-resident block
advances 16 timesteps before it is evicted (ZapFDTD
`src/fdtd_block_schedule.h`). At 3.1 GCUPS and an assumed ~65 GB/s of DRAM
bandwidth, Zap moves at most ~20 B per cell update. Merely reading and
writing six float32 components once costs 48 B, so Zap must reuse data across
timesteps.

This note asks whether a temporally tiled schedule can be written *cleanly* in
Futhark: as a regular `map` over tiles with a sequential `loop` inside,
without a hand-written scheduler.

## Step 0: understand the current baseline first

The current program (`fdtd.fut`) is not yet at its own streaming limit.
*Estimate:* each `phase_next` reads 3 old components, 3 curl inputs, 3 dense
`decay` arrays and 3 material/source arrays, and writes 3 components. That is
~60 B per phase and ~120 B per step, plus CPML slabs. At ~65 GB/s that bounds
a one-step-per-sweep schedule near 0.5 GCUPS; the measured value is 0.17.

The gap suggests per-cell work dominates: `inside`/`pec` tests, `metric`
dispatch, `packed` slab lookups in `stretched`, material decoding, and
`reduce_by_index` injection. Temporal tiling only pays once a kernel is
memory-bound. So before tiling:

- run `BEAMZ_FUTHARK_PROFILE=1` and measure bytes per cell with `perf stat`
  (or AMD uProf) on a multicore build;
- check whether interior-only specialisation (no PML/PEC branches) moves the
  baseline toward the streaming bound.

## Options considered

| Scheme | Fits Futhark? | Why |
|---|---|---|
| Zap-style skewed staircase with async dataflow | No | Needs atomic arrival counters, a work queue and in-place updates ordered across blocks. |
| Recursive cache-oblivious trapezoids (Frigo–Strumpen) | No | Futhark has no recursion. |
| Pyramids, then inverted pyramids (diamond tiling) | Poorly | 3D needs d+1 = 4 phases of differently shaped pieces (cells, faces, edges, vertices). Leapfrog overwrites in place, so each pyramid's sloped surface must be saved level by level for the next phase. That means ragged storage plus scatters. |
| **Overlapped (ghost-zone) trapezoid tiling** | **Yes** | One phase per pass, regular shapes, disjoint outputs. It costs redundant halo compute. |
| 2.5D streaming (xy tiles, rolling z planes) | Maybe | A sequential z loop with a few planes per stage; GPU-friendly but more intricate. |

Overlapped tiling is the "pyramid" idea in a form Futhark can express: each
tile redundantly computes the pieces the inverted pyramids would have handled.

## Design: overlapped trapezoid tiling

### Geometry

- Split the (padded) grid into tiles with a core of `b³` cells. Each tile
  loads a window of `(b + 2h)³` cells, where `h` is the halo and `k` the
  temporal depth (steps per pass).
- In Yee, `H(p)` reads `E(p)` and `E(p+1)`, and `E(p)` reads `H(p)` and
  `H(p-1)`. One full step therefore invalidates one cell at each window edge
  (H loses the top layer, E the bottom). After `k` steps the valid region is
  the window shrunk by `k` on every side, so `h ≥ k`.
- Monitor interpolation reads neighbours of the sampled point, so sampling
  after the last local step needs one more valid layer: use `h = k + r`, with
  `r` the interpolation radius (1 for the current trilinear plans).
- Reads outside the window may return anything (clamped index or `rotate`
  wrap). Those values only contaminate layers that are cropped.

### Exactness

Cells in the halo repeat the same arithmetic on the same inputs. The cropped
core should therefore be **bitwise identical** to plain stepping, provided
every decision uses *global* coordinates: `inside`, `pec`, `metric`, CPML slab
`packed` indices, material lookup and source targets. Bitwise equality with
the existing `program` entry point is the test oracle.

### Futhark shape

The following is pseudocode, not compiled:

```futhark
-- One tile: gather a window, run k full steps on it, return the core.
def run_tile (k: i64) (h: i64) (b: i64) (origin: (i64, i64, i64))
             (global: state) (co: coeffs) (sources: tile_sources) : tile_result =
  let w = b + 2*h
  let s = window global origin w            -- E, H, psi for this window
  let c = window_coeffs co origin w         -- decay, material, metric slices
  let (s, dft) =
    loop (s, dft) = (s, zero_dft) for t < k do
      let s = h_phase c origin s |> inject_h sources t
      let s = e_phase c origin s |> inject_e sources t
      in (s, sample_core_monitors origin s t dft)
  in (crop h b s, dft)

def pass k h b st co = untile (map (run_tile k h b ...) tile_origins st co)
```

The outer driver loops `nsteps / k` passes, plus one tail pass of depth
`nsteps % k`.

- **`multicore` / `ispc`:** the outer `map` is parallel across cores and the
  inner loop runs within one core. This is exactly the intended schedule.
  *Estimate:* a 40³ window × 6 components × 4 B ≈ 1.5 MB, which fits per-core
  L2/L3 on Zen 4.
- **CUDA/HIP:** this pays off only if Futhark generates an *intra-group*
  kernel that keeps the window in shared memory. That needs an inner parallel
  extent of at most the workgroup size (≤ 1024 threads) and a window that fits
  in ~100–228 KB. A 3D `b = 16` window is 4096 cells, so expect to need 2D xy
  tiles or 2.5D streaming. Check the profile to see whether you get one kernel
  per pass or `k` global-memory kernels.

### State inside a tile

- **Fields:** six components sliced from the common padded run storage
  (`pad P0 P1 P2`). Pad the run storage up to whole tiles plus halo once,
  outside the step loop.
- **Coefficients:** `decay` and material slices. Scalar or size-1
  broadcasting stays as `coefficient` handles it today.
- **CPML ψ** is the awkward part. The ψ arrays are *packed slabs*, written by
  `scatter` through `packed`/`with_coord`. Options:
  1. **Two tile kinds:** interior tiles carry no ψ. Boundary tiles unpack
     their ψ slab region into dense window-local arrays (zero elsewhere),
     advance them inside the loop, and scatter the core back into the packed
     slabs. These are two `map`s, but still one pass, because the outputs are
     disjoint. *(Preferred.)*
  2. Dense full-grid ψ. This is simpler, but 12 extra arrays roughly double
     memory and traffic.
  3. Tile the interior only and keep the existing per-step kernels for the
     PML shell. **Avoid this.** It is what BeamZ's CUDA two-step experiment
     did: CPML still advanced every substep, and the core–shell coupling band
     made it slower than the one-step schedule on realistic cases
     (`docs/reviews/cuda-realistic-temporal-integration-2026-09-18.md`). In
     the FDTD-Bench silicon block, about 80% of allocated cells are PML or
     padding, so the PML must be tiled too.
- **Sources:** these must be injected on the whole valid window, halo
  included, at the same H/E phase and step as today. Otherwise the halo
  diverges and the core result is wrong. Precompute a per-tile CSR source
  list on the host, padded to the maximum count per tile so the arrays stay
  regular. This beats filtering the global `K` list in every tile at every
  step.
- **Monitors:** each monitor point belongs to exactly one tile core. That
  tile samples it after every local step and accumulates the DFT in
  chronological order, which matches the current order. Return the per-tile
  partial accumulators and `reduce_by_index` them into `dft_re/dft_im`. The
  current `accumulate_dft` uses global flat indices, so remap them to window
  offsets.

## Cost model (*estimate*)

Traffic per pass is the loaded window plus the written core. Plain stepping
reads and writes the core every step. With halo `h = k`:

| b, k | Window | Redundant compute | DRAM traffic vs. `k` plain steps |
|---|---:|---:|---:|
| 32, 2 | 36³ | 1.10× | 0.61× |
| 32, 4 | 40³ | 1.32× | 0.37× |
| 32, 8 | 48³ | 1.89× | 0.27× |
| 16, 2 | 20³ | 1.21× | 0.74× |

Gains flatten past k ≈ 4 while redundant work keeps growing. Coefficient
arrays add load traffic without adding write traffic, which slightly improves
these ratios.

## Experiment plan

1. **Baseline profile** (Step 0). Record bytes/cell and per-kernel time for
   `multicore` and CUDA.
2. **Bare stencil prototype:** a standalone `.fut` with a periodic box, no
   CPML, sources or monitors, and homogeneous coefficients. This mirrors the
   bare CUDA experiment in
   `docs/reviews/cuda-cpml-tiles-and-bare-temporal-2026-09-18.md`.
   - Check `k ∈ {1, 2, 4, 8}` bitwise against plain stepping.
   - Sweep `b` on `multicore`, `ispc` and CUDA.
   - Report GCUPS from useful (core) cells only.
3. Add CPML with two tile kinds, then sources, then monitors. Re-check
   bitwise equality against `program` at each stage.
4. Integrate behind an opt-in attribute or environment variable (for example
   `BEAMZ_FUTHARK_TEMPORAL=k`). Keep `k = 1` as the default until it wins on
   the realistic workloads in `scripts/benchmark_futhark_jax.py`.

**Success criterion:** at least 1.3× over the best `k = 1` Futhark schedule on
the realistic CPU workload, with a bitwise-identical result.

## Open questions

- Does Futhark fuse the `k`-step inner loop into one kernel per tile on
  `multicore`, or does it materialise intermediate windows each step?
- On GPU, will incremental flattening pick the intra-group version for
  realistic window sizes? If not, is 2.5D streaming expressible without
  losing the clean structure?
- Can tile-local window slicing avoid a full copy, for example by indexing
  the global array with an offset in the first step?
- Is window-relative indexing with explicit global offsets cleaner than
  retrofitting `field_value`'s global-index derivatives?

## References

- ZapFDTD `src/fdtd_block_schedule.h`: skewed staircase, 4D Morton order and
  the dependency stencil. Its header still says "design document only", but
  `fdtd_block_schedule.cc` implements it.
- BeamZ CUDA temporal studies in `docs/reviews/cuda-*temporal*-2026-09-18.md`:
  bare two-step gives +27–32% on an RTX 3090; deeper tiles were slower; the
  realistic integration was slower than one-step.
- Frigo & Strumpen, "Cache oblivious stencil computations", ICS 2005.
- Nguyen et al., "3.5-D blocking optimization for stencil computations on
  modern CPUs and GPUs", SC 2010.
- Bondhugula, Bandishti & Pananilath, "Tiling stencil computations to
  maximize parallelism" (diamond tiling), SC 2012.
- Grosser et al., "Hybrid hexagonal/classical tiling for GPUs", CGO 2014.

## Results (2026-10-04)

All on one machine: Quadro RTX 5000 (Turing SM75, 448 GB/s nominal, 64 KB
shared memory per SM, 4 MB L2) and Ryzen 9 7950X; Futhark 0.27.1. Code is in
`futhark/tiling/`:

| File | Contents |
|---|---|
| `bare.fut` | Bare stencil (six components on one `nz×ny×nx` box, zero outside, homogeneous coefficients), the plain reference, a z-chunked generic-K tiled pass and a plain one-step pass on z-column tiled storage. |
| `bare2.fut` | Full-column streaming tile, generic K. |
| `bare3.fut` | Lean full-column streaming tiles with explicit K = 1…4 stages, plus an intra-block copy kernel as a streaming upper bound. |
| `bare4.fut` | **The working GPU design:** full-column streaming tiles whose planes live in fixed ring slots updated in place, for explicit K = 1, 2 and any K. |
| `patch_cuda.py` | Rewrote a generated intra-block kernel so the block result lives in global memory (see below). Removed: now the `#[intrablock_result(global)]` attribute of the Futhark checkout (FUTHARK_ISSUES.md). |
| `gen.py`, `fc.sh`, `bench.py` | Instantiate constants (`TY`, `TX`, `ZC`, `K`), compile with the Futhark checkout, and time two step counts so that setup cancels. |

Every tiled variant was checked bitwise against the plain reference on odd
shapes (e.g. 37×49×70) with zero differing samples, on `c`, `multicore` and
CUDA (CUDA with `--fmad=false`). GCUPS below are for 128×256×512, per step,
from the difference of two runs.

### Design actually used: 2.5D streaming, not 3D windows

A 3D `(b+2k)³` window is far over the 1024-thread limit of an intra-block
kernel. Instead each tile is an xy window of `(TY+2K)×(TX+2K)` that streams
z planes; stage `s` of `K` advances plane `z−s`, so stages lag by one plane
and only two planes per field and stage are live. On a CPU this is just a
`map` over xy tiles with a sequential z `loop`. A z chunk (GPU only) starts
`K` planes early so every stage is exact on the planes the next one reads.

### CPU (`multicore`): temporal tiling works

| Schedule | GCUPS |
|---|---:|
| Plain one-step (bare) | 0.42 |
| Plain one-step on z-column storage | 0.35 |
| Streaming tile, K=1 (H and E fused), 8×32 | 0.53 |
| K=2, 8×32 / best (32×64) | 0.97 / 1.05 |
| K=3, 8×32 / best (16×128) | 1.07 / 1.51 |
| K=4, 8×32 / best (16×128) | 0.94 / **1.55** |
| Generic-K pass with plane copies (`bare.fut`), best K=3 | 0.70 |

The best configuration is 3.7× the plain schedule, well past the 1.3× success
criterion, though for the bare stencil only. Two points matter for an
integration: planes must rotate through loop-carried tuples (the generic version
that copies planes between stage arrays loses half the gain), and wide tiles
(16×128, about 64 tiles for 32 threads) beat square ones.

### CUDA: ring buffers make K = 2 win

| Schedule (bare) | Block | Shared | GCUPS |
|---|---:|---:|---:|
| Plain one-step (`tabulate_3d`, 64-bit index split) | | | 3.62 |
| Plain one-step, z-column tiled storage, 32-bit indices | | | 4.0–4.24 |
| z-chunked intra-block tiles, K=1 / K=2 / K=3 (`bare.fut`) | | | 3.41 / 2.93 / 2.02 |
| Full-column, rotating tuples, K=1 / K=2 (`bare3`, patched) | 340 / 432 | | 3.32 / 0.86 |
| **Ring slots, K=1, 8×32 (`bare4`, patched)** | 340 | 16 KB | **5.01** |
| **Ring slots, K=2, 4×32** | 288 | 24 KB | **9.60** |
| Ring slots, K=2, 8×32 / 4×64 | 432 / 544 | 36 / 46 KB | 9.28 / 9.27 |
| Ring slots, K=3, 4×32 (one block per SM) | 380 | 59 KB | 6.29 |
| Ring slots, K=3 or 4 with larger tiles | | > 64 KB | does not launch |

Two Futhark behaviours decide this, and both are visible only in the generated
CUDA (`--dump-cuda`):

1. **Loop-carried arrays are double-buffered in global memory.** When an
   intra-block `loop` returns freshly computed arrays (rotating planes through
   a tuple), Futhark keeps the parameter in shared memory but hoists the
   buffer for the next iteration out of the kernel. Every plane is then copied
   to device memory and read back once per z iteration: two planes for K=1 and
   eight for K=2 in `bare3`. That, not barriers or occupancy, made those
   kernels slow. The fix is to keep each plane in a fixed slot of a
   loop-carried ring (`[2][WY][WX][3]`) chosen by plane parity, and update it
   in place (`ring[s, slot] = tabulate …`). Nothing is then double-buffered, and
   the barrier count drops to 6 (K=1) or 10 (K=2) per plane.
2. **The block's result is kept in shared memory** until the block ends, so a
   whole z column does not fit. `patch_cuda.py` points it at the block's slice
   of the global result instead (see the file's docstring). All CUDA numbers
   above for full-column tiles use it. Without it, the only option is z chunks
   with `(ZC+2K)/ZC` redundant planes, and the output chunk competes with the
   rings for shared memory.

   Measured with the ring design and no patch (`bare4.fut`, `CHUNKED=true`,
   4×32): K=1 reaches 4.15 / 4.19 / 4.03 GCUPS at ZC = 4 / 8 / 16, and K=2
   3.02 / 4.24 at ZC = 4 / 8. ZC ≥ 16 for K=2, and ZC = 32 for K=1, exceed 64 KB.
   **Without the patch, K=2 is no faster than plain one-step stepping**, so the
   patch is worth about 2.3×.

Other levers: only values whose in-plane neighbours are needed live in rings.
Old H and E at `z+1` are read from global memory, and the last stage computes E
on the core only, fused with the output write. Writing the output per cell
(`[TY*TX][6]`) matters: the intra-block block size is the largest inner `map`,
so a `[6][TY*TX]` output `tabulate` would have made blocks 768–1536 threads.
Removing `volatile` from Futhark's shared-memory declaration changed nothing
(K=1 3.34, K=2 0.86 in `bare3`).

The documented attributes give no other control over intra-block scheduling.
There is no block-size, coarsening or register-array attribute, and the block
size is always the largest inner parallel extent.

K=3 needs 11 ring slots (59 KB with 4×32 tiles), which leaves one block per SM
on Turing's 64 KB. GPUs with more shared memory per SM (RTX 3090: 100 KB,
A100: 164 KB, H100: 228 KB) may profit from K=3. That was not measured.

### Production one-step layout (reverted)

The plain z-column tiled storage gained 16% in the bare benchmark, so
`fdtd.fut` was converted to 8×32-tiled field storage, with offsets computed
once per cell and neighbours from fixed strides, plus source targets and
monitor plans retiled on entry. It passed the hardware tests but was not
faster on the realistic workload (`scripts/benchmark_futhark_jax.py`, 256
steps):

| Shape | CUDA | Futhark (main) | Futhark, tiled storage |
|---|---:|---:|---:|
| 64×96×128 | 2.04 | 1.74 | 1.60 |
| 96×160×256 | 2.48 | 2.35 | 2.17 |
| 128×256×512 | 2.60 | 2.54 | 2.59 |

H got faster (2.55 → 2.37 ms) and E slower (2.90 → 2.97 ms). Rounding the
common `(z+1, y+1, x+1)` storage up to whole tiles wastes 9% of cells at
128×256×512 and 24% at 64×96×128. Most of the bare gain probably came from
32-bit index decomposition, which production already uses. The change was
reverted.

### What this means

- On the bare stencil, Futhark reaches 9.6 GCUPS with K=2 on the Quadro RTX
  5000, 2.65× its own plain schedule. On the realistic workload, the CUDA backend
  manages 2.6 GCUPS and Futhark 2.54 today. Integrating K=2 with CPML, sources
  and monitors inside the tile is the path to beating the CUDA backend on GPU.
  It relies on `patch_cuda.py`, which `build.py` would have to apply. That is
  the same kind of generated-code patch `build.py` already applies to the CUDA
  context, but more fragile.
- On CPU the ring design performs about the same as rotating tuples (`bare4`,
  16×128: K=3 1.38, K=4 1.61 GCUPS), so a single tile design can serve both
  targets, with K≈3–4 and wide tiles on CPU and K=2 with 4×32 tiles on Turing.

## Production integration status (end of 2026-10-04)

The tiled path is live behind `BEAMZ_FUTHARK_TEMPORAL=2` (even step counts
only; odd counts still use plain stepping). Code: `temporal.fut` (kernels),
`fdtd.fut` (driver), `yee.fut` (shared cell update), `intrablock.py` (the
result-to-global patch, applied by `build.py`). Nothing is committed.

### Design

- **Core/shell split, Zap-style.** *Core* tiles have no CPML slab, source or
  boundary anywhere in their 8×36 window. Their interior z range runs in a lean
  kernel (`core_pass`) with plain Yee arithmetic. Everything else, the *shell*,
  runs in z chunks of `Zs` planes in a general kernel (`shell_pass`) with CPML,
  sources and boundaries.
- **Storage.** Fields are stored per kernel: core `[nc][Zc][128][6]`, shell
  `[ns][Zs][128][6]`. Both kernels read both arrays through `fget`/`locate`, and
  each writes only its own. CPML memory lives with the shell only:
  `[ns][Zs][128][12]`.
- **Shell rings.** H1 and E1 rings hold 9 values per cell: 3 fields and 6 CPML
  memories. Each stage writes its ring slot straight from one per-cell
  computation, with sources added inside that computation. This is what gets the
  shell into 64 KB of shared memory (58.7 KB plus E0); separate result planes
  and `unzip` temporaries don't fit.
- **Monitors.** DFT monitors sample the step a pass never writes out by
  recomputing it at the gather points (`recompute`, `dft_pass`).
- **Context.** All run data travels in three flat buffers (`fb` f32, `ib` i64,
  `kb` i32), addressed by offsets held in small scalar records (`loc`, `mat`,
  `mat.cf`, `srcs`).

### Compile time: solved (212 s → 63 s full build)

- **Front end.** Measure with `futhark dev -v -e`; almost all the time is the
  first `simplify` pass. Per piece today: core 10 s, shell 35 s, DFT 6 s, and
  the whole program 35 s, down from more than 120 s.
- **Root cause.** Closures capturing a big record. Defunctionalisation flattens
  a captured record (about 50 scalars here) into every closure, and every
  application copies all of them. The shell's lifted `run` had 23k lines of
  IR, mostly such copies.
- **Rule.** Closures (and partial applications such as `fget lc ib core shell`)
  must capture only the scalars or small sub-records they read. Bind the fields
  first, then build the lambda. Records passed into kernels must not contain
  arrays, even small fixed-size ones.
- **Diagnostic.** Run `futhark dev file.fut | awk` to rank functions by IR size
  before simplification. A huge `defunc_*`/`lifted_*` function means a fat
  closure capture.
- **Per-cell source branches.** These cost about 9 s in the shell; acceptable.

### Correctness

- `tiling/compare_temporal.py`: the PEC, PEC+CPML+DFT and CPML+DFT cases are
  bit-IDENTICAL at 90 and 45 steps.
- The realistic case differs by about 1 ulp from step 4 on. This is **not a
  tiling bug**: plain stepping is itself nondeterministic there. Two plain runs
  differ in 1347 values (overlapping mode-source entries are summed in atomic
  order), while two tiled runs are identical.
- Bug fixed today: `yee.fut`'s `inside` checks upper bounds only. Window halo
  cells at j or i < 0 read coefficients before the buffer start (an illegal
  address once the buffers were packed). `upd` now returns 0 for negative
  coordinates.
- Coefficient lookups must stay lazy (the `mat` closure inside `cell`). Hoisting
  `coefs` before the `inside` check reintroduced the illegal reads.

### Performance (128×256×512, 64 steps, Quadro RTX 5000)

| | plain | tiled |
|---|---|---|
| CPML all walls | 2.46 GCUPS | 0.83 GCUPS |
| CPML one wall (`--pml-edges right`) | 2.94 GCUPS | 0.88 GCUPS |

Per pass of 2 steps (`BEAMZ_FUTHARK_PROFILE=1`, run the benchmark with
`--case futhark:128x256x512 ...` so stderr isn't swallowed):

| kernel | one wall | all walls |
|---|---|---|
| shell | 23.8 ms (12,255 chunks) | 26.6 ms (4,842 chunks) |
| core | 8.95 ms (860 tiles) | 6.76 ms (776 tiles) |
| `copy_dev_to_dev` | 2.3 ms | 4.2 ms |
| DFT | 1.6 ms | 1.6 ms |

Plain stepping takes 11.4 to 13.7 ms per 2 steps.

### Next steps, in priority order

1. **Core kernel regressed.** It is 6.76 ms per pass now against 4.7 ms before
   the flat-buffer and closure rework, about 3 GCUPS, the same as plain. Even
   with a free shell, tiling can't win until the core is about 2× faster.
   - Check registers and spills per kernel. Offline compile is half done:
     `build/cuda/steps.cu` holds the prelude plus the step kernels. It currently
     fails with "unterminated #elif" because the prelude cut lands inside an
     `#if` block, so cut at a better point or include the whole prelude up to
     the first kernel.
   - Check whether the `common` variant is the one that runs. Uniform loads from
     `ib` (`icdim`/`ikdim`/`itdim`) per update may cost registers; consider
     passing the common-case scalars (H decay/source, E decay) directly.
2. **Shell chunking is pathological when z has no CPML.** `Zs = max(zb, P0 - zt)`
   is 3 there, so every non-core tile becomes 3-plane chunks, each with 3
   warm-up planes. That doubles the shell's work and launches 12k tiny blocks.
   - Fix: give non-core ("side") tiles whole-column storage of their own
     (`[nside][P0]`) and keep short caps for core tiles (`[2*nc][Zcap]`).
   - Cost: `locate` gets three cases, and the shell kernel runs over two item
     sets. Watch compile time if `shell_pass` gets instantiated twice.
3. **Shell per-plane cost** is about 3.5× the core's (general update, 168
   registers before today, with spills). Zap-style specialised shell kinds would
   help: boundary-only tiles, CPML slabs, source tiles.
4. **`copy_dev_to_dev`** runs 105 times per run (2.3 to 4.2 ms per pass). Find
   which pass-loop arrays get copied, probably the source-injection `add` that
   flattens and unflattens, or loop double-buffering.
5. Once tiled beats plain: remove the plain path and the `temporal` attribute,
   and round odd step counts up to whole passes in the runtime.
6. Memory: a 256×512×512 tiled run filled the GPU (14.9 of 15.4 GB); recheck
   after step 2.

### Idea for tomorrow: plain stepping for the edges, tiles for the core only

Suggested by the user. Plain stepping (2.5–2.9 GCUPS overall) beats today's
tiled shell by a wide margin, so the shell kernel could be dropped altogether:

- Core tiles run the lean two-step kernel as now.
- Everything else (CPML slabs, boundaries, sources) is stepped by the existing
  plain H/E kernels, restricted to boxes: z caps, y rows, x columns.

The catch is that a plain step needs neighbour values at the intermediate step
t+1, and core tiles never publish t+1. The fix is overlap, in the same spirit as
the tile halo:
- step 1 (t → t+1) runs the plain kernels on the edge region *plus a 2-cell
  margin* into the core;
- step 2 (t+1 → t+2) runs them on the edge region only.

The margin cells are interior and cheap.

Storage could go back to one padded layout, double-buffered (read t, write
t+2), since the core kernel reads halos at t while other tiles write t+2. CPML
memory would then stay in the plain layout, which removes the `mem` and
split-storage conversions and the 35 s shell compile. The plain code is
reused, not duplicated.

Rough budget per pass at 128×256×512 with one CPML wall:

| | estimate |
|---|---|
| Edge region, about 16% of cells plus margin, at plain speed | ≈ 2 ms |
| Core, 84% of cells, at 7 GCUPS | ≈ 4 ms |
| Core, 84% of cells, at today's 3 GCUPS | ≈ 9.4 ms |
| Plain stepping today | 11.4 ms |

That is about 1.9× faster than plain if the core reaches 7 GCUPS, and only
parity at today's speed, so priority 1 (the core regression) still decides the
outcome.

Open points:
- Intermediate-step monitors in core tiles (keep `recompute`).
- Sources stay in the edge region (`make_plan` already excludes busy tiles).
- How small the plain kernels' boxes can be without launch overhead
  dominating.

### Why the tiled CPML shell is so much slower than plain (analysis, unmeasured)

1. **Redundant work becomes real cost.** The 8×36 window computes stage-1 H
   and E and stage-2 H on 288 cells and stage-2 E on 128, about 3.9× the
   updates of 128 cells × 2 steps. For the lean core the GPU is
   memory-bound, so this is nearly free. The general CPML update is
   compute- and register-bound, so the multiplier lands directly on run time.
2. **Occupancy.** 168 registers × 288 threads means one block per SM, 9 warps,
   with spills. The plain kernels run at full occupancy.
3. **Chunk warm-up.** Each chunk recomputes 3 extra planes; with 3-plane chunks
   that doubles the work.
4. **Generality on every cell.** All 12 CPML memories are read and written for
   every shell cell, even where no slab is active (domain-edge tiles). That is
   48 extra bytes per cell each way, where plain only touches packed slab
   storage. Field reads go through `locate` and two arrays; components run in
   a dynamic loop; sources are checked per cell.
5. **Sync.** Four barriers per plane, in small blocks.

Plain stepping does one memory-bound update per cell, with CPML only on slab
cells. This is why plain-for-edges looks attractive.

### Why production core < prototypes (to bisect)

`lean.fut` already showed production generality halving speed
(PRODCELL/PRODMAT/PRODMETRIC switches). Candidates in today's core:
- `coefs`/`scale` reading offsets from `ib` per update (more registers);
- `fget`/`locate` on every E0 load and k+1 read (branch, two arrays, `ib`
  lookups);
- i64 index arithmetic;
- `core_step` compiling both the common and general variants.

Method: start from the fastest prototype and add production pieces one at a
time, measuring registers and ms per pass, until the drop appears.

### Region selection: bitmap vs static regions

A per-cell feature bitmap (which CPML terms, boundary, source) would select
behaviour per cell, but the selection branches per warp: divergence and the
union of all code paths' registers come back. Static regions are better, with
a uniform kernel per region, as ZapFDTD does. A middle ground is a per-*tile*
feature code computed once on the host or in setup, which picks one of a few
specialised kernels per tile kind. Each kind costs compile time, so keep the
kinds few (core, plain-stepped edges).

### Agreed game plan: plain edges with a wrong fringe, overwritten by core tiles

Per pass of K = 2 steps, with two full field buffers A (state t) and B (gets
t+2):

1. **Plain region.** The edge tiles plus a 2-cell margin into the neighbouring
   core tiles. Run the existing plain H/E updates for 2 steps on this region,
   restricted to boxes: z caps and y/x strips, with in-place slice updates.
   - Step 1 reads A and writes B.
   - Step 2 updates B in place.
   - Cells outside the region are never updated, so error spreads in from the
     region boundary by about one cell per step: the outer 2 cells (the
     fringe) are wrong. Because of the margin, the fringe lies entirely inside
     core tiles. Edge-tile cells are at least 2 cells from the region boundary,
     so they are exact.
2. **Core tiles.** The lean two-step kernel reads A and writes exactly its own
   tiles into B, after the plain step. This overwrites the wrong fringe.
3. Swap A and B.

Notes:
- **CPML memory** is only touched by plain steps (margin cells have no CPML),
  so it stays single-buffered in the plain layout. There is no `mem` split, no
  `untile`/`unpsi` conversion and no shell kernel at all.
- **Requirements carried over:** a core tile's window (tile ± 2) has no CPML
  slab, boundary or source. Busy tiles become edge tiles.
- **Monitors:** edge cells can sample t+1 from B between the two plain steps;
  core cells still need `recompute`.
- **Cost:** the fringe overlap is 2 cells around the core/edge interface,
  computed twice. Launches per pass: 2 steps × 2 phases × the number of
  boxes (≤ 6).

## Production integration, 2026-10-05

Code: `temporal.fut` (kernels, plan), `fdtd.fut` (driver), `yee.fut` (shared
update), `intrablock.py`, `tiling/core_bench.fut` (core kernel harness),
`tiling/compare_temporal.py` (bit-exactness oracle, now with a `gaussian`
case large enough for core tiles). Full CUDA build: 57 s.

### Why the production core was 2.5× slower than the prototypes

The 2026-10-04 core kernel ran at about 2.8 GCUPS. Rewritten to read one
tiled state directly, it reaches 7.0 GCUPS in the harness (8.5 without
material lookups).

- **Split storage.** Every field load went through `fget`/`locate`: an
  `instore` branch and two dependent `ib` loads (core index, shell base)
  before the field load itself. That applied to the E0 plane, old H and the
  z+1 reads. Registers were not the problem (112, no spills).
- **Coefficients through `ib`.** Offsets for the scalar coefficients and the
  codebook were read per update. They now arrive as scalars, plus three
  separate code and table arrays. `e_table_x ++ e_table_y ++ e_table_z` had
  fused into three-way branchy loads.
- **Material codes themselves** cost 18% (7.0 vs 8.5 GCUPS); the prototypes
  use constant coefficients.

### Design now

- **State:** E and H as `[tile][z][lane][3]` arrays, tiles of 8×16 (the
  x-rounding of 4×32 tiles put 96 of 513 columns in edge tiles; 8×16 puts
  48 there).
- **Pass (two steps):** the core kernel writes core cells of a fresh C.
  Plain step 1 runs over a compacted cell list (edge cells, core cells within
  **one** cell of them, and monitor cells with their lower neighbours) from
  A into a scratch T. Plain step 2 runs over the edge cells from T into C.
  Its E reads H(t+2) of core cells from C, which the core kernel has
  already written. No wrong fringe, no merge, no shell kernel. Odd step counts
  end with one plain step over all cells, so they are tiled too.
- **Margin of one cell:** step 2's H reads step 1's E at offsets {0, +e_a};
  that E reads H at {0, −e_b}. So step 1 needs H on R + {0, +e_a, +e_a−e_b,
  −e_b} and E on R + {0, +e_a}: Chebyshev distance 1, not a tile and not two
  planes.
- **Plain steps reuse yee.fut:** `forward`/`backward`/`field_value` take field
  accessors, and `field_cpml` also returns the advanced CPML memories. Edge
  phases write fields and memories in one kernel (double-buffered ψ,
  scattered), so the 24 `psi_next` kernels per pass are gone in the tiled path.
- **Sources** are injected per (E or H, timing) with a K-sized histogram per
  distinct target row and a row scatter (`source_rows`, `tinject`).

### Correctness

- `tiling/compare_temporal.py pec mixed pml gaussian`: all IDENTICAL at
  90 and 45 steps (odd counts take the final all-cells plain step), and the
  Gaussian 64×96×224 case (which has core tiles) at 64 and 33 steps.
- `tests/hardware/test_futhark_backend.py`: 4 passed, with and without
  `BEAMZ_FUTHARK_TEMPORAL=2`. yee.fut's plain path changed too (accessors,
  `field_value` via `field_cpml`).
- The mode-source `realistic` case still differs at the ulp level. That is
  plain stepping's nondeterminism (atomic order of overlapping entries): two
  plain runs differ in 1,059,050 values after 64 steps, two tiled runs in 0.

### Alternatives measured and dropped (128×256×512, all walls, ms per pass)

| Variant | core | rest | total |
|---|---:|---:|---:|
| 4×32 tiles, per-component arrays, tile-margin items | 3.3 | 10.6 | 13.9 |
| + ψ merged into edge kernels | 3.3 | 10.3 | 13.6 |
| + per-lane mask for the margin (warps rarely all masked) | 3.3 | 10.3 | 13.6 |
| + compacted cell lists | 3.3 | 10.2 | 13.6 |
| + 8×16 tiles | 3.1 | 9.9 | 13.1 |
| core kernel writes the intermediate state T everywhere, no margin | 4.1 | 8.8 | 12.9 |
| T only for tiles next to edges/monitors (branch, or redirected to plane 0) | 4.1 | 8.7 | 12.8 |
| `[3]` rows, no T, cell-list margin, one injection per E/H triple (kept) | 3.0 | 9.8 | 12.8 |

The kept variant ties the best for all walls and wins when the edge is small
(one wall: 4.17 vs 4.06 GCUPS at 128×256×512 for the T-writing variant).

### No CPML (`--pml 0`, 128×256×512), per pass (8.3 ms)

| Kernel | ms |
|---|---:|
| core (about 14 M cells, 6.9 GCUPS) | 4.07 |
| step 1, H + E | 2.11 |
| step 2, H + E | 1.32 |
| DFT, injection, other | 0.8 |

### Diagnostics without nsys/ncu

- `BEAMZ_FUTHARK_PROFILE=1` with `--case futhark:<shape>`: take the *last*
  JSON report; the first call is cold.
- `BEAMZ_FUTHARK_DUMP_PTX=<file>` (new in `build.py`), then
  `ptxas -arch=sm_75 -v` for registers and spills, and `cuobjdump -sass` for
  instruction counts.
- `futhark dev --gpu-mem fdtd.fut`: look for `manifest`/`copy` in the pass
  loop and `ctx_param_ext` (existential strides) on loop parameters.
- Futhark's own executable timing (`tiling/core_bench.fut`) is useful only
  with a fixed input: a loop-carried state adds copies to the measurement.

### Futhark pitfalls met on the way

Compiler bugs (with minimal reproducers), generated-code patches and these
codegen traps are collected in `FUTHARK_ISSUES.md`.

- `flatten` of a loop-carried array forces a full copy whenever Futhark does
  not know its outer strides (existential loop layouts). This produced
  2–4 `copy_dev_to_dev` of 70–850 MB per pass. Avoid flattening tiled state;
  use `scatter_3d` on `[3]` rows.
- `reduce_by_index_3d` with `[3]f32` elements is an internal compiler error
  in 0.27.1, and three independent multi-dimensional histograms fused
  horizontally hit another ("Bucket function has return type …").
- `map (map flatten)` on a unique array copies it.
- In an intra-block kernel, a row built by slicing (`v[sq, y, xx]`) becomes a
  parallel dimension (block of 384 instead of 240 threads); build rows as
  literals. An `if` choosing between two loop variants duplicates the block's
  shared allocations and defeats `intrablock.py`.
- A `tabulate_3d` over tiles fused into a per-tile plan kernel stores its
  result interleaved; `to_tiled` is therefore a flat `tabulate`.
- `intrablock.py`: the host sums shared sizes in its own order and folds
  constants, so moved results are now matched by size, not position.
- Profiles of a run's first call are cold (first touch): the first three
  passes take 20 ms instead of 3 ms. The benchmark's second report is the
  warm one.

### Results (Quadro RTX 5000, 256 steps, GCUPS)

| Shape | All walls, plain | All walls, tiled | One wall (`right`), plain | One wall, tiled |
|---|---:|---:|---:|---:|
| 64×96×128 | 1.73 | 1.47 | 2.48 | 1.96 |
| 96×160×256 | 2.31 | 2.03 | 2.99 | 3.27 |
| 128×256×512 | 2.51 | 2.60 | 3.00 | **4.17** |
| 256×512×512 | 2.68 | 1.13 | 2.98 | 1.32 |

No CPML (`--pml 0`), 128×256×512: plain 2.95, tiled **4.02**. The tiled path
before this work: 0.83 (all walls), 0.88 (one wall).

Per pass at 128×256×512, all walls (warm, 12.8 ms; plain 13.6 ms):

| Kernel | ms |
|---|---:|
| core (≈ 10.9 M cells, 6.9 GCUPS) | 3.0 |
| step 1, H + E (edge plus margin plus monitor cells) | 5.0 |
| step 2, H + E (edge cells) | 4.1 |
| DFT, injection, other | 0.7 |

### Why not 10 GCUPS

The bare prototype's 9.6 GCUPS is a core-only number on homogeneous material.
In production:

1. The core kernel runs at 6.9 GCUPS. Material lookups cost 18%, and the
   kernel now looks bandwidth-limited at roughly 250 GB/s effective (an
   estimate from byte counts). An extra 24 B/cell output costs about 35%,
   even when the writes all land on the same cache-resident lines. So
   making the core kernel also write the intermediate state (instead of step
   1's margin) is a wash: tried, about 1 ms either way.
2. Everything outside the core runs at plain speed. Even without CPML the
   core covers about 83% of cells: windows must avoid PEC planes and sources,
   and tiles round up. That remainder takes half of each pass.
3. With CPML, edge cells cost about 0.6 ns per update (as in plain stepping;
   the edge kernels are within about 20% of plain on the same cells). On all
   six walls that cancels the core's gain.

Step 1's margin and monitor cells cost about 1 ms per pass for about 7% more
cells, because x-margin columns and monitor planes (x = const) are scattered
in tiled order.

### Time budget from invocation to kernel (no CPML, 128×256×512, 256 steps)

10 GCUPS would be 16.8 M cells × 256 steps in 430 ms, or 3.4 ms per
two-step pass. Measured: 8.3 ms per pass (4.02 GCUPS).

- **Invocation:** one XLA FFI call runs all 256 steps; inputs alias
  without copies. The one-off cost per call is about 25 ms: padding (2.6 ms),
  `to_tiled`, plan and cell lists (a filter over 18 M cells), grouping
  source entries, untiling and cropping (about 3 ms), and the handler's
  output copy. That's about 0.2 ms per pass at 256 steps.
- **Storage:** GCUPS counts 16.8 M cells. The padded store is 129×257×513 =
  17.0 M; whole 8×16 tiles make it 129×264×528 = 18.0 M.
- **Split:** about 83% of cells can be core cells even without walls,
  because windows must avoid PEC planes and sources: z 123/129 planes,
  y and x 30/33 tiles.
- **Per pass:** core 4.07 ms (14 M cells, 6.9 GCUPS), step 1 2.11, step 2
  1.32, DFT/injection/other 0.8.
- **Ceilings of this design:** a 9.6 GCUPS core with today's edges gives
  about 4.7 GCUPS; everything at today's core speed gives about 6.9.
  10 GCUPS needs every cell, boundaries and sources included, in a kernel
  above 10 GCUPS on production material, and no fixed cost.

### Where the edge cells come from (128×256×512, PML 12 on all walls, 8×16 tiles)

Geometric counts from the plan rules, consistent with the measured totals:

| Region | Cells | Share |
|---|---:|---:|
| Core tiles | ≈10.86 M | 64% |
| CPML (12-cell slabs on all six faces) | ≈5.0 M | 30% |
| Non-CPML edge (halo, rounding, padding, source tiles) | ≈1.1 M | 6.5% |

- The edge is about 82% genuine CPML.
- The non-CPML part:
  - y rows 12–15 and 240–243 (halo plus rounding to 8, ≈0.39 M);
  - z planes 12, 13 and 115 (window lead-in, ≈0.32 M);
  - x columns 12–15 and 496–499 (≈0.18 M);
  - store padding (≈0.2 M);
  - source tiles (≈0.08 M).
- Of the CPML cells, 90% sit in one slab (4 of 12 terms stretched). Most
  are z slabs (2.7 M), because the domain is only 128 deep. Edges and
  corners (8 or 12 terms) are 0.5 M.
- Tile shape is therefore not the lever (at most about 0.5–0.7 ms per
  pass); the CPML share is.

### Next steps

1. **Memory.** Futhark's CUDA backend defaults to managed memory
   (`unified_memory = 2`, auto), so running out of memory pages instead of
   failing. At 256×512×512 Futhark held 10.2 GB, the GPU 14.9 of 15.4 GB, and
   kernels slowed by about 10×, with 3.6 s of one-off cost per call. T only
   needs the step-1 cells: store it compactly (indexed through the cell
   list) instead of as two more full states. Also consider setting
   `unified_memory = 0` in the handler so that this fails loudly.
2. **Core bandwidth.** The kernel moves about 69 B per cell per pass against
   48 B minimum. Candidates: the halo re-reads and two reads of the material
   codes (once per stage).
3. **CPML** is now the bulk of the all-walls cost: checkerboard CPML tiles
   (previous section) are the next big lever.
4. **Small shapes** lose (64×96×128): one-off setup per call (plan, cell
   lists, `to_tiled`/`untile`, about 20 ms at 128×256×512) and a large edge
   fraction. Fall back to plain stepping below a core-fraction threshold.
5. The CPU builds compile the same `temporal.fut` but were not run here.
6. The library in `beamz/simulation/futhark/_native` was not rebuilt; run
   `futhark/build.py`. Nothing is committed yet.

### Where to resume

Superseded: see [Re-ranked priorities](#re-ranked-priorities) at the end.

Build and measure with `futhark/build.py --work DIR --output DIR`.
Then `BEAMZ_FUTHARK_LIBRARY=DIR/libbeamz_futhark_cuda.so
BEAMZ_FUTHARK_TEMPORAL=2 scripts/benchmark_futhark_jax.py`, adding
`BEAMZ_FUTHARK_PROFILE=1 --case futhark:SHAPE` for kernel times (use the
last report). Check exactness with `tiling/compare_temporal.py pec mixed pml
gaussian`.

## Checkerboard tiling and state compression (2026-10-05)

Two ideas, each tested for exactness or accuracy first and then for speed.
Files in `tiling/`:

| File | Contents |
|---|---|
| `checker.py` | numpy model of checkerboard (split) tiling that tracks a time level per cell and component, so any stale or overwritten read is detected; bitwise check against plain stepping, work count. |
| `dct.fut` | The 2018 Loeffler 8-point DCT (blog post) ported to Futhark 0.27, plus separable 8×8×8 forward and inverse; checked against `scipy.fft.dct(norm='ortho')`. The 3D VR codec from 2018 was never published (no repo or gist), so 3D is new. |
| `compress_study.py` | Accuracy of compressing the whole state every K = 2 steps (fp16, bf16, ZFP, DCT block floating point, abstol). Needs `zfpy` (a scratch venv). |
| `bare3h.fut`, `bare4h.fut` | Copies of the `bare4` ring kernels with f16 field storage (generated by sed; arithmetic and rings stay f32). |

`bare.fut` gained `HY`/`HX` (window halo, default `K`), so `gen.py` can time
a kernel with a halo other than K. `intrablock.py` now also matches f16
copy-out loops (`fptobits_f16_i16(bitstofp_i16_f16(...))`).

### Headline: the K=2 GPU kernel is latency-bound, not bandwidth-bound

Two independent measurements on the Quadro RTX 5000, `bare4` K=2 at
128×256×512:

- **f16 storage** (half the bytes, results checked against f32 and the
  unpatched kernel) is *slower*: 7.48 GCUPS against 9.57 for f32. At K=1
  (chunked, ZC=16) f16 gains only 14% (4.03 → 4.58).
- **Removing the halo** entirely (window 4×32 = 128 cells instead of
  8×36 = 288: 2.25× less work and fewer loads; wrong results, timing only)
  gains only 13% (1.755 → 1.549 ms/step).

So time per z plane barely depends on bytes or work. *Estimate:* 1024 tiles
on 48 SMs at 2 blocks per SM is about 11 waves, which gives about 2.5 µs per
plane per block, roughly 400 cycles per barrier-separated stage. That fits
dependent global loads (E at z+1, old H) serialised by barriers, with only
about 18 warps per SM to hide them. Levers, untried: prefetch the next
plane's global values a stage ahead (hard to express in Futhark: values
crossing iterations land in shared memory, and a barrier waits for the load),
more resident blocks (`BEAMZ_TILE_MIN_BLOCKS`, smaller shared footprint),
fewer barriers per plane, or two planes per iteration for ILP.

### Checkerboard (split) tiling: exact, nearly no redundancy

Per pass, black tiles (checkerboard in xy, z streamed) advance in place on
their own tile only: a shrinking pyramid, no halo. White tiles then read
black cells from black's output (with their partial levels) and white cells
(own and diagonal) from the old state, and compute their own tile plus the
black border cells they own: each unfinished black cell belongs to the white
neighbour across its nearest tile edge.

`checker.py` results (several shapes, 2 passes, all **bitwise identical**
to plain stepping):

| Tile | K | Updates vs plain | White reach beyond tile (y, x) |
|---|---:|---:|---|
| 4×32 | 2 | 1.029 | 2, 3 |
| 8×16 | 2 | 1.025 | 2, 3 |
| 8×16 | 3 | 1.057 | 3, 4 |
| 8×16 | 4 | 1.102 | 4, 5 |
| 16×32 | 4 | 1.027 | 4, 5 |

The overlapped K=2 4×32 tile computes about 1.94× plain. Notes:

- No in-place hazard: whenever a white tile needs a value, the black tile
  holds exactly that level, never a newer one, for K = 2, 3, 4. This is
  because the pyramid shrinks one cell per half-step on the side the next
  half-step reads (H reads +1, E reads −1).
- There *is* a diagonal dependency, not within a half-step but across one
  full step: H(+1) then E(−1) gives offsets (−1, +1) and (+1, −1). White
  tiles therefore compute a few cells of their anti-diagonal white
  neighbours from the old state: 1.2% of plain work at K=2, 4×32.
- Races: white tiles must read white cells from A and write their results to
  B, and the black border cells they finish are read by other white tiles at
  intermediate levels. So the borders need a third location (or A, which no
  white tile reads for black cells).
- With 4-row tiles at K=2, black finishes almost nothing (E2 is missing two
  rows at each y edge), so white does nearly all final work.

**Speed proxy.** A black pass costs about a K=2 kernel with halo 0, a white
pass about one with halo (2, 3) (it reads B levels too, ignored here). Per
tile, ms/step:

| Target, tile | black (halo 0) | white (2,3) or (K,K+1) | mean | overlapped (K,K) | gain |
|---|---:|---:|---:|---:|---:|
| CUDA K=2, 4×32 | 1.549 | 1.839 | 1.694 | 1.755 | 3.5% |
| CUDA K=2, 8×32 | 1.672 | 1.842 | 1.757 | 2.034 | (best overlapped 1.755) |
| multicore K=3, 16×128 | 10.98 | 13.07 | 12.02 | 13.25 | 9% |
| multicore K=4, 16×128 | 8.46 | 11.94 | 10.20 | 10.37 | 1.6% |

**Verdict:** correct and nearly free of redundant work, but at most about
+3.5% on CUDA and +9% on CPU for the bare stencil, because run time isn't
proportional to window work (see the headline). Not worth a two-kernel
implementation for the lean core. It would matter where per-cell work is
heavy and redundancy is the problem, as in the tiled CPML shell (3.9×
updates), but the agreed plan steps the edges plainly instead.

### Time-skewed (staggered) checkerboard (2026-10-05, model only)

A variant of the above: the colours are offset in time. White advances
`lead` steps, then the colours alternate, each advancing K steps from where
it is (K=2: white to 1, black to 2, white to 3, …). Every tile, of either
colour, runs the same kernel, and there is one state in place. Modelled in
`checker.py --stagger STEPS [--lead L] [--k K] [--depth D] [--halo R
--halo-comps H|E|EH]`. The model is level-tracked: a read of a too-new or
stale value never happens silently; the cell just doesn't advance. A
backward pass counts only the halo values that were actually needed.

- **Own cells only, no halo: it degenerates to one step per sweep.** It
  stays exact and nothing is computed twice, but every tile interface
  alternates H on one side and E on the other (H reads E at +1, E reads the
  new H at −1). So each turn moves it one level, and interiors can't run
  ahead of their edges. With an in-place state, border history makes no
  difference to this.
- **With a private halo (computed for the turn, then discarded) and some
  history of border levels, every turn reaches its target.** It is bit-exact
  and needs no draining. At the best offset (lead ≈ K/2) it needs:

| K | lead | Border history (levels) | Halo used | Redundant (8×16 tiles) | Synchronous checkerboard |
|---|---:|---:|---|---:|---:|
| 2 | 1 | 1 | H only, 1 cell | 2.9% | 2.8% |
| 3 | 1 or 2 | 2 | E and H, 2 cells | 7.8% | 6.5% |
| 4 | 2 | 2 | E and H, 2 cells | 10.3% | 11.7% |

  Without history, or with less than listed, cells get stuck: a neighbour
  has overwritten the level they need. Tile sizes at K=2: 4×32 4.4% (synchronous
  2.8%), 16×32 1.5% (0.7%). Grid 67×135 (z untiled), 12 steps.

**Compared with the synchronous version:** about the same redundant work.
But both colours use one kernel, symmetric and reaching K/2 cells
instead of 2–3 on one side. There is no black pyramid, and white doesn't
have to load black's partial levels. Instead of a third location for the
owned border cells, each colour keeps a short history (K/2 levels) of its
own border strip for its neighbours.

**Not yet known:** cost on the GPU. The model counts updates only, for
the bare stencil, with no CPML, sources or boundaries. Next: a two-kernel
proxy (one kernel per colour, halo R, history strip) in `lean.fut` style,
timed against the overlapped and synchronous proxies, first on z-slab CPML
tiles, where removing redundancy pays.

### Lossy state compression: ZFP wins; a naive DCT is unstable

`compress_study.py`: 64³ PEC box (a resonator: errors never leave), dielectric
slab (ε=4) and sphere (ε=6), soft Ez pulse at about 20 cells per vacuum
wavelength, 1200 steps (about 30 periods). All six components round-trip
through the codec every 2 steps. Relative L2 error of all fields at the end
against the float32 run; energy = sum of squared fields relative to float32.

| Scheme | bits/value | error at t=1200 | energy |
|---|---:|---:|---:|
| fp16 | 16 | 5.7e-3 | 1.000 |
| bf16 | 16 | 4.6e-2 | 1.008 |
| **ZFP fixed rate 16** | 16 | **7.5e-5** | 1.000 |
| **ZFP fixed rate 12** | 12 | **1.2e-3** | 1.000 |
| ZFP 8 | 8 | 6.2e-2 | 1.004 |
| ZFP 6 / 4 | 6 / 4 | diverges | ≫ 1 |
| DCT 8³, block float, zonal bits, round | 16 / 12 / 8 | 8.4e-3 / 1.3e-1 / diverges | |
| same, round toward zero | 16 / 12 / 8 / 4 | 4.4e-3 / 3.2e-2 / 1.3e-1 / 2.9e-1 | 0.99 / 0.96 / 0.95 / 0.73 |
| DCT 8×8 per z plane, toward zero | 16 / 12 / 8 | 3.3e-3 / 3.3e-2 / 1.7e-1 | |
| block float, no transform, round | 16 / 12 / 8 | 1.1e-3 / 1.8e-2 / 3.2e-1 | |
| abstol 1e-5 / 1e-6 (variable rate, entropy) | 6.1 / 8.0 | 3.1e-3 / 3.2e-4 | 1.000 |

Field amplitudes are about 0.1–0.3, so abstol 1e-6 is about 5e-6 relative.
Full table: run the script; it also tunes the zonal slope per rate.

Findings:

- **Requantising inside the time loop can go unstable.** Rounding to nearest
  feeds energy back, as in fixed-point IIR filters (limit cycles), and
  diverges at low rates, even for ZFP at 6 bits. The truncation alone (exact
  low-pass, f32 coefficients) is stable and slightly dissipative, and almost
  free in vacuum (1.7e-5 at u+v+w ≤ 18); its error comes from the dielectric
  interfaces. **Rounding toward zero** (magnitude truncation) cannot add
  energy in an orthonormal basis (‖Q(c)‖ ≤ ‖c‖) and is always stable, at the
  cost of damping.
- **ZFP is the right codec.** It is fixed rate, so it allows random access per
  4³ block. Its embedded bit-plane coding beats a hand-tuned zonal DCT by
  more than 10× at the same rate, and at 16 bits it is 75× more accurate
  than fp16. ZFP 12 (2.7× compression) beats fp16 (2×).
- **An abstol** (one global quantisation step) is the most efficient by far
  (about 8 bits/value for 3e-4) but variable rate. Random access then needs
  per-tile offsets and an entropy decoder in the kernel. A fixed-rate
  compromise: choose the ZFP rate per run from the abstol, or use ZFP's
  fixed-accuracy mode with a per-tile size table.
- All lossy schemes end bitwise equality with plain stepping. The oracle
  becomes an error budget against float32 (and against the discretisation
  error, which at 20 cells per wavelength is far larger than 1e-3).

**Speed:** given the headline (f16 storage is slower at K=2 and only 14%
faster at K=1), no codec pays on this GPU today: decoding adds exactly the
instructions and latency the kernel is limited by. Revisit compression only
when a kernel is shown to be bandwidth-bound: the CPU at high thread counts,
plain stepping of the CPML edges, or after the latency levers above.

### Checkerboard for CPML tiles (2026-10-05, later)

Question: CPML cells do much more work per update than core cells, so does
removing the halo's redundant work pay off there, where it barely did for the
lean core?

**Proxy.** `tiling/lean.fut` with `HY`/`HX` halo constants (default 2),
4×32 tiles, `ZL=64` so every plane of a 129-plane column is a z-CPML slab
plane, `ZRING=true`. A black tile is timed as halo (0, 0), a white tile as
(2, 3), overlapped tiling as (2, 2). `bench_plain` is a plain one-step
baseline with the same `upd` (one H and one E kernel, dense z memory). New
entry `check_plain` compares one tiled pass with two plain steps bitwise.

Correctness first:
- `lean.fut` with `ZRING=false` (the default) is **not exact**: stage 2
  recomputes the stage-1 z memory with a bare `cpml_term`, skipping
  `dsample`. It was only ever a cost experiment. With `ZRING=true` the tiled
  pass is bitwise identical to two plain steps, for bare, z-CPML,
  z-CPML + `PRODCELL`, and + `PRODMAT PRODMETRIC`, on 29×37×70. `SPLITG` can't
  be checked: it reads a second array in reversed z order.
- `esrc` read material codes at halo cells outside the box (illegal address
  with `PRODMAT`, the same bug class as `yee.fut`'s); now guarded with
  `instore`.
- `PRODCELL` was updated to the current `yee.cell` signature.

**Results**, ms/step at 129×257×513, Quadro RTX 5000, reproducible on rerun
(an earlier run with `ZRING=false` and odd bare timings was discarded):

| Kernel | black (0,0) | white (2,3) | checkerboard mean | overlapped (2,2) | checker gain | `bench_plain` |
|---|---:|---:|---:|---:|---:|---:|
| bare | 1.58 | 2.51 | 2.04 | 2.43 | 1.19× | 8.00 |
| z-CPML, lean `upd` | 2.51 | 5.16 | 3.83 | 5.09 | 1.33× | 12.29 |
| z-CPML, production `cell` | 3.55 | 9.71 | 6.63 | 9.44 | 1.42× | 12.18 |
| + production material and metric | 4.03 | 11.11 | 7.57 | 10.90 | 1.44× | 12.54 |

The heavier the per-cell update, the more the halo costs: with the
production cell, halo 0 is 2.7× faster than halo 2 (bare: 1.5×). So the
CPML tile is work- and occupancy-bound where the lean core is not.

**Against production plain stepping.** `bench_plain` is weak (2.13 GCUPS bare
against `bare.fut`'s 3.62), so the real reference is production.
`scripts/benchmark_futhark_jax.py --backends futhark --shapes 128x256x512
--steps 128` at `--pml 12` (5.00 M CPML cells) gives 6.73 ms/step, and at
`--pml 40` (13.13 M) 9.24 ms/step. Solving the two equations: production
plain costs about **0.31 ns per interior cell** (3.2 GCUPS) and **0.62 ns per
CPML cell** (1.6 GCUPS). At PML 12, CPML is 30% of the cells and 46% of the
step.

Per cell, CPML proxy with production generality:

| Schedule | ns per cell-step |
|---|---:|
| Production plain, CPML cell | 0.62 |
| Overlapped K=2 tile (2,2) | 0.645 |
| Checkerboard K=2 tiles | **0.448** |

Overlapped CPML tiles are no faster than plain, which matches the
production shell experience (it was slower still, for the other reasons
listed above). Checkerboard CPML tiles are about **1.38× faster than plain**.

*Estimate* for the whole 128×256×512 PML-12 case per step, assuming the core
reaches the lean bare speed (2.43 ms per 16.9 M cells for overlapped, 2.04
for checkerboard):

| Plan | core (11.77 M) | CPML (5.00 M) | total | vs plain today (6.73) |
|---|---:|---:|---:|---:|
| Agreed plan: tiled core + plain edges | 1.69 | 3.09 | 4.78 | 1.41× |
| Checkerboard everywhere | 1.42 | 2.24 | 3.66 | 1.84× |

Caveats:
- Proxy CPML is z slabs only: each x/y component has one stretched term.
  Edges and corners of the real PML have two or three.
- The white proxy computes its whole window and ignores loading black
  levels from B and writing the owned border; the black proxy ignores the
  pyramid shape (the kernel computes the whole tile anyway).
- No sources, monitors or boundaries; core speed assumes the core regression
  (next steps, item 1) is fixed.
- One K: deeper K on CPML tiles is untested.

**Next**, if this is pursued: write the real black and white kernels for
z-slab CPML tiles in `lean.fut` style (static level geometry from
`checker.py`, owned border cells written to a third buffer), check bitwise
against `bench_plain`'s `plain_step`, and compare with the proxy.

## Related work: LRnLA diamond tiling, and re-ranking (2026-10-05)

**Framing.** The target is large domains. The small shapes are what fits this
16 GB GPU, and the one-wall CPML case is the deliberate proxy for a large
domain. All-walls at 128×256×512 (30% of cells, 46% of step time in CPML) is a
worst case. At 1024³ with 12-cell CPML on all six walls, CPML is about 7% of
the cells and, at the measured 0.31 vs 0.62 ns per update, about 13% of the
step.

### The papers

Both are from the Keldysh Institute (Levchenko, Perepelkina et al.).

**DiamondTorre FDTD.** Zakirov, Levchenko, Perepelkina, Zempo, *High
performance FDTD code implementation for GPGPU supercomputers*, Keldysh
preprint 44/2016, doi:10.20948/prepr-2016-44-e.

- 4th-order Yee, double precision, Tesla K20x, up to 256 nodes of TSUBAME 2.5.
- The tile is the smallest diamond in xy (DTS=1), stacked TH=100 steps and
  sliding along x. Diamonds in a row along y run asynchronously.
- z runs across threads, one thread per z point, and the column lives in
  registers. Nz must be 384 (double) or 768 (single); at Nz=128 they reach
  only about 30% of peak because too few requests are in flight to hide
  memory latency.
- DTS=1 is forced by the register file (about 85 registers per thread) and
  the instruction cache (about 1500 instructions against about 16 KB).
- Result: about 1.05 G updates/s per device, 90% of their bandwidth ceiling.
- **Bandwidth-bound by design.** Even with TH→∞ each update loads about 3
  cells and stores 1 (192 B in double), an operational intensity of 0.57
  against 0.23 for naive stepping: only about 2.5× better.
- *Estimate* for this GPU: about 96 B per update in float at 448 GB/s gives a
  ceiling of about 4.7 GCUPS. Our core kernel (6.9 GCUPS, about 69 B per
  cell) already beats it; our tiled path (4.17, one wall) is about level.
- PML, TFSF and Drude materials are listed as features. Their handling and
  cost are not described, which is reasonable at 10⁸–10¹¹ cells.
- Its distinct contribution: a moving *calculation window* streams data
  between host (or SSD) and device, so the domain is not limited by device
  memory. Deep TH hides PCIe.

**DiamondCandy.** Perepelkina, Levchenko, *Recursive DiamondCandy:
non-memory-bound LRnLA algorithm for 3D cross stencil calculations on CUDA
GPU*, HP3C 2020, doi:10.1145/3407947.3407951.

- The tile follows the dependency cone: an octahedron plus two tetrahedra,
  which tiles 3D space with no gaps or overlaps, tiled along (1,0,1),
  (0,1,1), (−1,−1,0). It is extruded into a prism leaning in z–t, thousands
  of steps tall (NT), and the prisms are coordinated through global memory
  and semaphores. No redundant work.
- **The whole working set is in registers.** Each thread holds 2×2×4 pairs
  (32 values), each block 16³ pairs; warp shuffles within a warp, shared
  memory only between warps, about 2 `__syncthreads` per step.
- Benchmark: homogeneous scalar wave equation, two constant coefficients,
  no boundaries, sources or materials. 261 GCUPS on V100, 100 on an RTX 2060
  (Turing, like ours).
- Their own caveats: integer offset arithmetic and the compiler's register
  minimisation are the main obstacles. Their earlier 2D DiamondTorre was
  faster on a GTX 970 (50 vs 38 GCUPS); they argue DiamondCandy wins only on
  GPUs with less bandwidth per flop.

### What transfers

| Idea | Relevance for us |
|---|---|
| Register-resident tile, 32 values per thread, shuffles instead of most barriers (DiamondCandy) | **High**: it is the cure for our latency-bound core (see "Headline" above). Futhark can't express it fully: no warp shuffles, and values carried across loop iterations go to shared memory. More cells per thread and fewer barriers per plane can be tried within Futhark. |
| Deep asynchronous time blocking (NT, TH ≫ 2) | Needs every region, CPML, sources and monitors included, to live inside the tiles; the core-plus-plain-edges design syncs every K steps. Not expressible in Futhark (no cross-block semaphores). |
| Calculation window, out of core (DiamondTorre FDTD) | Only if domains must exceed device memory. Compact T and no managed-memory paging come first. |
| z across threads, sliding diamond (DiamondTorre FDTD) | Low: bandwidth-bound at about 96 B per update, which the core kernel already beats. |
| Diamond tile shapes | Low: see below. |

### Diamond tile shapes don't remove our redundancy

`tiling/shapes.py` reruns `checker.stagger` with the rectangular colour mask
replaced by other 2-colourable tilings of the same area (128 cells): diamonds
(rotated squares) and parallelograms sheared both ways. Grid 67×135, 12
steps, each at the least history and halo that stays exact; all bit-exact.

| Tile shape (128 cells) | Redundant, K=2 (lead 1, history 1, halo 1 H) | Redundant, K=4 (lead 2, history 2, halo 2 EH) |
|---|---:|---:|
| Rectangle 8×16 | **2.9%** | 10.3% |
| Diamond (rotated square) | 3.4% | **9.5%** |
| Sheared `/` 8×16 | 3.4% | 11.0% |
| Sheared `\` 8×16 | 3.8% | 11.7% |
| Sheared `/` 16×8 | 3.9% | 11.8% |
| Sheared `\` 16×8 | 4.8% | 13.7% |

Our tiles are prisms: a tile owns a fixed set of xy cells for a whole turn.
With two colours, every face carries dependencies both ways within a turn,
so a strip of about perimeter × K/2 must be recomputed (halo) or read old
(history), whatever the shape. Diamond tilings differ in two ways:

1. A cell's owner depends on (x, y, t): faces slope at one cell per step, so
   every dependency crosses a face in one direction only. The synchronous
   checkerboard does half of this (black pyramid, white inverted pyramid) but
   still has redundancy where four tiles meet, since diagonal tiles share a
   colour. Yee's full step reaches the anti-diagonal ((+1,−1) and (−1,+1)),
   which is why DiamondCandy's tiling directions are skewed.
2. The schedule is a wavefront, not two colours: DiamondTorre sweeps along x
   and is parallel only along y; DiamondCandy orders tiles with semaphores.
   In 1D two colours of diamonds suffice; in 2D, removing the redundancy
   needs more phases or an ordered sweep.

Adopting that means owner-dependent index arithmetic per step, a
non-rectangular data layout, more launches per K steps (or cross-block
sync), and fewer tiles in flight, all against a latency-bound kernel, to
save about 3% of updates at K=2. If deeper K pays, larger tiles cut the
redundancy more cheaply (synchronous checkerboard: 2.7% at 16×32, K=4).
Keep rectangles.

### Re-ranked priorities

This replaces "Where to resume" in the production section. Superseded in
turn by [Where to resume (2026-10-06)](#where-to-resume-2026-10-06): item 2
turned out to be bandwidth, not latency, and item 5 does not pay.

1. **Memory per cell** (done, see the next section). Store T compactly (it
   only needs the step-1 cells, not two full states) and set
   `unified_memory = 0` so that running out fails loudly instead of paging. Bytes per cell set the largest domain on
   any GPU. This also favours the staggered checkerboard (one in-place state
   plus a short border history) over the synchronous one (B plus a border
   buffer).
2. **Core latency.** More cells per thread and fewer barriers per plane, as
   far as Futhark allows (the DiamondCandy lessons). Measure on `bare` and
   `lean` first.
3. **Edge cells that don't shrink with domain size.** Sources and monitors
   are stepped plainly today; handle them inside the tiled path.
4. **Staggered checkerboard proxy.** Time `lean.fut` with `HY=HX=1` (an
   upper bound for a staggered tile, since it computes both E and H in the
   halo). Its case is now in-place memory and room for deeper K, not CPML
   speed. Whether K=4 pays on the GPU is still unmeasured.
5. **Checkerboard CPML tiles.** 1.38× on CPML cells in the proxy, which is a
   few percent of total time at large-domain CPML fractions.

## Memory per cell (2026-10-05, evening)

Priority 1 above, done. Futhark's peak device memory (profile report), and
GCUPS with `--pml-edges right`, 256 steps:

| | 128×256×512 | 256×512×512 | B/cell (256×512×512) | GCUPS 256×512×512, one wall / all |
|---|---:|---:|---:|---:|
| plain | 1194 MiB | 4753 MiB | 71 | 2.98 / 2.67 |
| tiled, before | 2479 MiB | ≈10.2 GB (paged) | 155 | 1.31 / 1.13 |
| tiled, now | 1149 MiB | 3948 MiB | 59 | **4.14 / 3.05** |

128×256×512 is unchanged in speed (one wall 4.20, all walls 2.53–2.63, plain
3.01 / 2.51). Small shapes too (64×96×128: 1.89, 96×160×256: 3.19 one wall).

### Where the memory went

`BEAMZ_FUTHARK_LOG=1` (new in `build.py`: Futhark's allocation log) showed,
at the peak inside the pass loop:

- **The initial A was held for the whole loop**, and with the rotating
  design (T recycled from A) the initial T too: up to five states instead of
  three. Generated host code declares every memory block as a function-level
  variable and releases it only when the variable is reassigned or the
  function returns. A loop's initial value, or anything allocated once in its
  first iteration, is therefore never freed early.
- **Padded copies** of the inputs (the plain path's `pad`) stayed alive, and
  `to_tiled` built rows transposed and copied them (2 × 24 B/cell transient).
- **`filter` keeps a block of n elements**: both cell lists cost 4 B/cell
  each however short they were. `source_rows`' histogram over all rows
  (4 B/cell) was held for the whole run.
- `untile` then `crop` wrote every output twice.

### What changed

- **T is compact** (`store`, `item_slots`, `item_at` in `temporal.fut`): it
  holds only the (tile, plane) items with step-1 cells, as `[n][1][128][3]`,
  with a slot per item (`nt·P0` i32, 1/128 of a cell's worth). Items without
  a slot map to a spare row: step 1 computes E on its outermost margin cells
  from H at offsets outside its cells, results nobody reads (see "Margin of
  one cell"), so those reads only must not fault. T is per-pass scratch,
  hoisted to one allocation.
- **A pass holds two whole states (A, C) plus T.** The pass loop starts from
  an empty state and builds the real one in its first pass. Every later pass
  calls `to_tiled` with zero tiles, so the variable holding the first one is
  reassigned and the block freed (see above; a loop-invariant size also got
  the allocation hoisted out of the loop).
- `to_tiled` reads the unpadded inputs directly through a flat scalar
  `tabulate` (no transposed copy). The first pre-E injection goes through
  `tinject`. `untile` writes the cropped outputs. Cell lists are compacted to
  their exact length (`compact`), and `source_rows` is `#[noinline]` so that
  its histogram is freed on return.
- Stores carry a `place = (bool, []i32)` (compact flag, slots) rather than a
  closure. With closures, defunctionalisation specialised `step`,
  `edge_phase` and `field_cpml` per kind of store (132 copies of
  `edge_phase`) and the full build took 113 s. With plain slots alone, the
  branch on `length slots == 0` cost 2.5%. A literal flag folds after inlining:
  80 s and no slowdown. The build was 56 s before; the third `step` call site
  (odd final step) accounts for about 8 s of the first `simplify`.

### A Futhark bug on the way: OOM returns a NULL buffer

With `unified_memory = 0` (now the default in the handler;
`BEAMZ_FUTHARK_UNIFIED=1` restores managed memory), 256×512×512 died with
`CUDA_ERROR_ILLEGAL_ADDRESS` in the core kernel. The generated
`memblock_alloc_device` ignores `gpu_alloc`'s status, and an out-of-memory
return sets no error, so the block was "received" with a NULL pointer
(found with `compute-sanitizer` and `BEAMZ_FUTHARK_DEBUG=1`, also new). Patched
in `build.py` (`_patch_alloc_failure`); see FUTHARK_ISSUES.md. Managed memory
hid both this and the footprint by paging.

### Next

Priorities 2–5 above stand. Memory left per cell: two whole states (48 B,
padded to whole tiles), T for the edge region, CPML memory double-buffered,
and the XLA-side buffers (the benchmark doesn't donate its state, so XLA
holds inputs and outputs too).

## Core kernel: bandwidth, not latency (2026-10-05, night)

Priority 2 of the re-ranked list. Outcome: the tiled path gains 5–10% on
the shapes that fit this GPU, from 16×32 tiles with core cells chosen per
8×16 quadrant. Barrier and occupancy work helped the core kernel in
isolation but hardly in production, because the production kernel is
limited by DRAM bandwidth, not by latency.

### Results (Quadro RTX 5000, 256 steps, GCUPS)

| Shape | One wall, before | One wall, now | All walls, before | All walls, now |
|---|---:|---:|---:|---:|
| 64×96×128 | 2.26 | 2.03 | 1.61 | 1.52 |
| 96×160×256 | 3.44 | 3.40 | 2.06 | 2.16 |
| 128×256×512 | 4.26 | **4.48** | 2.64 | **2.91** |
| 256×512×512 | 4.15 | **4.46** | 3.02 | **3.31** |

Bit-identical to plain stepping (`compare_temporal.py pec mixed pml gaussian
gaussian-right`, CUDA and multicore; a deliberately perturbed core kernel
fails both Gaussian cases, so they do cover core cells). The new
`gaussian-right` case has CPML on one face only, so core tiles reach the PEC
faces of the store. Peak device memory at 256×512×512: 5.77 GB, was 5.61 GB
(padding to whole 16×32 tiles: 513 columns round up to 544, not 528). The
small shapes lose: few large tiles underfill the GPU, and plain stepping
(2.48 at 64×96×128, one wall) beats both, so the fallback in priority 4
covers them. Multicore: tiled 0.33 GCUPS at 128×256×512, unchanged.

HIP (RX 7600 XT, 720-thread blocks, 52.7 KB of shared memory): bit-identical
on the same cases. Tiled GCUPS with one CPML wall, before → now: 1.02 → 0.98
(64×96×128), 1.61 → 1.58 (96×160×256), 2.31 → 2.29 (128×256×512) and 2.02 →
**2.77** (256×512×512). With 8×16 tiles the large shape ran slower than the
smaller one, as on CUDA. All-walls runs were cut short: this card also
drives the desktop, and the large shapes made it unresponsive.

Per pass at 256×512×512, one wall: core 15.9 ms (was 18.4), edge kernels
10.9 ms (step 1 H 2.75 and E 3.95, step 2 H 1.85 and E 2.37), monitors 2.2 ms.

### What the core kernel was doing

Harness: `tiling/core_bench.fut` at 129×257×513, with three different code
arrays (it used to pass one array for all three components, which flatters
material lookups). Generated kernels were inspected with `futhark dev
--gpu-mem` and the CUDA source in the executable.

- **Copies and barriers.** The loop had 14 barriers per plane. Each new H1
  plane and two E1 components were built in a temporary and copied into
  their ring slot, one barrier per copy, and the clamped `min(y + 1, WY - 1)`
  reads of E0 were materialised as two shifted copies. Cause: after
  `r[s] = v`, the simplifier forwards reads of `r[s]` to `v`, which keeps `v`
  alive and blocks short-circuiting (FUTHARK_ISSUES.md). Reading through
  `opaque` slot numbers gives 5 barriers and 20 KB of shared memory instead
  of 25 KB: 7.2 → 7.9 GCUPS in the harness.
- **Trimmed rings.** Each ring holds only what later stages read (H1 11×19,
  E1 10×18, H2 9×17 for 8×16 tiles), so no read needs clamping and fewer
  warps idle: 8.0 GCUPS, 18.8 KB, 3 blocks per SM. A one-row E0 ring cannot
  be updated in place (FUTHARK_ISSUES.md).
- **In production this gave 2%** (core 4.03 → 3.94 ms per pass at
  128×256×512) and nothing at 256×512×512.

### Why: DRAM traffic, mostly halo

Experiments on the 8×16 harness kernel (2.82 ms per pass), each changing
one thing:

| Change | ms per pass |
|---|---:|
| none | 2.82 |
| outputs written to a two-plane buffer that stays in L2 | 2.20 |
| window reads clamped to the block's own tile (no halo from neighbours) | 2.18 |
| constant material coefficients | ≈2.5 |
| more shared memory, forcing 2 blocks per SM instead of 3 | 2.95 |
| component-planar tiles, `[tile][z][3][lane]` | 2.81 |
| bands of 2, 4, 8 tile rows numbered column by column | 2.71, 2.90, 3.07 |

- Removing 263 MB of writes per pass saves 0.62 ms, which is about 420 GB/s,
  the GPU's whole DRAM bandwidth: the kernel is bandwidth-bound. By the same
  rate it moves about 1.2 GB per pass, about 109 B per cell, where reading
  and writing the state once needs 51 B.
- Halo reads from neighbouring tiles account for another 0.64 ms: they mostly
  come from DRAM, not L2. A block's y neighbours run a tile row of blocks
  later. Numbering tiles in bands, so that y neighbours run together, does
  not help (it moves x neighbours apart). Each block also streams its whole
  column, so neighbours drift apart in z; at 256×512×512 the core ran at
  6.1 GCUPS against 7.0 at 128×256×512.
- Time is linear in the number of core tiles (432 → 840 tiles: 1.46 →
  2.82 ms), with no wave quantisation: a lone block runs 7× faster per plane
  than a block on a busy GPU.
- `nvidia-smi dmon` shows the memory controller busy 96–100% of the time.
  CUPTI's profiler (the `profiling_injection` sample builds against
  `/opt/cuda/extras/CUPTI`) would give DRAM and L2 byte counts, but needs
  access to the performance counters
  (`NVreg_RestrictProfilingToAdminUsers=0`, or root).

### Larger tiles, finer edge

Less halo per cell needs larger windows. Harness at 257×513×513 (E0 loaded
per iteration rather than carried, which fits the larger rings):

| Tile | Window | Shared memory | GCUPS |
|---|---|---:|---:|
| 8×16 | 12×20 | 18.8 KB | 6.4–6.7 |
| 2×2 groups of 8×16 tiles per block | 20×36 | 52.7 KB | 7.2 |
| 16×32 | 20×36 | 52.7 KB | 8.6–8.9 |

The 16×32 storage tile beats the same window over four 8×16 tiles. In
production, 16×32 tiles alone made the core 28% faster at 256×512×512 but
grew the edge region by as much (each PEC face costs a whole tile, 16 or 32
cells, and the x walls 32 + 33 columns of 513), so the total did not move.

The kept design separates the two: core cells are chosen per 8×16 quadrant
(`cmask`, a bitmask per tile; `busy_quadrants` for sources), so the edge
region is what it was with 8×16 tiles. The core kernel runs every tile with
a core quadrant through its whole window, clamping reads outside the store.
Results in a non-core quadrant are wrong, but step 2 overwrites every edge
cell, and a core quadrant's result depends only on its own window, which
avoids every boundary. T, the compact scratch state, holds quadrants of
items (`[n][1][128][3]`, slots per quadrant): with whole 16×32 items it took
0.6 GB more at 256×512×512.

### Next

- Small shapes: fall back to plain stepping below a core fraction (priority
  4); this change made them slower.
- The edge kernels run at about half the per-cell speed of plain stepping
  (about 0.32 ns per cell and phase against 0.17, from cell counts estimated
  with the plan rules). Not because of the CPML or PEC
  logic, which plain stepping shares, but because of the edge machinery:
  cell-list decoding, `tile_of`/`lane` and compact slot lookups on every
  read, and scattered writes. Most edge cells are plain interior cells (z
  caps, the tile next to each PEC face).
- The core kernel's remaining traffic: the halo still comes from DRAM, and
  the writes are fixed at K = 2. More steps per pass would cut both, but
  shared memory already limits the 16×32 kernel to one block per SM.

## Edge kernels and monitors: what did not help (2026-10-05, late night)

All measured in production (Quadro RTX 5000, 256 steps, profile per pass of
two steps) and reverted; none moved end-to-end GCUPS beyond noise.

Where the time goes at 256×512×512, from the profile and a numpy model of
the plan rules (core 59.8 M cells, edge 7.9 M one wall; 52.8 M and 14.9 M all
walls):

| Kernel, ms per pass | one wall | all walls |
|---|---:|---:|
| core | 15.9 | 13.8 |
| step 1 H + E | 6.7 | 12.7 |
| step 2 H + E | 4.2 | 10.0 |
| monitor gathers + DFT histograms | 2.3 | 2.3 |

Per cell and step that is about 0.13 ns in the core, 0.5–0.75 ns in the edge
kernels (step 1 dearer than step 2), and 0.31 ns for plain stepping. Plain's
E phase is 0.16 ns per cell, the step-2 edge E phase 0.36 ns with all walls.
Even at 1000³ with 12-cell CPML on all walls, the edge would be about a third
of the time at these rates.

Tried, each against the same baseline:

- **Fast path for interior edge cells** (no slab, boundary or padding case:
  skip the CPML, PEC and slot logic, same arithmetic): step 1 E 143 → 151 ms
  per 128 passes at 128×256×512, one wall. The branching is not the cost.
- **32-bit row indexing** in the store accessors: all edge kernels 2–5%
  slower.
- **Component-planar stores**, as three `[tile][z][lane]` arrays: core
  +12%, edge kernels +2–9%. Interleaved `[lane][3]` is better: a cell's
  three components share a sector. (`[tile][z][3][lane]` was worse still:
  scatters to it cannot fuse with the map computing the values, which
  brought transposes and copies.)
- **Monitors.** The plan gathers all six components with four neighbours,
  but masked components (here Ex, Hx) have zero weights and Ey/Ez one zero
  neighbour. Skipping those in `gather`, and watching only used samples (and
  H samples without their lower neighbours) in `edge_cells`, cut watched cells
  by 29% at 64×96×128 and the gathers by 24% at 128×256×512, but nothing at
  256×512×512, nor end to end, nor for plain stepping. The step-1 monitor
  cost (1.6 ms per pass at 256×512×512, the x columns of the monitor planes
  scattered over tiled rows) is set by the sampled columns, which the change
  keeps.
- **Material codes tiled for the core kernel** (three `[tile][z][lane]u8`
  arrays, so a window row is one 32-byte sector): core 2031 → 2125 ms per
  128 passes at 256×512×512. A constant material instead (wrong results, as
  a bound only) gave 1660 ms, so lookups cost up to 18%, but in the
  dependent code → table loads, not in code bytes from DRAM.

Not tried yet: plain fallback for small shapes (priority 4), and CPML cells
in tiles, the only lever left that changes the edge's per-cell cost by more
than layout tweaks.

## CPU, 2026-10-05

Ryzen 9 7950X, `multicore` build of the Futhark checkout.

**Production (`scripts/benchmark_futhark_jax.py`, 32 steps).** The tiled
path is bit-identical to plain stepping on CPU as well
(`compare_temporal.py`), but K=2 gains little there:

| Shape, CPML | plain | `BEAMZ_FUTHARK_TEMPORAL=2` |
|---|---:|---:|
| 64×96×128, one wall | 0.294 | 0.210 |
| 128×256×512, one wall | 0.299 | 0.308 |
| 64×96×128, all walls | 0.180 | 0.140 |
| 128×256×512, all walls | 0.237 | 0.245 |

Wider tiles (TY×TX = 16×64, 16×128, 32×64, with `LY`/`LX` to match) do not
help: 0.295, 0.256 and 0.264 one wall.

**K scaling, bare stencil (`tiling/bare4.fut`, generic K, 128×256×512).**
Plain one-step `bench_plain`: 0.41 GCUPS. Bitwise checks pass (37×49×70).

| K | 8×32 | 16×128 | 32×64 |
|---:|---:|---:|---:|
| 1 | 0.52 | 0.60 | 0.60 |
| 2 | 0.83 | 1.10 | 1.11 |
| 3 | 1.05 | 1.39 | 1.33 |
| 4 | 1.03 | 1.52 | **1.58** |
| 5 | | 1.52 | 1.57 |
| 6 | | 1.50 | 1.58 |
| 8 | | 1.02 | 1.45 |

So the CPU wants K≈4 and wide tiles (3.8× plain, saturating by K=4–6), but
production's K=2 path gets almost none of it: bare K=2 runs at 2.0–2.7×
plain, production at 1.03×. That gap is not yet explained.
The multicore profile does not name kernels, so it does not show where the
time goes.

### Future exploration: configurable tile size and K

Today only on/off is a runtime choice: `BEAMZ_FUTHARK_TEMPORAL` becomes the
entry's `temporal` argument, and `fdtd.fut` tiles when it is 2 (anything else
steps plainly). The rest is fixed in `temporal.fut`:

- **Tile size** is compile-time: `TY = 8`, `TX = 16` when this was written
  (16×32 since the night of 2026-10-05, with core quadrants of half that),
  with `LY`/`LX` their log2. `tile_of`, `lane` and the row decoding use shifts and masks, so
  `LY`/`LX` must match powers of two; a mismatch segfaults rather than
  failing cleanly. `TY * TX` appears in the storage types (`tiled`, `store`),
  so it also fixes the layout and the block size. The window is
  `WY = TY + 4`, `WX = TX + 4` (a halo of K = 2 per side). To try another
  size today, edit these four `def`s (that is how the CPU tile sizes above
  were measured).
- **K = 2 is structural.** `core_pass` is written out for two steps (rings
  E0, H1, E1, H2 and a fixed sequence of four plane updates). The halo, the
  margin computed twice, and the edge plan (`interior`, `make_plan`,
  `edge_cells`) all assume a two-step pass.

Ways to open these up:

1. **Tile size per build (cheap).** A `build.py` option that rewrites
   `TY`/`TX` and derives `LY`/`LX` (as `tiling/gen.py` does for experiments),
   rejecting non-powers of two, with a default per platform (8×16 on CUDA,
   wider on CPU). Keep it compile-time: a runtime tile size would make block
   and shared-memory sizes dynamic, which likely costs speed on the GPU.
2. **K as a parameter (real work).** Generalise `core_pass` to K stages
   along the lines of `bare4.fut`'s `rpassk` (halo 2K), rework the edge plan
   for a K-step margin, and re-verify bit-exactness with
   `compare_temporal.py`. On CUDA, K = 3 already ran out of shared memory with
   larger tiles in the `bare4` experiments, so this mainly matters for CPU,
   where the bare stencil peaks at K = 4–6.

Before investing in K on CPU, explain why production's K = 2 gains only 3%
there while the bare stencil's K = 2 gains 2.0–2.7×. Wider tiles alone did
not help production (see above). Note that `tiling/lean.fut`, the
intermediate between `bare4` and production, no longer compiles (it uses
`route`, since removed from `yee.fut`) and needs updating first.

## CPML in the tile pass: a general shell kernel (2026-10-05, late night)

Goal: take the CPML (and the other edge cells) out of the plain edge steps,
which run at about 0.6 ns per cell-step, and into a two-step tile kernel.
Overlapped first, since the checkerboard's black and white kernels are
variants of the same kernel (halo 0, and halo (2, 3) with partial levels).

Baseline rerun today (256 steps, GCUPS): tiled 4.48 / 4.47 one wall, 2.91 /
3.30 all walls (128×256×512 / 256×512×512); plain 3.01 / 2.98 and 2.50 / 2.67.

### What exists (uncommitted, not used by fdtd.fut yet)

- `yee.fut`: `stretched`/`field_cpml`/`field_value` read CPML memories
  through accessors (`field`), callers wrap arrays in `at3`; new
  `term_memory` recomputes the memory a term advances to. fdtd.fut still
  type-checks; **the production build and its speed are not rechecked yet**.
- `temporal.fut`: `shell_pass` (with `route`, `pick`, `pmem`, `comp1`,
  `comp`, `umem`). One block per job (tile quadrant `tile * 4 + q`, z start,
  `zc` planes), window = 8×16 quadrant + 2-cell halo, the four stages of
  `core_pass` but each update is yee.fut's `field_cpml` through ring
  accessors. Step 1's CPML memories are not stored: step 2 recomputes them
  (`umem`) from E0 (ring of 2 planes) and H1 (ring of 3 planes). Results go
  to per-job columns `[nj][zc][128]` of E, H, 6 H memories and 6 E memories
  (still to be scattered into C and the packed slabs).
- `tiling/shell_bench.fut`: synthetic grid with CPML of width w on chosen
  faces, PEC elsewhere, dense decay arrays, codebook E sources. `check`
  compares one shell pass over every quadrant (any z chunking) with two plain
  yee.fut steps; `bench`/`cells` time it on the quadrants make_plan leaves to
  the edge; `bench_plain` times plain steps. Compile takes ~4 min.

### Results

- **Bit-exact**: `check` gives 0/0/0 mismatches (E, H, memories) on
  40×50×70, 29×37×70, 33×40×90, 100×60×60, 33×100×200 and 128×256×512, with
  CPML on all, some or no faces and z chunks of 7–200 planes. Perturbing one
  input memory by 1e-3 gives 222k/145k/280k mismatches, so the check bites.
  (A first version read out of bounds: the out stage evaluated updates
  before its `ok` guard; small grids hid it.)
- **Slow**: at 128×256×512, ns per cell-step on the shell quadrants:

| CPML | ms per pass | ns per cell-step |
|---|---:|---:|
| none (w=0, PEC quadrants only) | 4.91 | 0.58 |
| x high face only | 5.43 | 0.64 |
| z faces only | 5.13 | 0.61 |
| all faces | 8.52 | 0.82 |

  Edge kernels in production: 0.5–0.75; plain in this harness 0.44
  (average over all cells, dense decay). So no gain as it stands.
- **Why**: not CPML (0.58 without any) and not occupancy (176 registers,
  1 block of 240 threads per SM; `__launch_bounds__(…, 2)` via
  `--dump-cuda`/`--load-cuda` gives −3%, `(…, 3)` spills and is 20% slower;
  `--nvrtc-option=-maxrregcount` has no effect, it never reaches the JIT).
  The kernel is 7024 SASS instructions: 2376 IMAD, 1122 ISETP, 1067 IADD3,
  351 SEL, 319 LEA against 72 FFMA, 120 FMUL, 48 FADD. It is integer index
  arithmetic: 64-bit ring/window/tile indexing behind every accessor read,
  boundary and slab tests, dims picked per component, dense coefficient
  indexing. Matches lean.fut's earlier finding that the production `cell`
  doubles the cost of a lean update.

### Where to resume

1. Make the harness production-like first: scalar decays/H source (the
   common configuration the core kernel requires), then remeasure.
2. Write a lean shell update, as `hcore`/`ecore` plus CPML terms, bit-exact
   with `field_cpml`: 32-bit window indices computed once per cell, slab
   rows/columns/planes and packed indices precomputed per window row,
   column and plane (per (phase, axis), checking the per-term slabs agree),
   PEC as explicit masks (beware −0 vs +0 at the high face: `0 - x` is not
   `-x` for x = +0). lean.fut's lean z-CPML `upd` reached 0.30 ns at 4×32,
   halo 2; aim for ≤ 0.3 here, against 0.6 for the edge kernels.
3. Only then the checkerboard (halo 0 black: lean z-CPML 0.15 ns in the
   proxy), and the integration: scatter job columns into C and the packed
   slabs, shrink the step-1/step-2 cell lists to source quadrants and
   monitors, z-cap jobs for core tiles.
4. Recheck production build time and edge-kernel speed after the yee.fut
   accessor change (or revert it if the lean kernel doesn't use it).

## Lean CPML tiles: halo and registers (2026-10-06)

Questions: does a checkerboard (less halo) pay for either tile type? Can
CPML tiles beat plain stepping once they waste no compute or bandwidth? Is
the CPML tile slowed by its halo's bandwidth? Which tile size suits each
kernel?

Quadro RTX 5000; 128×256×512 cells (store 129×257×513), CPML width 12;
"faces" as in the harnesses (63 all, 32 x high, 3 z). Built with a private
worktree of the Futhark checkout at 32f49631b: the checkout had uncommitted
intra-block result changes, with which this kernel's outputs went to shared
memory and the launch failed.

### Core tiles: halo is at most 15%

`core_bench` at 257×513×513 with window reads clamped to the tile's own cells
(no neighbour halo traffic; wrong results): 11.5 → 9.8 ms per pass (8.5 →
10.0 GCUPS). That bounds any scheme that reduces the halo. The staggered
checkerboard halves it and adds border history, so ≲7%: not worth a second
kernel.

### CPML tiles are not bandwidth-bound

`nvidia-smi dmon`, memory controller busy: core kernel 100%, `shell_pass`
32%, the lean kernel below 43%. In the lean kernel (8×16 jobs), with
CLAMP (window reads from the job's own cells: no halo traffic) the time drops
6–17%; with SKIP (no compute outside the job, same block shape) 1–13%.
Neither the halo's bytes nor its work dominates. Registers do (below).

### The lean kernel

`tiling/cpml_bench.fut`, `cpml_pass`: two steps for jobs of JY×JX cells
(whole columns, any origin), window = job + 2, z streamed through rings as in
`core_pass`. Production's common configuration (isotropic, scalar H
coefficients and E decay, codebook E sources), yee.fut's CPML and PEC rules
with literal axes and 32-bit coordinates, CPML memories in tiled layout
(`[tile][z][lane][6]` per phase). Step 1's memories are recomputed in step
2 from E0 (two planes kept) and H1 (three planes) rather than kept in rings
(`RECOMP`). `check` compares one pass over every job with two plain
yee.fut steps bitwise. 0 mismatches on seven shapes (CPML on all, some or no
faces; 20×30×40 up to 128×256×512), and compute-sanitizer is clean. An
injected CPML bug gives 240k mismatches, a boundary bug 41k.

Two out-of-bounds reads on the way, both for halo cells outside a
component: yee.fut's `packed` is non-negative for x ≥ n (yee.fut only gets
there after its `inside` test), and the codebook lookup needs the component's
extent, not the store's.

Progress at 8×16 jobs, ns per cell-step (job cells, two steps per pass):

| Version | all faces | x high | z | no CPML | barriers per plane |
|---|---:|---:|---:|---:|---:|
| `shell_pass` (previous) | 0.82 | 0.64 | 0.61 | 0.58 | |
| lean, packed indices | 0.67 | 0.56 | 0.53 | 0.51 | 71 |
| + coefficient tables, inlined helpers | 0.54 | 0.50 | 0.41 | 0.39 | 8 |
| + step-1 memories recomputed | 0.52 | 0.41 | | 0.30 | 8 |

**Block-level branches cost a barrier each.** Futhark hoists what depends
only on the plane (the z terms' packed index and coefficients, edge-bit tests)
out of the per-cell map. In an intra-block kernel every hoisted `if` then
runs at block level between barriers: 71 per plane. Precomputed per-coordinate
coefficient tables (unconditional loads), edge flags as booleans and
`#[inline]` on small helpers (a non-inlined function inside `&&` became an
`if`) leave none. Loading coefficients lazily (only inside a slab) put the
branches back (57 barriers, slower).

**Where the time goes** (no CPML, 8×16, tables version; probes are compile-time
switches, wrong results):

| Variant | ns/cell-step | registers |
|---|---:|---:|
| core arithmetic (`BARE`), no memory rings or outputs | 0.12 | 63 |
| `BARE` with memory rings and outputs | 0.20 | 76 |
| general update, no CPML terms (`NOCPML`), no memory plumbing | 0.17 | 69 |
| general update, no boundary tests (`NOBND`), no memory plumbing | 0.16 | 117 |
| general update, no memory plumbing | 0.34 | 169 |
| general update | 0.39 | 165 |

The streaming structure is as fast as the core kernel. CPML terms or
boundary tests alone are cheap; together they need about 165 registers, which
allows one 240-thread block (7.5 warps) per SM. `__launch_bounds__(…, 2)`
(edited into `--dump-cuda` output, run with `--load-cuda`) takes the
no-plumbing kernel from 0.34 to 0.22 (all faces 0.46 → 0.32), but only once
shared memory allows two blocks: 35 KB with memory rings, 21 KB with
`RECOMP`. A cap for three blocks spills and is slower. A thread per
component and cell (`cpml_pass3`: 80 registers, 720 threads) was no faster
(0.57/0.50/0.47/0.45).

### Job size

`RECOMP` kernel, best of no cap and the launch-bound caps, ns per cell-step
(repeat runs agree within 1%; across sessions within about 10%):

| Job (window) | no CPML | x high | all faces |
|---|---:|---:|---:|
| 8×16 (12×20) | 0.30 | 0.41 | 0.52 |
| 4×32 (8×36) | 0.35 | 0.44 | 0.49 |
| 8×32 (12×36) | 0.30 | 0.34 | 0.41 |
| 16×16 (20×20) | 0.32 | 0.34 | 0.39 |
| 16×32 (20×36) | 70 KB shared memory, does not launch | | |

Larger jobs help: less of the window is halo. 400–432-thread blocks get
120 registers without spills (with a cap; 128 and spills without), one block
per SM. 16×32 needs less shared memory first.

### Against plain stepping

`bench_plain` (yee.fut's `phase_next`/`psi_next`) in the same harness, per
step: 0.29 ns per interior cell, and from the CPML cell counts 0.53–0.58 per
CPML cell (all faces, w = 12 and 40), 0.49 (z faces), 0.77 (x high face).
Production measured 0.31 and 0.62. So CPML tiles at 0.34–0.39 are 1.5–2.2×
faster than plain stepping on CPML cells. Without CPML they only match plain
interior speed: a CPML tile's gain is handling CPML cells at about the cost
of plain interior ones.

*Estimate* for production at 256×512×512, if CPML tiles at about 0.4 ns took
all edge cells: edge kernels 10.9 → 6.3 ms per pass with one wall (7.9 M
cells; total 29 → 24 ms, +19%), 22.7 → 11.9 with all walls (14.9 M; total
39 → 28 ms, +39%). Sources and monitors would stay plain, and the outputs
would have to reach C (a scatter, or a kernel that writes C's tiles).

### Checkerboard for CPML tiles

A black (halo 0) 8×16 tile has the block shape of an overlapped 4×12 job
(window 8×16). As a proxy, that job costs 34 / 41 / 48 ns per plane (no CPML /
x high / all faces), i.e. 0.13 / 0.16 / 0.19 ns per cell-step for 128 black
cells. A white tile (halo 2 to 3, plus loading black's partial levels) costs
at least the overlapped 8×16 job, 0.30 / 0.41 / 0.52. The mean, 0.22 / 0.29 /
0.36, is up to about 25% better than overlapped 16×16 jobs (0.32 / 0.34 /
0.39) before the white tile's extra loads. A second kernel and partial-level
storage for that: not before the items below.

### Next

Superseded: items 3 and 4 were done next and the result was negative, see
the following section. Items 1 and 2 would speed up the kernel, but not
enough to make it pay.

1. A launch-bounds attribute in Futhark (or a `build.py` patch of the
   generated CUDA): 20–35% once shared memory allows two blocks.
2. Fewer registers: specialise jobs by which slabs and faces their window
   touches (uniform per job in x and y, per plane in z), so a job carries
   only the terms it needs.
3. Integration: tiled CPML memories (packed slabs converted at entry and
   exit), jobs for non-core quadrants plus core quadrants' z caps, outputs
   into C, sources and monitors stepped plainly.
4. `shell_pass` and the accessor change in yee.fut are superseded by this
   kernel; revert them before integrating.

## CPML tiles in production: a split state (2026-10-06)

Next steps 3 and 4 of the previous section, ahead of 1 and 2, to measure the
end-to-end gain first. Outcome: correct and slower. The patch is kept in
`tiling/split_state.patch`; it applies to a clean checkout of a16d1d85.
Built with a private worktree of the Futhark checkout at de8e16e5f.

### Design

Results must reach the state without a copy pass: copying the CPML kernel's
outputs into C would cost about half its gain. So E and H are stored split
three ways, in whole z columns:

- K: the tiles with a core quadrant, `[tile][z][lane][3]`, valid on core
  cells only (the core kernel's output);
- XC: every other quadrant, `[slot][z][qlane][3]` (the CPML kernel's
  output, one 8×16 job per quadrant column);
- XZ: the z caps of the core quadrants, planes `[0, zb)` and `[zt, P0)`,
  `[2 * pair + top][D][qlane][3]` (the CPML kernel again, with jobs of
  `len` planes from `kb`).

CPML memories live only in XC and XZ (`[..][6]`): core cells never lie in a
slab. Every read goes through `locate` (per tile its K slot, per quadrant its
XC slot or cap pair). Plain steps remain only for source quadrants (step 2)
and for those plus their one-cell margin and the monitored cells (step 1), as
before. An odd last step and runs outside the common configuration use the
plain path, from run storage. yee.fut's `stretched` reads memories through an
accessor `psi p k j i`, so that plain steps read either layout.

Bit-identical to plain stepping on `compare_temporal.py pec mixed pml
gaussian gaussian-right`. A perturbed CPML kernel (E2 decay × 1.001) fails
them. The 256×512×512 runs failed with an unspecified Futhark error, not
investigated (peak memory at 128×256×512: 2.5 GB with one wall, 3.0 GB with
all walls).

### Results (Quadro RTX 5000, 128×256×512, 256 steps, ms per pass)

| | one wall, committed | one wall, split | all walls, committed | all walls, split |
|---|---:|---:|---:|---:|
| core kernel | 3.65 | 4.66 | 2.58 | 3.30 |
| plain edge steps | 3.32 | 0.75 | 8.42 | 0.77 |
| CPML kernel, columns | | 4.67 (193 jobs) | | 6.87 (253 jobs) |
| CPML kernel, caps | | 1.72 (1792 jobs) | | 5.55 (1672 jobs) |
| copy (one state, layout) | | 1.35 | | 1.18 |
| total | 7.52 | 13.67 | 11.56 | 18.21 |
| GCUPS | 4.48 | 2.47 | 2.91 | 1.85 |

- **The core kernel is 28% slower.** Same kernel, same shared memory
  (59 KB), 80 registers, but every window read goes through `locate`. The
  harness check that suggested the lookups were free
  (`tiling/split_bench.fut`, 10.8 ms per pass at 257×513×513) was compared
  against an older session's core_bench number, not run side by side.
- **The CPML kernel is 1.6–1.7× slower than in its harness:** 0.82 ns per
  cell-step on columns and 0.89 on caps with all walls (harness: 0.52 at
  8×16 jobs), 0.76 with one wall (harness 0.41). Causes: `locate` behind
  every read (189 registers instead of 165, still one block per SM), and
  wave quantisation (193 jobs on 48 SMs run as 5 waves for 4.02 waves of
  work). Shallow caps pay three lead-in planes each: with PEC z faces a cap
  is 3 planes and takes 6 iterations.
- **More cells than the plain edge.** Quadrant columns include the padding
  slivers (store row 256 and column 512 cost a whole quadrant row or column)
  and core quadrants' z caps; with all walls the CPML kernel covers about
  7.3 M cells against 6.1 M edge cells for plain stepping.
- **Plain edge cells cost 0.69 ns per cell-step** (all walls), and that is
  what the CPML kernel has to beat on the same cells. At its harness speed
  (0.52 at 8×16, 0.39 at 16×16) on the cells above, the edge would cost
  7.6 or 5.7 ms against 8.4, minus about 0.7 ms for the slower core: from
  nothing to about 15% with all walls, and nothing with one wall.

### Futhark pitfalls met on the way

- **Flattened kernel results are copies.** `flatten (flatten r)` of an
  intra-block result forced a transpose or copy of each state (5.5 ms per
  pass); keeping `[item][plane][lane][3]` arrays removed most of it. One
  1.2–1.35 ms state copy per pass after the core kernel remained, not
  traced.
- **"No global alias for intra-block result"** at de8e16e5f: core_bench.fut
  with both its `bench` and `digest` entries fails (either alone compiles).
  The result tiles' LMAD gets an existential stride (`lmad_arg = 3 * n`),
  which `LMAD.isDirect` does not recognise as direct. Reproducer and
  analysis: `../futhark/beamz-repros/` (README.md, global_alias_lmad_arg/).
- **Size expressions over record fields crashed the internaliser**
  (`Exps.hs:1842`, non-exhaustive patterns in the binary-operator handler)
  when one function returns arrays sized by a record parameter's field
  and another returns several, e.g. `length xs * g.P0 * QL` and
  `... pl.D ...`. Passing the extents as scalar parameters avoids it.
  Four-line reproducer: `../futhark/beamz-repros/size_record_field.fut`.
- **Local functions do not generalise `[]` sizes**: unannotated `[]`
  parameters of a local function unify across uses, so a helper used with
  both K and XZ arrays needs explicit size parameters.

### What this means

CPML tiles pay only where CPML cells dominate the edge, and there by little:
even at harness speed the kernel is only about 1.3–1.8× faster per cell
than plain edge steps, and quadrant rounding and cap lead-in give it more
cells to update. For the target, large domains with a thin shell, the edge
is mostly non-CPML cells (z caps, PEC-adjacent tiles), where the kernel's
advantage is smallest. Steps 1 and 2 of the
previous section's list (launch bounds, job specialisation) would speed up
the kernel itself, but not the core kernel's lookup penalty, which a split
state brings. Not recommended as the next lever.

## Calibration: how far from the limits (2026-10-06, evening)

Item 1 and 2 of the list below, checked before starting on them. Built
with a private worktree of the Futhark checkout at 8a1cdc1de. The working
tree's superseded `shell_pass` and yee.fut accessor change are now in
`tiling/shell_pass.patch` and reverted. (`--case` runs of
`benchmark_futhark_jax.py` need `XLA_PYTHON_CLIENT_PREALLOCATE=false` by hand;
only the launcher sets it, and without it 256×512×512 fails with "unknown
Futhark error", which is what the split-state runs hit.)

### Bandwidth of this GPU

A native CUDA test (float4, grid-stride, 1 GiB arrays; `nvcc -arch=sm_75`)
reaches 294 GB/s copying and 361 GB/s reading. A Futhark `map` copy reaches
321 GB/s. Earlier sections assumed 420 GB/s (the spec is 448), and so
overstated the core kernel's traffic.

### Core kernel: production equals the harness

Per tile cell and pass: production 0.226 ns (15.35 ms, 561 blocks, about
528 with a core quadrant over 251 planes, at 256×512×512, one wall),
`core_bench` at 257×513×513 0.228 ns (11.1 ms for 48.8 M cells). There is
no production penalty to find. At 300–320 GB/s, 0.227 ns is 68–73 B per
cell. What the kernel must move: 24 B read and 24 B written per cell, 3 B of
material codes, and the halo, if it all came from DRAM, 8.5 B more (E window
20×36 and H 19×35 per 16×32 tile): about 60 B. So the kernel is within about
1.2× of its own traffic. The remaining levers are those already bounded:
the halo (at most 15%) and K > 2. K = 3 does not fit Turing's 64 KB of
shared memory with 16×32 tiles (rings about 80 KB); with 8×32 tiles it fits
(about 59 KB) but cuts DRAM bytes per cell-step by only about 17%, for more
redundant compute. Larger GPUs (100–228 KB of shared memory) could run K = 3
or 4 with 16×32 tiles.

The generated kernel stages each result plane in shared memory and copies it
out coalesced (two barriers, 12 KB); the writes cost no extra DRAM traffic.

### Edge kernels: machinery 1.3× plain, the rest is CPML

128×256×512, all walls, ms per pass (two steps):

| | plain stepping | tiled, all cells edge | tiled (production) |
|---|---:|---:|---:|
| cells stepped as edge | | 17.0 M | 6.15 M |
| step 1 H + E | | 7.9 | 4.6 |
| step 2 H + E | | 8.5 | 3.9 |
| ns per edge cell-step, step 2 | 0.36 (all cells, with CPML kernels) | 0.50 | 0.63 |
| GCUPS | 2.44 | 1.88 | 2.85 |

"All cells edge": a build with `common` false, so that no tile is core.
Over the same cells plain stepping does, the edge kernels cost 1.3–1.4× per
cell: that is the cost of cell lists, tiled indexing and the compact T store.
In production they cost 1.26× more per cell again, because there the edge is
mostly CPML cells (a 12-cell slab plus the 2-cell halo, rounded to 16), which
read and write their memories and coefficients, against about 30% of all
cells. The earlier "0.69 vs 0.31 ns" compared CPML-dense edge cells with
plain stepping's average over a mostly non-CPML domain.

So making the edge kernels as fast as plain stepping would save at most
about a quarter of the edge time (with all walls at 256×512×512 the edge is
about 23 of 39 ms, so about 15% end to end), and CPML tiles, which would
change that, are ruled out.

### The halo's 15% is not DRAM traffic

Can neighbouring tiles share the halo through L2? `core_bench` at
257×513×513, kernel time from CUPTI activity tracing (below), ms per pass:

| Variant | ms |
|---|---:|
| production kernel | 11.5–11.7 |
| window reads clamped to the tile (no halo; wrong results) | 10.0 |
| halo reads redirected to an L2-resident 1.5 MB region (same address pattern) | 11.9 |
| same, redirected to a few hot lines | 12.3–12.6 |
| z chunks of 64 / 32 planes, dispatched chunk-major | +5% / +10% over one chunk |
| window loaded by a flat index (coalesced; 720 threads × 3 floats) | 13.7–13.8 |

With every halo byte served from L2 the kernel is not faster, so ordering,
chunking or synchronising neighbours (anything that turns halo DRAM reads
into L2 hits) cannot recover the 15%. What clamping removes is L1 misses:
a window row spans three tiles' rows 1.5 MB apart, so a warp's loads touch
more sectors and lines. Chunking costs only its lead-in planes (3 per
chunk).

A flat-index load (consecutive threads read consecutive floats) was 18%
slower: it needs 64-bit divisions by 3 and by the window width, and an extra
barrier or interleaved shared reads.

**This kernel is sensitive to small code changes.** The single-chunk
version of the chunked kernel was 20% slower (13.7 vs 11.5 ms) with the same
grid, shared memory (59 280 B), registers (64, no spills), allocations and
near-identical SASS (3744 vs 3720 instructions, same loads, stores and
barriers). Each of its two source changes alone (block → tile/chunk
arithmetic; output rows and write index) was as fast as the original
(11.3, 11.5 ms), and the write offset did not matter (13.6–13.8 for 0–64
planes). Unified memory on or off made no difference. With one block per
SM and five barriers per plane, the kernel depends on how early loads issue,
which suggests it is partly latency-bound after all: prefetching the next
plane would address that, but needs shared memory (6 KB of headroom under
Turing's 64 KB; the result staging takes 12 KB).

**Kernel timing without the profiler counters:** CUPTI activity tracing
needs no admin rights. Build `/opt/cuda/extras/CUPTI/samples/cupti_trace_injection`
(copy it and `../common`, `make CUDA_INSTALL_PATH=/opt/cuda`), then run any
executable with `CUDA_INJECTION64_PATH=.../libcupti_trace_injection.so` and
`LD_LIBRARY_PATH=/opt/cuda/extras/CUPTI/lib64`: it prints every kernel's
duration (`CONCURRENT_KERNEL ... duration NS, "name"`). The first call of a
kernel writing fresh output pays about 100 ms of first-touch. Futhark
executables' `-P` prints no report in this build.

### What this means

On this GPU, at K = 2, both kernels are within about 25% of what their
design allows. Further gains of the size seen so far need a change of
design or of hardware: K > 2 (more shared memory), fewer bytes per cell
(lower precision or compressed state: not bit-exact), or a different CPML
scheme. Hardware counters would confirm the byte counts above; they need
`NVreg_RestrictProfilingToAdminUsers=0` (`RmProfilingAdminOnly` is 1 now)
and a CUPTI or Nsight Compute build (neither `ncu` nor `nsys` is installed).

## Where to resume (2026-10-06)

Production is the committed core-plus-plain-edges design (a16d1d85):
4.48 / 4.46 GCUPS with one wall at 128×256×512 / 256×512×512, 2.91 / 3.31
with all walls, against plain stepping's 3.0 / 3.0 and 2.5 / 2.7. Per pass
(two steps) at 128×256×512, one wall: core kernel 3.65 ms, plain edge steps
3.3 ms, of 7.5 ms. At 256×512×512, one wall: core 15.9, edge 10.9, monitors
2.3, of about 29 ms.

The superseded `shell_pass` (temporal.fut) and the yee.fut accessor change
are reverted and kept in `tiling/shell_pass.patch` (`shell_bench.fut` and
`cpml_bench.fut` need it applied).

Done or ruled out: CPML tiles in any form (split state, checkerboard); edge
layout and indexing tweaks (see "Edge kernels and monitors: what did not
help"); barrier and occupancy work on the core kernel (bandwidth-bound).

Open, roughly by expected value for large domains with a thin shell
(*estimates*; items 1 and 2 revised by the calibration above):

1. **Core kernel traffic.** Within about 1.2× of its own reads and writes
   (see Calibration). The halo's 15% is not DRAM traffic, so L2 sharing
   cannot recover it (see "The halo's 15% is not DRAM traffic"). Left:
   latency (prefetching the next plane, if shared memory allows), and
   K > 2, which needs more shared memory than Turing has.
2. **Plain edge cost.** The machinery costs about 1.3× plain stepping per
   cell; the rest is CPML. At most about 15% end to end with all walls.
   Stepping the z caps or PEC faces in the core kernel would shrink the edge
   of the one-wall proxy, but not the target's, whose edge is the CPML shell.
3. **Small shapes**: fall back to plain stepping below a core fraction
   (64×96×128 is slower tiled than plain).
4. **Sources and monitors** are stepped plainly; a mode-source plane makes
   whole quadrant columns plain. Monitors cost 2.3 ms of 29 per pass at
   256×512×512, one wall; source quadrants not measured separately.
