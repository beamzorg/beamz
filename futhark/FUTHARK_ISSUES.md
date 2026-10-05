# Futhark issues met by BeamZ's backend

Compiler bugs, former patches to generated code, and code-generation
behaviour that cost performance. BeamZ found these on Futhark 0.27.1 and now
builds with the Futhark checkout next to this repository (`../futhark`, or
`FUTHARK_SRC`; branch `all-fixes` of
[pepijndevos/futhark](https://github.com/pepijndevos/futhark), on 0.28.0
prerelease), which fixes all of them. Each fix is also on its own branch,
reported upstream. `build.py` and `tiling/fc.sh` build that checkout with
`cabal` (from `~/.ghcup`) and use its compiler; set `FUTHARK` to override.

## Compiler bugs

### 1. `reduce_by_index_3d` with array-valued elements: internal error

```futhark
entry main [n][m][k][l] (f: *[n][m][k][3]f32) (is: [l](i64, i64, i64)) (vs: [l][3]f32) : *[n][m][k][3]f32 =
  reduce_by_index_3d f (map2 (+)) [0, 0, 0] is vs
```

```
Internal compiler error.  Please report this:
After internalisation:
In function defunc_0_reduce_by_index_3d_7567
...
expecting 2 arguments of type(s)
[m_7740]f32, [3i64]f32
```

The element size gets confused with the array's second dimension `m`.
Fails on both backends.
Fixed in the fork (`3b28a76f4`). `temporal.fut`'s `tinject` now uses
`reduce_by_index_3d` on the tiled store directly; its workaround (a 1-D
histogram per target row, then `scatter_3d`, plus `source_rows`, a
histogram over every tiled row) is gone.

### 2. Two fused multi-dimensional histograms: type error after fusion

```futhark
entry main [n][m][k][l] (a: *[n][m][k]f32) (b: *[n][m][k]f32) (r: [l]i64) (v: [l]f32) =
  let idx = map (\x -> (x / (m * k), (x / k) % m, x % k)) r
  in (reduce_by_index_3d a (+) 0 idx v, reduce_by_index_3d b (+) 0 idx (map (* 2) v))
```

```
Type error after pass 'Fuse SOACs':
Bucket function has return type {i64, i64, i64, i64, i64, f32, i64, i64, f32, i64, i64, f32}
but should have type {i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, f32, f32}
```

(That message is from the three-histogram version; two are enough to
fail.) Horizontal fusion interleaves the index and value results per
histogram where the fused operator expects all indices first. The 1-D
`reduce_by_index` version of the same program compiles.
Fixed in the fork (`37cb494e5`).

## Former patches to generated code, now compiler features

`build.py` used to patch Futhark's generated C. Each patch is now a fix or
option in the checkout:

1. **Results of intra-block kernels go to global memory** (was
   `intrablock.py`, `tiling/patch_cuda.py`). Futhark keeps every block-level
   array of an `#[flattening(only_intra)]` kernel in shared memory, the
   block's results included, and copies them out when the block ends. A
   temporal tile streams a whole z column, so its results cannot fit. Now:
   the preliminary attribute `#[intrablock_result_global]` next to
   `#[flattening(only_intra)]` (`32f49631b`, CUDA and HIP only). Same kernel
   speed as the patch. It assumes each block's result is laid out
   contiguously like its slice of the global result.
2. **CUDA primary context** (was `_patch_primary_context`). XLA uses the
   device's primary context, and raw device pointers only alias across the
   two if Futhark retains it. Now `futhark_context_config_set_use_primary_context`
   (`93f060de3`), which the generated handler calls. HIP has one context per
   device and needs nothing.
3. **ISPC stdlib clash** (was `_patch_ispc_stdlib`). Futhark declared
   `erf`/`erfc(double)` externs that ISPC 1.31's stdlib already defines.
   Fixed in `5a3853e8f`.
4. **Out of device memory yielded a NULL buffer** (was `_patch_alloc_failure`).
   0.27.1's generated `memblock_alloc_device` discarded `gpu_alloc`'s status
   and checked only `ctx->error`, which an out-of-memory return does not
   set; the next kernel then faulted with `CUDA_ERROR_ILLEGAL_ADDRESS`
   (found at 256×512×512 with `unified_memory = 0`). Fixed in `a062c7eca`.

The 0.28 manifest names an entry input's uniqueness `consumed` (was
`unique`); `build.py` reads the new key.

## Moving to the checkout (2026-10-05)

Quadro RTX 5000, `scripts/benchmark_futhark_jax.py --backends futhark
--pml-edges right`, GCUPS, plain / `BEAMZ_FUTHARK_TEMPORAL=2`:

| Build | 128×256×512 | 256×512×512 |
|---|---:|---:|
| 0.27.1 + patches | 2.99 / 4.11 | 2.92 / 3.91 |
| checkout, same `.fut` (attribute instead of patch) | 2.99 / 4.01 | 2.91 / 3.72 |
| checkout, per-component window rings (below) | 2.99 / **4.18** | 2.92 / **3.93** |

All walls at 128×256×512: 2.60 tiled. On the RX 7600 XT (128×256×512, 64
steps, one wall) the checkout build is unchanged: 1.98 / 2.18 before, 1.97 /
2.21 after.

**The 5% core-kernel regression, bisected.** With the old `.fut`, the tiled
path's core kernel was 5% slower on the checkout (4.05 → 4.26 ms per call at
128×256×512), with every other kernel unchanged. Bisecting the 105 commits
between v0.27.1 and the fork's base (production build and benchmark per
step) gives upstream `2164c474b` "Simplify indexing of reshapes via symbol
table" (4.13 on its parent, 4.01 on it). Before it, the compiler could not
see through a reshape and copied each shifted H row out of the
component-interleaved `[12][20][3]` shared window into a flat `[12][20]`
shared array; after it, it reads the interleaved window directly. That is
fewer instructions and barriers, yet slower: per-component planes suit the
hardware better than stride-3 reads. Fix in `temporal.fut`: the window rings
are three `[n][WY][WX]` arrays, one per component (`rget`/`rset`), which beats
both earlier versions.

## Code-generation behaviour that cost performance

These are not bugs, but each one cost hours; see TEMPORAL_TILING.md for
measurements.

- **Loop-carried arrays rebuilt every iteration are double-buffered in
  global memory**, even in intra-block kernels whose parameters live in shared
  memory. Every plane went to device memory and back. Fix: fixed ring slots
  updated in place (`ring[slot] = tabulate …`).
- **`flatten` of a loop-carried array copies it in full** when Futhark does
  not know its outer strides (existential layouts on loop parameters, e.g.
  when the initial value and the loop result come from different kernels).
  This gave 2–4 copies of 70–850 MB per pass. Use `scatter_3d` on the
  unflattened array. `map (map flatten) f` copies as well.
- **A sliced row in an intra-block `tabulate` becomes a parallel dimension**
  (`tabulate n (\l -> ring[s, y, x])` with `[3]` rows gave 384-thread blocks
  instead of 240). Build rows as literals: `[v[s, y, x, 0], v[s, y, x, 1], …]`.
- **Block size is always the largest inner parallel extent.** There is no
  attribute for block size, register arrays or coarsening.
- **`if` choosing between two loop variants in an intra-block kernel**
  allocates both branches' shared arrays, and the copy-out reads from a
  branch-dependent pointer.
- **A per-tile `tabulate_3d` fused into another kernel over tiles** stores
  its per-thread result interleaved (stride = number of threads). Readers
  then got uncoalesced accesses. Fix: a flat 1-D `tabulate`.
- **Indexing `a ++ b ++ c` fuses into a chain of branches** at every read
  (`index_concat`). Pass the parts separately when the part is known
  statically.
- **Managed memory by default.** The CUDA backend's `unified_memory`
  defaults to 2 (auto), which means managed memory on GPUs that support it.
  Oversubscribing the GPU then pages instead of failing: 10× slower kernels
  and 3.6 s of one-off cost per call at 256×512×512. Consider setting
  `futhark_context_config_set_unified_memory(cfg, 0)` in the handler.
- **Closures capturing a large record** are copied field by field at every
  application after defunctionalisation; the first `simplify` pass then
  blew up compile time (212 s → 63 s once fixed). Capture only the scalars
  used.
- **Profiling the first call** measures cold memory: early passes ran about
  6× slower. Use the second report.
- **Reading a ring slot right after updating it forwards the read to the new
  value**, which keeps that value alive as a separate array: array
  short-circuiting then can't build it in the ring, so it is built apart and
  copied in, with a barrier per copy. With `r[s] = tabulate …` followed by
  reads of `r[s, y, x]`, the core kernel had 14 barriers per plane instead of
  5 and 25 KB of shared memory instead of 20 KB. Fix: read through an opaque
  slot number (`let s' = opaque s`), which costs nothing at run time.
- **Updating the only row of a one-row array** (`r[0] = v` with `r: [1][…]`)
  is simplified into replacing the array, even with an opaque index, so a
  loop-carried one-row ring is double-buffered and copied every iteration.
  Use two rows (one unused) or don't carry it.
