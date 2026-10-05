# Futhark issues met by BeamZ's backend

Compiler bugs, patches BeamZ applies to generated code, and code-generation
behaviour that cost performance. All on Futhark 0.27.1. Each bug has a
minimal reproducer, checked with `futhark c --library` and
`futhark cuda --library`. None has been reported upstream yet.

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
**Workaround** (`temporal.fut`, `tinject`): sum per target row into a small
1-D histogram, then read-modify-write the rows with `scatter_3d`, which does
accept `[3]` rows.

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
**Workaround used for a while:** `#[noinline]` on the function holding one
histogram, so that calls cannot fuse. Superseded by the workaround for bug 1.

## Patches applied to generated code (`build.py`)

1. **Results of intra-block kernels go to global memory**
   (`intrablock.py`). Futhark keeps every block-level array of an
   `#[flattening(only_intra)]` kernel in shared memory, the block's results
   included, and copies them out when the block ends. A temporal tile
   streams a whole z column, so its results cannot fit. The patch points each
   runtime-sized result at the block's slice of the global result, disables
   the copy-out and removes its size from the host's shared-memory request.
   About 2.3× on the bare K=2 kernel. Fragile: it pattern-matches generated
   code and asserts every shape. (2026-10-05: the host sums the shared sizes
   in its own order and folds constants, so results are now matched by size,
   not by position.)
   *Wanted upstream:* an attribute to keep a block's result in global
   memory, or doing so automatically when it cannot fit in shared memory.
2. **CUDA primary context** (`_patch_primary_context`). The generated
   context calls `cuCtxCreate`. XLA uses the device's primary context, and
   raw device pointers only alias across the two if Futhark retains the
   primary context (`cuDevicePrimaryCtxRetain`/`Release`).
   *Wanted upstream:* a config option to use the primary context.
3. **ISPC stdlib clash** (`_patch_ispc_stdlib`). The generated ISPC code
   declares `erf`/`erfc(double)` externs that ISPC 1.31's stdlib already
   defines; the patch drops them.

4. **Out of device memory yields a NULL buffer** (`_patch_alloc_failure`).
   In 0.27.1's generated `memblock_alloc_device`, the result of `gpu_alloc`
   is discarded (`(void) gpu_alloc(...)`) and only `ctx->error` is checked.
   `gpu_alloc_actual` returns `FUTHARK_OUT_OF_MEMORY` *without* setting an
   error once the free list is exhausted, so the block is "received" with a
   NULL pointer, the log even says so, and the next kernel writing it faults
   with `CUDA_ERROR_ILLEGAL_ADDRESS` (found at 256×512×512 with
   `unified_memory = 0`; managed memory hides it by oversubscribing). The
   patch also checks the returned status.
   *Wanted upstream:* propagate the status (one-line fix).

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
