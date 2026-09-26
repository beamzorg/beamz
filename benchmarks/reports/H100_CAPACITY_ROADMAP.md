# Capacity follow-up to the eight-H100 study

This draft is stacked on PR #287 (`bench/h100-backend-comparison`, initially
`e97fe05`). It records the diagnosis, memory budget, and implementation gates.
The follow-up now implements direct host-to-shard placement, local field
padding, removal of redundant CUDA material grids, and donating capacity probes.
See [implementation and measured results](H100_MEMORY_CAPACITY.md). Field and
CPML precision remain FP32; shard-local material rasterization remains future work.

## What actually failed

The [capacity study](H100_CAPACITY_STUDY.md) successfully timed 4,607,442,944
physical cells. The 5,754,585,088-cell probe failed before scan compilation:

1. The worker builds the simulation/program inside a CPU default-device context.
2. Outside that context, it prepares/distributes the initial state.
3. `sharding.place_tree(program, program.coefficients)` visits coefficient leaves
   and calls `_place_array`, ultimately `jax.device_put(value, target)`.
4. The recorded JAX stack enters `shard_device_array` and `x._multi_slice`.
5. GPU 0's allocator cannot satisfy another 2,883,718,656-byte (2.69 GiB)
   allocation. No warm timestep is measured for this case.

The allocation size equals one local 896³ FP32 block plus staggered/padded
extent overhead. This supports a temporary placement/resharding explanation.
It does **not** identify the coefficient leaf, prove that every coefficient is
replicated, or distinguish allocator fragmentation from total live demand.
The next implementation needs allocation/residency logging per leaf to settle
those questions. CPU construction by itself does not guarantee that subsequent
device placement avoids intermediate allocations on GPU 0.

The successful 832 case provides independent evidence of the imbalance:

| Measurement | Maximum per GPU |
| --- | ---: |
| Cumulative peak live allocation, reached during preparation | 58.82 GiB |
| Live allocation with final timed output retained | 40.24 GiB |
| Allocator pool | 75.22 GiB |
| Sampled device residency | 77.60 GiB |
| Reported physical capacity | 79.65 GiB |

The allocator pool and device residency are not array payload sizes. Also, the
benchmark deliberately retains its initial state (`donate_state=False`) while
producing an output. Its memory ceiling is not the ceiling of an in-place
production solver. Current observed retained-output storage is approximately
75 bytes per physical cell across the eight devices; this is a measured
snapshot including runtime allocations, not a universal solver constant.

Relevant code: `scripts/benchmark_modal_stepping.py` (setup context and placement),
`beamz/simulation/sharding.py` (`place_tree`, `_place_array`, `prepare_state`),
and `beamz/simulation/compile.py` (material/update coefficient construction).
Compilation currently carries electric update coefficients and permittivity
arrays in `UpdateCoefficients`; uniform/zero arrays may be elided. Dense-array
elimination must be based on actual backend use and alias-aware liveness, not
merely summing all named leaves or deleting referenced inputs blindly.

## The 50-billion-cell budget

A conventional dense 3D Yee state stores Ex, Ey, Ez, Hx, Hy, Hz. At large sizes,
one FP32 copy costs approximately **6 × 4 = 24 bytes per physical cell**.
Staggering, padding, and halos add overhead. In-place leapfrog updates can avoid
a second full copy; they cannot eliminate the six persistent components.

At 50 billion cells, these fields alone need **1.2 TB**, or **150 GB per GPU**
on eight devices. This excludes material data, 12-cell CPML state, communication,
source/monitor arrays, scratch space, and allocator headroom. Consequently,
ordinary dense FP32 50-billion-cell FDTD cannot reside entirely in eight H100
SXM 80 GB GPUs, even after perfect initialization fixes and buffer donation.

[NVIDIA specifies 80 GB per H100 SXM](https://www.nvidia.com/en-eu/data-center/h100/).
For budgeting this particular host, use its recorded 81,559 MiB/device rather
than silently equating marketing GB with GiB. Eight devices reported 684.17
decimal GB; an illustrative 90% usable budget is 615.75 GB. At 50 billion cells
that leaves **12.315 bytes/cell for everything**. The 90% choice is a planning
assumption, not an experimentally validated safe allocator setting.

| Representation/budget | Bytes/cell | Memory at 50B | Arithmetic capacity at 615.75 GB |
| --- | ---: | ---: | ---: |
| FP32 fields alone | 24 | 1,200 GB | 25.66B |
| FP32 fields plus 8 B/cell other storage | 32 | 1,600 GB | 19.24B |
| FP32 fields plus 12 B/cell other storage | 36 | 1,800 GB | 17.10B |
| Previous retained-state snapshot, extrapolated | ~75 | ~3,750 GB | ~8.21B |
| 16-bit fields alone, unvalidated | 12 | 600 GB | 51.31B |
| 16-bit fields plus 2 B/cell material IDs, unvalidated | 14 | 700 GB | 43.98B |

These are memory arithmetic, not achieved capacities or forecasts. The 32/36
byte scenarios assume all remaining overhead fits the stated allowance; no
implementation currently guarantees that. Exact inputs and outputs are in
[H100_CAPACITY_BUDGET.json](H100_CAPACITY_BUDGET.json). Reproduce with
`usable_bytes = 8 * 81559 * 2**20 * 0.9` and
`capacity_cells = usable_bytes / bytes_per_cell`.

Six 16-bit fields almost consume the entire 50B budget; at 90% usable memory,
only 15.75 GB remains globally. Even a two-byte material ID per cell would
exceed the host's full physical memory when added to those fields. FP16/BF16
storage therefore is not a drop-in answer. It changes numerical behavior and
would need long-propagation phase, attenuation, resonance, CPML, and spectral
validation, potentially with scaled or compensated storage. The existing 64 nm
CUDA/JAX raw-DFT discrepancy must be resolved before loosening precision.

## Implementation order and acceptance gates

1. **Measure and remove setup amplification.** Log each coefficient's shape,
   dtype, source devices, target sharding, and per-device allocation deltas.
   For host inputs, slice/build one local shard at a time and transfer directly
   to its destination; avoid a full global GPU-0 staging buffer. Ultimately
   rasterize/materialize by shard to bound host RAM as well. Preserve existing
   arrays already distributed across the mesh. Validate values and sharding on
   small CPU devices, then repeat the failing H100 case with the same precision.
   Acceptance: no global-volume GPU-0 transient; success/failure documented
   independently of the retained-state throughput benchmark.
2. **Bound production working memory.** Add a separately labelled donation/
   in-place capacity harness. Ensure input state, compiled closures, caches,
   and previous outputs do not keep duplicate full-domain arrays alive. Measure
   preparation, first execution, and repeated continuation peaks separately.
   Acceptance: full-state and modal parity, documented ownership/invalidation,
   and measured bytes/cell, with an unchanged comparison benchmark retained.
3. **Compact material storage without reducing field precision.** Identify
   redundant dense permittivity/update arrays; use scalar, separable, or
   material-palette representations when exact for the geometry. Account for
   staggered interfaces, smoothing, loss, anisotropy, and dispersive state.
   CPML must remain boundary-slab storage; monitors must remain aperture-sized.
   Acceptance: exact semantics for the supported subset and explicit fallback
   for other materials. Measure any arithmetic/throughput tradeoff.
4. **Establish a useful full-precision capacity milestone.** First demonstrate
   10–15B cells with realistic boundaries/source/monitors, then evaluate whether
   17–20B fits the measured storage budget. These are investigation targets,
   not promised performance or capacity. Include host construction cost and
   stable continuation, not only successful allocation.
5. **Treat 50B+ as a different storage strategy.** On the same eight H100s,
   investigate validated compression/reduced precision or an out-of-core tiled
   solver backed by host memory. Out-of-core operation can exceed HBM capacity
   but introduces transfer/scheduling costs; previous GCUPS do not predict its
   speed. Simple full-domain read/write streaming of six FP32 fields would move
   2.4 TB per complete step at 50B, before other arrays. Temporal blocking may
   amortize that traffic but complicates halos, CPML, and observations. An
   alternative is more aggregate accelerator memory while preserving FP32.

Merely omitting apparently empty cladding is not a safe capacity optimization:
fields and radiation can propagate there. Adaptive meshes, symmetry, or reduced
dimensionality may reduce the necessary cell count for suitable physical
problems, but change the formulation and must be evaluated separately.

## Review and validation status

The original memory arithmetic uses the recorded device capacity. Implementation,
targeted tests, H100 capacity evidence, and remaining limitations are recorded in
[H100_MEMORY_CAPACITY.md](H100_MEMORY_CAPACITY.md). OpenAI Codex authored the
analysis and implementation. Human review remains pending; the PR is a draft.
