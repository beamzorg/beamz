# Custom CUDA tracing: implemented design and evidence

2026-11-11. Research scope: Python cell physics, CUDA codegen and checked schedules.
**CPML is 12 cells per face at every resolution.** No 10x result is claimed.

## Frontend and semantic graph

[`cuda_stencil_trace.py`](../../../scripts/cuda_stencil_trace.py) traces
operator-overloaded scalar/field proxies once per cell/region signature, rather
than creating Python objects for every physical simulation cell. Supported
arithmetic is deliberately narrow: addition, subtraction, negation, explicitly
rounded multiplication/FMA, and BF16 storage rounding. Dynamic indices,
data-dependent branches/comparisons and unsupported operations fail explicitly.
This is not arbitrary Python/PyTorch tracing and does not depend on JAX tracing.

An evolving read records `(field, relative timestep version, offset_zyx)`.
H/E field IDs are Hx,Hy,Hz,Ex,Ey,Ez = 0..5. A one-step trace reads old H/E at
version 0; E consumes updated H at version 1. Read-only coefficients separately
record scalar, one-axis profile, or dense-cell layout and FP32 type. They do not
acquire evolving-state versions or accidentally enter field dependency closure.

The deterministic DAG uses exact structural common-subexpression elimination:
operand order, field versions, affine offsets, coefficient layouts, signed zero
and precision conversions remain significant. It never algebraically
reassociates a curl or merges distinct field versions. Constant values are
rounded to their declared FP32 representation before serialization. Streamed
update generators are retained while identity memoization runs, preventing
Python object-ID recycling from aliasing unrelated graph nodes; a regression
test reproduces the previously found failure mode.

`dependency_cone` computes the bare Yee graph's structural initial-state
footprint through multiple steps, without inlining the full multistep arithmetic.
`emit_cuda_block` emits scalar SSA, and `cell_liveness` reports last uses and
conservative live scalar counts. These are cell-DAG analyses, **not** automatic
CTA scheduling, CUDA register allocation, or an achieved-occupancy estimate.

## Numerical contracts and CPML

The semantic reference remains
[`yee_primitives.cuh`](../../../cuda/src/yee_primitives.cuh). Derivative
products round before curl subtraction, and the field update is
`fma(±source, curl, mul_rn(decay, old))`. The emitter uses explicit CUDA RN/FMA
intrinsics; changes in staging must preserve those rounding points.

The frontend also traces CPML face/edge/corner region equations and private
auxiliary recurrences. Stored BF16 psi rounds after every substep, while the
current field correction uses the **unrounded FP32 next psi**, matching the
native engine. Profiles are typed read-only axis values. Active-axis selection
describes a region already admitted by the caller; physical slab packing,
staggered-boundary admission and global ownership are not inferred by this
local arithmetic trace.

## Actual production integration boundary

[`generate_cuda_yee_cell.py`](../../../scripts/generate_cuda_yee_cell.py)
deterministically produces
[`yee_cell_generated.cuh`](../../../cuda/src/yee_cell_generated.cuh), including
a semantic graph hash and a `--check` mode. Its private helper is:

```cpp
AdvanceYeeCell<Phase, Component>(inv_spacing, read, coefficient)
// read(fieldID, relative_version, dz, dy, dx)
// coefficient(kind, component); kind: decay=0, source=1
```

The native packed core consumes this arithmetic through the
`BEAMZ_CUDA_MSTEP_TRACED_CELL` build option, selected for the validated default
build after matching manual arithmetic performance. The enclosing native
stage still proves the field version and chooses global/shared/register loads,
coefficient decoding and output ownership. The generated header alone does not
select a faster schedule. This production lowering admits the fixed Yee read
and coefficient signature; cross-component own-family reads or material coupling
require a new lowering/admission proof and parity gates. A host-compiled
independent oracle compares all six
cell outputs bitwise over 6,144 randomized cases with distinct versions and
nonunit/physical derivative scaling.

## Generated temporal schedules and measured scope

[`cuda_stencil_rolling.py`](../../../scripts/cuda_stencil_rolling.py) generates
the independently checked rolling-plane schedule template. The newer
[`cuda_stencil_register.py`](../../../scripts/cuda_stencil_register.py)
uses fixed physical-XY lane ownership and one shared plane per intermediate
stage. H needs prior-plane E transverse values, so E publication is delayed
until that H stage finishes reading. Current H is then published for E's
transverse reads. Short-lived same-XY caches supply z-neighbor and old-own-field
values; the best tested mode removes persistent per-stage register histories.

This halves the generated probe's intermediate shared allocation. Production
D2 retains its two-plane input-prefetch bank, so its total shared reduction is
about 35–36%, rather than 50%. Generated schedules still use frozen global
inputs, distinct output banks, and duplicated halos. They are hand-derived
schedule templates parameterized by traced equations, not a general automatic
scheduler for arbitrary dependency graphs.

At the unchanged RTX 3090 250 W limit, a controlled 17-step periodic probe on
256 cubed equal-extent cells measured 20.917 GCUPS for register depth two versus
14.734 for the same-tile two-shared-plane rolling control (owned tile z,y,x =
32,8,16). One-slot depth-four tiles measured roughly 17.1–17.5 versus 13.1–13.2
GCUPS; the larger two-slot tile used 113 registers and was less competitive.
All twelve tile/layout/size runs matched the independent scalar C++ reference
bitwise, and all six small memcheck/racecheck/synccheck runs passed.

These are **synthetic codegen measurements**, with periodic common supports,
uniform normalized coefficients, no CPML/source/monitor workload, and
host-driven submission inside the CUDA event interval. The harness reports
the median of nine event samples after two warmups; it does not preserve
individual samples or establish confidence intervals. They cannot be compared
directly with complete BeamZ GCUPS or presented as an application speedup.

Raw manifests, generated sources, binary hashes, build/run logs, paired summary,
GPU telemetry and sanitizer reports are archived under
`.cache/cuda-autoresearch-2026-11-11/tracing-probes/large-matrix/`; the local CPML
probe is under `tracing-probes/cpml-cell/`. Manifests retain original execution
paths under `/tmp`. The local CPML GPU probe checks all eight active-axis masks
with FP32/BF16 recurrence against independent C++ FMA equations, bitwise.

## Arithmetic evidence versus schedule evidence

The generated-header and CPML numerical probes validate arithmetic separately
from the integer/token proofs. Those proofs check exact field versions,
coordinates, ring bounds, unique final ownership, partial masks, real staggered
supports, discarded clamped halos and short column segments. Negative controls
remove temporal reads/publication or prefetch too early and must fail.

The full-bank first-intermediate proposal has a separate taint oracle:
H1/E1 run on the unchanged first queue S, while second-step edges Q read full
banks. E1 in discarded S halos may be invalid. The oracle infers required
E1 from Q's H2 positive-neighbor stencils, Q's own E2 values and DFT requests,
then proves its H1 negative-neighbor closure lies in S. It permits taint only
outside that region and inactive auxiliary state, while checking CPML clocks,
source/observation chronology, nonzero origins, odd tails and split calls.
This does not prove floating-point equivalence or external input immutability;
full-state GPU tests and retained-input/repeated-call checks remain required.

## Reproduce

Use a CUDA-enabled interpreter: `/home/quentinwach/Code/.venv-cuda-pr233/bin/python` here.
Local nvcc: `/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/bin/nvcc`.

```bash
python scripts/generate_cuda_yee_cell.py --check
python scripts/cuda_stencil_trace.py --depth 4 --output /tmp/cell-trace.json
python -m pytest tests/unit/test_cuda_stencil_*.py
python scripts/probe_cuda_stencil_register.py --output-dir /tmp/register-probe \
  --tile 32 8 16 --shape 256 256 256 --steps 17 --samples 9 \
  --e-history shared --h-history shared --nvcc /path/to/nvcc --run
python scripts/cuda_stencil_cpml_probe.py --output-dir /tmp/cpml-probe \
  --nvcc /path/to/nvcc --run
BEAMZ_CUDA_CPML_TEMPORAL=1 python scripts/benchmark_cuda_realistic.py --backend cuda_streamed --pml 12 --output /tmp/complete-cpml.json
python scripts/build_cuda_candidate.py --output-dir /tmp/traced-native-new \
  --python /path/to/cuda/python --nvcc /path/to/nvcc \
  --option BEAMZ_CUDA_MSTEP_TRACED_CELL=ON
```

The candidate builder freezes source/hashes without replacing the installed component. Serialize GPU benchmark jobs.

## Build on the existing complete temporal CPML path

[`temporal_cpml.cuh`](../../../cuda/src/temporal_cpml.cuh) already implements
complete two-step CPML behind `BEAMZ_CUDA_CPML_TEMPORAL=1`, separately from
mstep. It keeps initial fields/psi frozen, stages H1/E1/H2 and intermediate
psi in two-plane rings, fuses source injection, partitions face/edge/corner
ownership by active-axis masks, and publishes final fields/psi only from their
owners. Three field banks, two psi banks and paired DFT processing preserve
continuation/observation semantics. BF16 storage is supported already.

The [realistic integration study](../cuda-realistic-temporal-integration-2026-09-18.md)
predates complete CPML fusion and describes the separate unblocked absorber.
The later [CPML development record](../cuda-temporal-cpml-development-2026-09-18.md)
implements private auxiliary stages and records a still-slower, numerically
validated complete pair. Its historical timings are not today's 250 W baseline.

Code-based hypotheses to test, rather than established stall diagnoses:

- Overlapping halos and short z tiles add work: the unmasked 16x8x16 D2 field
  envelope is about 1.45 times useful phase updates; a 16x8x4 edge tile is about
  1.87, before bounds clipping. Longer segments/other ownership need measurement.
- Two-plane field and psi rings consume shared storage. Fixed-XY lanes could
  retain own-cell psi history in registers, since psi has no spatial-neighbor
  recurrence; measure register pressure/spills alongside any shared reduction.
- Packed-coordinate/profile/boundary work remains per cell. Active-axis and
  oriented tiling already exist; profile before hoisting addresses/profiles.
  The historical spatial-only control also lost, so auxiliary storage alone
  does not explain the gap. Preserve the existing controls instead of rebuilding
  their functionality under another name.

## Next architectural work

Large-workload profiling identified about 49% of summed GPU activity in edge
kernels, including CPML/scratch work; see the [baseline study](cpml12-baseline.md).
This is event-duration attribution, not unprofiled wall time. Optimizing only
the core cannot approach a 10x complete-solve gain while that work persists.

First profile the existing complete pair on matched CPML12 cases against today's
best verified mstep/streamed schedules. Reuse its source, BF16, odd-tail and DFT
gates. Existing psi offsets/source-halo assumptions are two-step-specific.
Generated D4/D8 remains a hypothesis: measure halo work, liveness, memory headroom,
parallelism, extra banks and startup before admission. Preserve frozen
halo inputs, private intermediate psi, unique final publication, per-substep
BF16 rounding and exact source/DFT times. Chronological observations need explicit
ownership or buffered samples/barriers. Skewed/non-overlap schedules require
read/write proofs; persistent queues also need residency-safe ready work and
device publication ordering, without CTAs spinning on unscheduled producers.

## Concrete workspace follow-up contract

The accepted full-bank schedule does not access compact scratch, but this run
retains its allocation to isolate scheduling changes. The next memory change
should bind workspace admission to the actual registered native implementation,
not infer capabilities from a nominal backend name or unchanged ABI number.

Prefer an additive optional FFI target that promises full-bank intermediates;
retain the original target and signature. Register the optional target only in
a full-bank component. New Python can select it and allocate six 128-value
placeholder buffers; without that target it must retain complete compact
scratch. This preserves all four combinations of old/new Python and component:
old callers still allocate sufficient memory, and new callers cannot shrink
scratch for an old component. Freeze the selected target in executable/cache
identity. The capability is a proposed contract, not implemented in this pass.

Keep the geometry/slot plan unchanged initially. Report actual allocated scratch
separately from its theoretical compact footprint, and use actual allocation
in memory admission. Test both compatibility paths, retained/donated inputs,
odd continuation, changing clocks, dense/packed materials and allocator peaks
before any capacity claim. The benchmark's equal sampled peaks reflect common
preparation/driver high-water marks; they do not measure the minimum live field
workspace or establish that a 1 nm case fits.
