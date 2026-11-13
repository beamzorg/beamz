# Moving-cache field-bank alias proof

The current owned moving-cache schedule can use corresponding input/final field
storage in place, subject to the assumptions below. This is a CPU legality proof;
the current native ABI rejects aliases and no allocation or CUDA code was changed.

For one physical coordinate `p`, owned tile extent `B`, and positive temporal
depth `T`, a bootstrap reader tile has index
`C ∈ {floor(p/B), floor((p+1)/B)}` because its input envelope is
`[C*B−1, C*B+B−1]`. The final writer has index `W=floor((p+T)/B)` because
its output envelope is `[W*B−T, W*B+B−T−1]`. Hence `C≤W` on every axis.
Distinct reader/writer tiles therefore have strictly increasing rank
`rank=z+y+x`; equal rank implies the same tile. Earlier ranks finish before
the writer starts. Same-tile startup reads finish before its final scatter.
Consequently a final write cannot destroy any pending bootstrap input read,
even when distinct CTAs of the same rank execute in an arbitrary interleaving.
Clipping either envelope to any of the six actual staggered supports preserves
this separation. Final envelopes also partition each field support uniquely.

The proof uses the canonical versioned Yee operator for initial-read offsets and
the actual canonical support deltas. Source binding was reviewed against
`source-plane-axis-cg-v1/source/src/asymmetric_native.cuh`, SHA
`f082980ad0687ef165c7807b415e87cbc1003edfdbd2acf7a4ee018dae4d4547`:

- `Initial` is called only by startup branches, with `first==1` in windowed paths.
- Late-window startup imports positive-version trails, never the input fields.
- Positive-version field providers use registers, shared frames, or trails.
- Final field stores occur at `n==Depth`.
- Current per-rank stream/event joins retire every preceding rank. A relaxed
  cohort DAG would require its own equivalent field read/write hazard proof.

The executable proof covers 28 rectangular/ragged/cube shape, geometry, and depth
cases; all six staggered fields; depths 4/16; original owned `7×15×31` and cube
owned `15³`. Factorized enumeration checks 17,183,861,022 three-dimensional
reader/writer links without allocating whole grids. Eight seeded token schedules
also scatter each tile before the next same-rank tile bootstraps. The canonical
IR contributes 480 initial-read corner checks. Reverse final shift, unshifted
version-zero writes, output pre-clear/upload, and reverse rank order all fail
with concrete overwritten nonzero input tokens.

Reproduce from the repository root:

```sh
python scripts/prove_cuda_field_bank_alias.py \
  --source .cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source \
  --output .cache/cuda-two-hour-2026-11-13-0819/field-bank-alias-cpu-v1/proof.json
/home/quentinwach/.cache/beamz-pr233-ci/bin/python -m pytest -q tests/unit/test_cuda_field_bank_alias.py
```

The result concerns six corresponding field banks with identical dense shapes
and strides. Different components remain disjoint; so do fields and material,
source, profile, ψ, trail, clock, and raw observation buffers. It does not prove
those side effects correct, change their ownership, or establish alias legality
for the ordinary two-step Mstep kernel. Observations must continue to consume
the computed values at their existing times rather than reread overwritten input.

API implementation still needs separate admission, graph-key/output mapping,
reset/preserve snapshots, allocation ownership, and tail handling. In particular,
the constructor currently uploads the spare output bank after the input bank:
aliasing those allocations without suppressing that upload/clear would corrupt
the input before computation. A zero-start benchmark could conceal this defect.
The proof's seeded pre-clear negative catches it. The input/output pointer fact
must also match actual corresponding component layouts and exclude accidental
cross-component aliases. This is an enabler for a reviewed implementation, not
an implemented optimization or performance measurement.

Removing one six-component bank would save exactly 3,230,668,800 bytes at `512³`,
4,317,205,680 bytes at `564³`, or 6,306,209,280 bytes at `640³`, including actual
staggered terminal nodes. These are storage calculations, not admitted workload
sizes or measured GPU allocations.
