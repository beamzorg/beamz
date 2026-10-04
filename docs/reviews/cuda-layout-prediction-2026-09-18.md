# CUDA layout prediction without calibration — 2026-09-18

Default automatic selection now uses a deterministic geometry estimate. No GPU
performance benchmarks were run for this change: all performance conclusions
below come from saved measurements. Explicit `BEAMZ_CUDA_AUTOTUNE=calibrate`
retains the previous measured selector; `auto` no longer measures or reads its
profiles. FP32, the physical domain, 12-cell CPML and sampling remain unchanged.

## Pattern and its physical basis

The strongest repeatable effect is a short contiguous x dimension. In the
same-physics cyclic study, moving physical z into storage x improves
1024×256×63/64/65 by 22.5–24.8%, including all layout conversions. The developed
smooth 1024×256×64 case improves by 24.8%. The CPML shell fraction and total
logical volume are identical before and after rotation, ruling out boundary
volume alone as the explanation. These cases also rule out an exact power-of-two
alignment requirement. See the [cyclic study](cuda-cyclic-storage-study-2026-09-18.md).

The ordinary queue launches 64×4 threads per block. Its x-face blocks pack two
12-cell boundary slabs, plus the staggered extent: 25 columns occupy a 64-column
tile. This leaves many inactive lanes; their contribution grows as x gets
shorter. Yee staggering adds another tail: nominal x=64 includes rows of width
65. A long contiguous dimension reduces these geometric penalties. This is an
explanation supported by kernel structure, not a hardware-counter attribution
of the observed runtime to one particular stall.

Neither “always put the longest axis last” nor “minimize inactive threads” is
enough to select every optimum. The earlier physical-shape sweep had
1024×64×256 at 8.31 GCUPS and 64×256×1024 at 8.14, despite the latter's lower
queue-work estimate. Swapping y/z at x=256 also substantially changed performance.
Those are different physical domains with the source still along x, so they are
supporting evidence of model limitations, not same-physics rotation comparisons.
See the [shape study](cuda-shape-study-2026-09-18.md).

## Deterministic rule

For stored dimensions `(z,y,x)`, let `iz=z-24`, `iy=y-24`, `ix=x-24`, and
`C(n,t)=ceil(n/t)`. For either Yee triplet the maximum extent in each direction
is `n+1`, and the minimum is `n`. The launch counts in
`cuda/src/update.cu:LaunchCpmlQueueTile` give:

```
z_face_blocks = C(x+1,64) * C(y+1,4) * 25
y_face_blocks = C(x+1,64) * C(25,4)  * iz
x_face_blocks = C(25,64)  * C(iy,4)  * iz
core_blocks   = C(ix,64)  * C(iy,4)  * iz
work = 256 * (z_face_blocks + y_face_blocks + x_face_blocks + core_blocks)
```

Evaluate this for the three right-handed cyclic orders. Change from canonical
012 only when the minimum work is at least **25% lower**. This is a conservative
policy margin informed by the existing studies, not a predicted 25% speedup or
a statistical confidence bound. Do not enable padding or CPML temporal pairing.
Retain shell 64×4: 32×8 helps unrotated short-x domains by about 5%, but provides
no established benefit after the larger layout problem is corrected.

| Physical shape (z,y,x) | 012 slots/cell | 120 slots/cell | 201 slots/cell | Selection |
|---|---:|---:|---:|---|
| 1024×256×64 | 2.033 | 1.133 | 1.274 | 120 / 64×4 |
| 128×256×512 | 1.152 | 1.518 | 1.292 | 012 / 64×4 |
| 257×193×341 | 1.169 | 1.275 | 1.365 | 012 / 64×4 |

The estimate is cheap host integer arithmetic and works for unseen dimensions;
it is not an exact-shape lookup. Native fusion remains governed by the existing
dispatch rule. Its different core work, cache/coalescing behavior, transpose
cost and source/monitor costs are not modeled. The margin deliberately leaves
some possible gains unexplored, including 64×1024×256, where the geometric
reduction is too small to justify automatic rotation under this rule.

The runtime gate remains: one visible RTX3090, ≥8,388,608 cells, ≥32 steps,
uniform 3D lossless diagonal materials, FP32 default flags, six-face CPML12,
slab sources and native-compatible DFT monitors. Other workloads and explicit
manual CUDA overrides retain existing dispatch. The geometric function itself
can be inspected without a GPU:

```python
from beamz.simulation.cuda.tuning import predict_layout
print(predict_layout((1024, 256, 64)))
```

## Retrospective check against saved timings

The [replay artifact](cuda-layout-prediction-replay-2026-09-18.json) covers 13
existing profiles and **five distinct domain shapes**, including widths 63/64/65,
wide and irregular controls, binary/smooth material, developed fields and a
1,025-step request profiled over one native chunk. Repeated profiles are not
independent shape evidence. This is retrospective consistency checking using
the same evidence that informed the rule, not held-out validation.

The narrow profiles have a predicted work reduction of 40.7–44.3%; the wide and
irregular controls already minimize the proxy in canonical order. Changing the
policy margin from 25% to anywhere in 20–35% therefore leaves all these choices
unchanged. This sensitivity check does not establish the best cutoff for unseen
shapes.

The selected layout is within **0.45% of the fastest measured candidate** in
all 13 profiles. The small misses are the wide control, where differences are
below the old tuner's 3% threshold. No profile predicts a slower choice than
canonical; narrow cases select the recorded 22.5–24.8% improvement over the
branch canonical layout. CPML-pair variants are excluded because they were
slower; some profiles contain only the three ordinary storage orders, while
the newer profiles contain all six storage/shell combinations.

The rule selects the same configurations used in the latest saved main/branch
comparison: **7.283→8.555**, **8.386→8.772**, and **6.965→8.277 GCUPS** for the
narrow, wide and irregular controls, respectively. These are historical warm
execution measurements, not new predictor benchmarks. The 33–37 seconds of
first-use calibration are removed; normal simulation setup and compilation
still take time. See the [original comparison](cuda-autotuning-2026-09-18.md).

Reproduce the offline analysis without GPU execution:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python scripts/analyze_cuda_layout_prediction.py
```

Shape alone cannot guarantee an improvement over origin/main for every arbitrary
simulation. Large monitor loads, different source placement, short chunks and
unmeasured aspect ratios can alter the best choice. This change estimates layout
from the evidence available; it does not establish consistent ≥9 GCUPS or repair
the separately recorded CPML numerical differences from origin/main.

## Implementation validation

- 152 CPU tests passed across selector behavior, backend/cache contracts, CUDA
  runtime contracts and architecture checks. Tests make calibration, telemetry
  and persistent-profile access fail if the default selector attempts them.
- Two GPU correctness tests passed, with seeded fields, CPML auxiliary state and
  DFT accumulators; one mode source and two mode monitors; both cyclic rotations;
  default/GPU and CPU setup; 33 and 519 steps, including chunking and an odd tail.
  Every final leaf matched canonical execution bitwise and caller state stayed
  unchanged. These are deliberately small correctness tests, not throughput
  measurements; the large-domain eligibility gate is bypassed only in the tests.
- Ruff and `git diff --check` passed. The native CUDA binary was not rebuilt.
  No new performance benchmark or calibration trial was run, and nothing was
  pushed.

## Subsequent prospective check

A [four-shape random-domain comparison](cuda-random-final-comparison-2026-09-18.md)
then tested previously unmeasured large shapes with the predictor frozen.
The branch beat current origin/main by 6.6–19.1% in all four cases without
calibration, reaching 7.09–8.73 GCUPS. This compares revisions, not all candidate
layouts, so it does not extend the retrospective claim of near-optimal selection.
Fields and monitors passed tolerance; CPML state had failures in three cases.
