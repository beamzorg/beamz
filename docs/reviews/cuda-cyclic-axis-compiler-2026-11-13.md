# Cyclic coordinate specialization: CPU feasibility

The ordered Yee/CPML cell graph is exactly equivariant under the three proper
cyclic XYZ permutations. This supports researching an internal tensor layout
that makes the existing world-X source plane normal to cache Z. It is not an
implemented backend, a rotated benchmark result, or a speed prediction.

## What was proved

`scripts/prove_cuda_cyclic_axis_equivariance.py` relabels actual typed field
reads, versions, constant offsets, component coefficients, mutable CPML states
and independent profiles. It retains every operation and ordered operand of
`sub`, `mul_rn`, `fma_rn`, sign multiplication and BF16 storage rounding. It does
not replace them by algebraically equivalent expressions.

The transformed graph matches a separately retraced canonical operator for all
3 cycles × 8 active-axis regions × FP32/BF16 local storage: **48 ordered CPML
graphs**, plus 3 plain Yee graphs. Another 1,914 checks bind canonical staggered
supports and the phase-specific CPML active/packed-coordinate formulas. Unknown
axis-bearing symbols fail closed. A signless odd permutation, unchanged profile
or state labels, and reversed subtraction operands have counterexamples.

Output definitions are compared by their tensor names; their expression
operands remain ordered. Permuting independent component definitions additionally
requires distinct mutable states and no cross-component source side effects.
The actual cell trace has only its own old same-family component and opposite
family stencil reads. Native inputs already allocate separate component/state
buffers; that admission must remain explicit in a future lowering.

The graph proof covers homogeneous active-axis cell operators. Native support
checks, PEC/boundary bits, physical packed storage, source injection and monitor
execution are separate integration obligations. Proper permutations exchange
axis labels without reversing low/high sides. The existing scalar Cartesian
spacing is unchanged; this pass does not admit an unlabelled anisotropic metric.

## A useful proper cycle

Use old XYZ → new XYZ `(Z,X,Y)`, equivalently:

| World quantity | Internal quantity |
|---|---|
| ZYX tensor coordinates | `(worldX,worldZ,worldY)` |
| Storage transpose axes | `(2,0,1)` |
| World Hz / Ey | Internal Hy / Ex |
| Post-source presence mask 20 | Mask 10 |
| Source groups 5 / 7 | Groups 4 / 6 |
| X-normal source slab `[count,Nz−24,Ny−24,1]` | Z-normal `[count,1,Nz−24,Ny−24]` |

All source IDs within a group, waveform/time rows, H/E phases and clocks must
retain their order. The typed source-address oracle checks 108 batches and
40,986 point/clock cases, including noncoincident and clipped slabs. It preserves
the sequence of coefficient identities and source time indices after inverse
mapping. It does not execute a CUDA source kernel.

The existing fast source path admits mask 20. Simply changing backend axes would
therefore select a different fallback for mask 10. A useful experiment needs an
explicit generated/admitted rotated source class; changing only the input
orientation is not an equivalent engine comparison.

## Cohort geometry is a second motivation

The actual host planner marks a tile for a unit-thickness source at coordinate
q when its normal origin satisfies `q−B+2 ≤ origin ≤ q+T`. Its complete owned
trajectory spans `B+T−1` positions. At depth16 this width is 46 for the current
normal X ownership31, versus 22 for normal Z ownership7.

The actual planner was compiled against the frozen v12 headers with the full
plane shapes and correctly remapped source groups/starts. It gives:

| Domain / source planes | Normal-X origins | Normal-Z origins | Source-tagged tiles X → Z | Valid component updates in tagged tiles X → Z |
|---|---|---|---:|---:|
| 512³, q104/105 | 93 | 105,112,119 | 2555 → 1785 | 773,147,657 → 528,869,943 |
| 564³, q115/116 | 93,124 | 112,119,126 | 6156 → 2166 | 1,874,485,246 → 638,115,828 |

These are **31.6% and 66.0% fewer valid field-component updates assigned to
source-tagged CTAs**, not fewer physical FDTD updates: total valid updates are
identical. Combined clean source Policy3/4 tile counts change 2077→1395 and
5100→1632; source+CPML/boundary Policy5 changes 478→390 and1056→534. Observers were
omitted to isolate source geometry; adding observation flags splits Policy3/4
without changing those combined source counts.

The normal owned-width ratios21/31 and21/62 describe the same alignment effect.
They are not performance ratios. The figures exclude material classification,
instruction issue, cache sectors, field traffic and backend fallback costs.
Z-normal sources could also improve active-lane utilization and coefficient
coalescing, but that remains a hardware hypothesis.

## World-order observations are mandatory

Keep the original world monitor/component labels, gather records, FP32 weights,
neighbor reduction order, substep chronology and phase recurrence boundaries.
Only each record's field address changes. Sorted internal publication keys may
carry the **original world raw ID** as their lookup payload. The existing raw
packet and DFT reduction can then retain world ordering.

Sorting the reduction by internal keys is unsafe. Four valid bilinear neighbor
weights of 1/4 and field values `[4e8,−4e8,13,0]` produce 3.25 in world neighbor
order. The internal key sort exchanges the middle neighbors and produces 0.
This counterexample uses exact power-of-two products; it applies whether the
ordered multiply/add is contracted into FMA or kept separate. The current
`io.cu` neighbor loops deliberately preserve record order.

The world raw key prefix also includes component ID and staggered array shape.
It cannot be reused as an internal linear offset. Decode the world key, map
field/coordinates, and build an internal-key → original-raw-ID lookup. Preserve
valid zero-weight records too. Remap observer ownership flags from these actual
internal coordinates rather than treating monitor faces as a symmetric proxy.

## Concrete integration obligations

* Transpose already decoded Yee material coefficients and masks exactly, then
  losslessly repack. Recomputing material averages or interpolation weights
  with a different axis operation order could change rounding.
* Rename and transpose every packed psi slab and each independent phase/term
  profile; the differentiated target extent remains `Naxis+phase`, with 12 CPML
  cells per side. Do not merge profiles or H/E storage.
* Preserve source order and timing, derive mask 10 admission, and map world raw
  observation IDs as above. Retain world outputs through inverse mappings for
  all 30 leaves, including seeded psi and clocks.
* Admit a cyclic internal layout as a typed program/storage fact, with hashes
  binding world inputs and the mapping. Keep it fixed through the solve.
* Report setup transpose/repack memory and latency, final conversion cost and
  actual solve scope. Avoid full extra-domain temporaries on memory-limited
  large cases. A physically rotated scene without inverse world outputs is not
  this experiment.

The strongest next compiler step is a typed coordinate-layout pass that carries
field/state/coefficient/source and observation mappings together, with the
ordered graph certificate as admission. No backend implementation is proposed
for the current frozen timing campaign.

## Reproduction and evidence

```sh
python scripts/prove_cuda_cyclic_axis_equivariance.py --output /tmp/cyclic-proof.json
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu /home/quentinwach/.cache/beamz-pr233-ci/bin/python -m pytest -q tests/unit/test_cuda_cyclic_axis_equivariance.py
```

Local evidence is under
`.cache/cuda-two-hour-2026-11-13-0819/cyclic-axis-cpu-v1/`: the ordered graph proof,
actual host source-cohort C++/JSON and its exact compiler/source hashes.
