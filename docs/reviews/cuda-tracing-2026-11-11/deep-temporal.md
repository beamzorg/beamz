# Native four-step tracing experiment, 11 November 2026

This is an opt-in research implementation. `BEAMZ_CUDA_DEEP_D4` remains OFF.
The default production D2 engine is the control. Complete-physics correctness
is established for frozen v6; v6 is substantially slower, so it is not promoted.
The purpose is to test communication schedules against the real engine rather
than infer application speed from a periodic stencil.

## Implemented schedule and lowering

`scripts/cuda_deep_temporal_warp.py` consumes the existing typed, versioned,
offset-aware Yee/CPML trace. Its arithmetic emitter preserves explicit rounded
multiplication/FMA and CPML recurrence order. It lowers canonical native wall
derivatives separately, preserving staggered supports and PEC behavior.

One warp owns a fixed physical X row. X-neighbor derivatives use full-mask warp
shuffles outside lane-varying validity branches. Y-neighbor derivatives use
shared planes. The Y field component has no Y-neighbor reads in the Yee curl,
so its previous-plane values remain in per-stage registers. Current/old values
are fresh inside each rolling Z wave. Face CPML histories remain in registers;
edge/corner histories use private, same-thread shared slots. Halo computations
keep private intermediate auxiliary state; only uniquely owned final state is
published to the next global bank. Initial and final packed psi banks differ.

The kernel consumes real native `PairCpmlPhase`, field/source descriptors and
`BeamzProgramLaunch`; it introduces no second physics format. It handles real
staggered shapes, ragged ownership, sources, CPML12 and all six field components.
Generic admission rejects unsupported layouts, pre-E sources and aliased banks.
The current integrated hook requires FP32 fields/psi and scalar H coefficients.

## Observations and clock semantics

Each uniquely observed component/cell is staged once per substep. A dense
six-bit publication mask gates key lookup; unobserved bulk cells avoid binary
search. Staging stores raw values, including records having zero gather weight.
The original gather order/weights are retained through remapped indices.
Existing, unchanged DFT kernels consume four staged tables chronologically.
This preserves weighted FMA order and avoids a replacement DFT implementation.

Source/monitor clocks stay fixed within a native chunk; macro-step offset plus
substep is passed to existing source/DFT helpers. The standard 256-step chunk
boundary remains the phase-cache reset boundary. Tails of one to three steps
use the existing native solver at the correct offset. NativeSession handles
field/psi bank parity, continuation, reset and preservation of caller inputs.

## Frozen candidates and exact reproduction

The checked-in header is **v12 host-admission cleanup of v8 single-frame kernels**, not the
generic default generator. The query reports generic or typed admission before
NativeSession allocates/captures, giving unsupported inputs a useful error.
Regenerate it from the repository root with:

```sh
python scripts/cuda_deep_temporal_warp.py --bundle --typed-mie \
  --output cuda/src/deep_temporal_warp_generated.cuh
```

Its SHA256 is
`a3ec109b0f04f10711afdff31f100a83e4ccef5f6b014b7d03241815a6a6ba21`.
`tests/unit/test_cuda_stencil_deep_generated.py` verifies the explicit flags
reproduce that file and verifies the generic control remains reproducible.

The generator also preserves these controls:

| Candidate | Generator bundle flags | Header SHA256 prefix |
|---|---|---|
| v6 generic | `--bundle` | `47110f5a9914fafd` |
| v8 typed | `--bundle --typed-mie` | `bf501896d0f68f04` |
| v10 two frames, before admission query | `--bundle --typed-mie --double-frames` | `687564a41077007d` |
| v11 same v10 kernels, admission query | same flags | `558b21a2bee07ea99` |
| v12 selected v8 kernels, admission query | `--bundle --typed-mie` | `a3ec109b0f04f107` |

The default bundle now includes a generic admission-kind query. Removing that
single host-only line reproduces the historical v6 SHA exactly; the test checks
this explicitly. Frozen v8/v10 sources retain their original query-free form.

Immutable native sources, compiler logs, binaries and manifests are under
`.cache/cuda-native-autoresearch-2026-11-11-1607/native-v6-deep`,
`native-v8-typed` and `native-v10-double-frames`. They are separate modules;
the installed v13 production component is unchanged by these probes.

Typed v8/v10/v11/v12 admission requires exactly the Hz+Ey source-component signature
(bitmask 20), scalar FP32 H coefficients, FP32 E codebooks and rank-one byte
packed material codes. Compressed rank-two material storage is rejected.
Positive E logical volumes must fit INT_MAX and packed storage must contain
enough words. The code uses existing typed material lookup with int32 logical
indices; field bounds, PEC checks and rounded arithmetic remain intact.
This is a workload-admitted lowering, not an arbitrary-physics frontend.
The rectangular/mode-source fixtures use Hy+Hz+Ey+Ez (signature 54), so they are
intentionally rejected by this typed candidate; generic v6 handles them.

V10 doubles shared field frames for the bulk and single-axis CPML faces only.
Old/new frames alternate by wave, so previous-plane readers cannot collide with
current-plane writers. The pre-overwrite barrier is removed; each H stage still
ends with its publication barrier. Edge/corner kernels retain the v8 schedule.
Core/face shared field storage rises from 21,504 to 43,008 bytes. Both remain one
CTA per SM in the measured register regime, so this trades spare shared memory
for four rather than eight barriers per wave.

## Evidence and limits

Frozen v6 matches all 30 canonical state leaves with exact finite array equality on seeded Mie136³,
ragged 64×96×128, odd continuations 3+3+1 and 1+1+5, 259 steps with a 256-step chunk,
127+132 continuation, realistic256³ and Mie272³. This includes fields, 12 packed
CPML states, six-component DFTs and clocks. Repeat/reset, input immutability and
the absence of JAX in the native runner were checked. Evidence is in
`.cache/cuda-native-autoresearch-2026-11-11-1607/deep-v6-results/summary.json`.
The same v6 kernels with v7 host cleanup passed native-only memcheck, synccheck
and racecheck on seeded 64×96×128/7, with zero errors/warnings. Root's
`deep-v7-*` sanitizer logs are separate from uninstrumented timings. The selected
v12 single-frame Mie136³/7 gate also passes memcheck, synccheck and racecheck
with zero errors/warnings. V10 changes
barriers and therefore requires fresh sanitizer coverage before promotion.

The bounded same-v6-module rejection screen measured realistic256³ D2=0.4387 s
versus D4=1.9662 s, and Mie272³ D2=0.6045 s versus D4=2.6176 s. These are rejection
screens, not statistical speedup estimates. D4 loses roughly 4.3–4.5×. Root's
Mie profile attributes 1.2875 s to bulk mask0 alone, larger than the entire D2
solve. Corner kernels account for only 2.2%; corner spills are not the main
large-case explanation. Balanced six-process Mie272 tests show v8 medians 2.16993/2.16817 s versus
v6 2.60428/2.61259s and v10 2.53826/2.53492s. V8 is 1.2026× faster than v6;
v10 is 1.169× slower than v8. All 30 leaves pass exact finite array equality. All remain over 3×
slower than D2, so v12 selects the v8 single-frame geometry for research only.
Evidence is in `deep-balanced-results/summary.json` under the native cache root.

Exact launch-geometry accounting gives executed/useful field-component updates
2.65484× at 136³, 2.05283× at 256³ and 2.08901× at 272³. The 1.77246× ideal full-tile
estimate hides ragged/CPML-region costs. Counts exclude CPML expressions, source
loops, publication and DFT work. Reproduce with
`python scripts/analyze_cuda_deep_work.py FIXTURE.json --output work.json`.

One-core compiler diagnostics on SM86 use nvcc 13.3, `-O3 -DNDEBUG -lineinfo`,
relocatable device code and the real frozen update.cu. Generic source63 uses
91 registers/no spills and 9,552 static SASS instructions. Source20 alone and
scalar H alone retain 91 registers. Combined source20/scalar-H/typed-byte lowering
retains 91 registers but reduces static SASS to 6,184 instructions. Two frames
use 96 registers/no spills. Static code size is not dynamic instruction count.
Actual v10 native face kernels retain 128 registers but acquire 144–168 bytes of
stack, whereas v8 faces have no stack. This may erase the barrier saving; the
one-core diagnostic did not predict face resources.
An unchecked full-interior diagnostic increased registers to 102 and is not
integrated. D8 resource probes remain negative controls, not a promised next win.

Compiler/proof artifacts are in
`.cache/cuda-autoresearch-2026-11-11/deep-warp/`, including `native-work.json`,
`sass-summary.json`, `typed-source20-diagnostic/report.json` and
`double-frame-typed-diagnostic/report.json`.

## Independent proof and CPU checks

```sh
python scripts/prove_cuda_deep_warp.py --field-frames 2 --output tokens.json
python scripts/prove_cuda_deep_shared_races.py
python scripts/prove_cuda_deep_interior.py --output bounds.json
JAX_PLATFORMS=cpu python -m pytest -q tests/unit/test_cuda_stencil_deep_generated.py
```

The token oracle checks version/coordinate dependencies, register histories and
unique final ownership over D1/D2/D4/D8 and CPML masks. Wrong shuffle row,
missing ramp retirement and wrong frame age are rejected. The conservative CTA
epoch proof treats each complete shared plane as possibly conflicting; it
rejects missing pre-overwrite barriers with one frame and missing publication
barriers with two frames. These are schedule proofs. They do not establish GPU
warp convergence, memory safety or floating arithmetic; full-state GPU gates
and bounded sanitizers address those separately.

Further work should use the measured full-physics bottleneck: reduce repeated
halo updates and metadata/synchronization, then test moving space-time ownership
with explicit CPML ownership, source timing and DFT publication. Deeper D4/D8
alone cannot solve the observed deficit. No 10× speedup is claimed here.
