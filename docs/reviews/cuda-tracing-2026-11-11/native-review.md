Native review, 2026-11-11 12:24 UTC. Scope: the default packed-material
decoder correction and the default-off `FULL_INTERMEDIATES` schedule in
`cuda/src/{mstep.cuh,update.cu,program.cu}`, including their existing FFI,
dependency-plan, storage-transform and graph-cache contracts. This is a source
review; hardware results and sanitizer results must be attached independently.

No new promotion-blocking source defect was found. The selected binary passed
the full-state hardware gates and a completed tiny memcheck; racecheck/synccheck
coverage remains incomplete, including a matched control racecheck timeout.
The source review supports promotion with those tool limits recorded. The most
important semantic condition is that deliberately discarded E1 halo values can
be invalid: correctness depends on retaining the proven first-queue closure,
not on every computed intermediate being physically meaningful.

- **Resolved correctness defect: constant-block materials.** The preparation
  cache can share coefficients between streamed and mstep execution, so a
  stream-first request can give the packed mstep core rank-two block headers.
  Treating these headers as rank-one byte IDs was incorrect. The new private
  `code_ranks[3]` metadata selects the same signed-header interpretation as
  `yee_primitives.cuh::PackedMaterialSource(BeamzBuffer,BeamzBuffer,...)`:
  negative header means pool index `~header`; nonnegative header means
  `header+(logical&31)`. Rank-one IDs retain the previous decoder. Mixed
  representations are handled per component, and optional host specialization
  dispatches only after proving all three ranks. There is no floating-point
  conversion or coefficient approximation. The fix applies to the fast
  scalar-H packed core; generic packed/dense-H already uses the canonical
  buffer-aware decoder.

- **Resolved related storage defect, separate ownership.** Cyclic storage must
  transpose/repack rank-one byte IDs as well as rank-two headers. Root fixed
  that pre-existing issue in `beamz/simulation/cuda/storage.py`; the native
  decoder change alone cannot make a physically misordered codebook grid
  correct. Cache-order, mixed-layout, and cyclic-byte-ID regression gates are
  required together. The native code does not validate individual device-side
  pool indices; it relies on the existing validated Python packer, as does the
  canonical decoder. Malformed foreign device payloads are outside this proof.

- **Full-bank dependency proof.** The unchanged temporal core finishes its
  frozen-A reads and writes final core values into B before any first-edge
  kernel mutates A. H1 is computed on every original `first_cells` row S and
  reads the untouched E0 family. E1-S then reads H1-S and old E0. Some deep-core
  E1-S halo results can consume stale H1 outside S; these results are discarded.
  Required E1 comprises Q's own old-E samples, all E1 neighbours used by H2-Q,
  and requested first-step E observations. The existing S closure supplies all
  H1 reads of this required region. Second H2 is written to B-Q; second E2
  reads that complete H2-B family, not H1-A. Final B is the union of the core
  and clipped Q, so no uninitialized B region reaches the returned state.
  `test_cuda_stencil_full_intermediates.py` checks this closure using version
  tokens, explicitly exercises invalid discarded E1, and rejects missing H1,
  incorrect second-H bank selection and missing observed-neighbour closure.

- **Aliasing and race topology.** `MstepFullEdgePhase<...,CopyCore=false>` writes
  only the phase's own component family. Same-family old-field reads are at the
  same row/component being updated; all transverse neighbour reads belong to
  the opposite family. The generated queues are unique sorted row IDs, so
  different rows do not write the same field cell. Separate same-stream kernel
  launches order core, H1, H sources, E1, E sources, observations, H2 and E2.
  CPML terms have one writer per target row/term. There is no new cross-CTA
  synchronization requirement. `CopyCore=false` is compile-time; FULL-only
  packed-core view has no publication member and no publication stores.

- **Generic coefficients and supported scope.** FULL uses the original
  `EnqueueMstepCoreImpl<false>` dispatch for packed/scalar-H, packed/dense-H,
  ordinary/scalar-H and ordinary/dense-H kernels. The full-edge helper still
  calls canonical `UpdateComponent`, preserving scalar/broadcast/dense
  materials, PEC handling and F32/BF16 CPML. Mstep remains the existing
  Cartesian diagonal Yee schedule: tensor, dispersive and nonuniform-metric
  generalization is not introduced by this change. Dense-H, dense conductive,
  no-CPML and unequal-face CPML gates are necessary because the benchmark's
  packed/scalar-H material alone cannot validate these dispatches.

- **Isolated typed full-edge candidate.** Default-off `FULL_EDGE_TYPED` adds
  only homogeneous auxiliary-type and absent-PEC specialization to the full
  edge helper. Uniform CPML phases inspect all six input and output psi
  descriptors before selecting F32 or BF16. A zero metallic-edge mask is the
  only admission condition for removing PEC checks. Mixed types retain the
  dynamic psi/PEC kernel; nonuniform CPML and no-CPML dispatches are unchanged.
  This candidate contains neither the rejected interior bypass branch nor an
  active-axis switch. The new mixed-term precision hardware regression seeds
  alternating F32/BF16 H and E psi arrays at CPML12 and independently compares
  five-step streamed/mstep full state, including the odd tail. Its native result
  is pending at this review update; existing homogeneous BF16/F32, PEC and
  unequal-face tests remain necessary.
  Static inspection of frozen typed-v10 finds no spills in any full-edge
  variant. F32/no-PEC H and packed-E use 27 registers versus combined-v7's 29;
  selected static instruction counts drop from 888 to 752 for H and from 1072
  to 752 for packed E. These are compiler-resource observations, not throughput
  measurements.

- **CPML and source ownership.** The planner excludes every active CPML slab
  from C and excludes source columns with the existing two-cell transverse
  dependency margin. Every active CPML recurrence is therefore in Q and is
  advanced exactly once in each H/E phase of each physical timestep. Extra
  first-queue rows P lie in inactive-CPML core territory, so tainted discarded
  E1 cannot advance observable auxiliary state. Pre-E injection happens before
  the frozen core; the second pre-E injection happens before H2-Q. H/E source
  timing, coincident-source ordering and all observation launches are unchanged.
  Observations read A after the first step and complete B after the second.

- **Bank rotation, odd tails and input immutability.** After each pair B becomes
  current and A becomes the next destination. An odd tail uses the unchanged
  ordinary phase kernels on the complete current bank. Python returns bank
  `(nsteps//2)%2`, matching the native rotation, and fewer than two steps/no core
  use the existing ordinary fallback. FULL mutates more of its aliased input
  bank than compact mstep, so public immutable-state behavior must be tested,
  not inferred from native pointers. `_graph_io` declares field/psi aliases to
  XLA; retained non-donated inputs must receive the compiler's copy protection.
  The new publication test checks retained input values after whole, repeated,
  reference and split calls, including CPML12 and DFT/source clocks. Native raw
  callers do not receive independent ownership merely from this wrapper.

- **Graph reuse and concurrent streams.** The graph key includes a distinct
  FULL schedule prefix, stream handle, timestep count, all phase metadata and
  buffer descriptors, all plan buffers, both field workspaces, source/monitor
  buffers and packed-material ranks. Graph replay preserves the same sequence
  and uses runtime device contents for changing clocks and waveforms. Cache
  entries are per device, locked, and completion-event guarded. Distinct
  streams have distinct keys; this schedule adds no global mutable device
  storage. The per-kernel dynamic-shared-memory attribute is a constant for
  each template instantiation. Simultaneous execution on disjoint states has
  not been stress-tested by this review; sharing writable raw banks across
  streams still requires the caller's ordering contract.

- **Integer bounds and metadata trust.** Python rejects envelope volume at or
  above `INT32_MAX`, and rejects compact scratch offsets at that bound.
  Native `FitsIntOffsets` validates dimensions/products and requires dense
  canonical A/B row/plane strides. Thus generated row IDs, full field offsets,
  codebook logical indices and `ny*nx` fit signed int. Rank-two pool size is
  bounded by the packer. Native shape validation does not inspect queue row
  contents, uniqueness, source/core exclusion or bounds values on the GPU; the
  device plan is trusted generated metadata, just as in the compact schedule.
  FULL removes compact lookups from edge fields without widening these bounds.

Evidence available at review time:

- Root reports 453 CUDA/memory CPU tests and 13 FULL taint tests passing; the
  production register-coordinate proof also passes all 9 cases with fresh
  wave-local register dictionaries. Root reports Ruff clean. A separate traced
  hardware test has a formatting-only wrap pending; it is not a native defect.
- The frozen FULL-v6 binary is
  `.cache/cuda-autoresearch-2026-11-11/builds/full-intermediates-v6/build/_cuda.cpython-311-x86_64-linux-gnu.so`,
  SHA256 `8348f41e28354e0707b1aa6ff561c5ea6327a3df3e894203b4a8ded98d71608f`.
  Its packed-core register counts exactly match control-v6 for all four tiles
  (48/48/56/48); full uniform-CPML edge kernels use 29 registers, no stack/local
  memory. Hardware matrix and sanitizers are queued with the benchmark agent.
- The benchmark owner subsequently reports frozen FULL-v6 and combined-v7 each
  passed their complete 157-case hardware suite. The suite now has 158 cases
  after adding mixed per-term psi precision; that additional case and the final
  selected candidate's sanitizer gates remain pending. These results do not
  automatically certify register or typed-edge variants.
- The benchmark owner then reports typed-v10 passed all 158 hardware gates,
  including mixed per-term precision, and exact full-state comparisons for all
  three timed fixtures. Its frozen binary SHA256 is
  `e80907a522581968b319924131170a3ecacc3d48d008e9de21ca2da0bb968255`.
  A typed-plus-generated-cell variant is being screened before selecting the
  exact binary for sanitizers; none has been sanitizer-certified yet.
- Existing hardware `_compare` uses peak-scaled `1e-5` tolerance for floating
  arrays (some JAX comparisons use `3e-5`) and exact comparisons for integer
  leaves. Replay/retained-input tests additionally require array equality.
  These are full-state comparisons, not a physical convergence or energy-error
  study. Host taint proofs establish dependencies, not CUDA memory ordering.

Sanitizer commands, to be run only by the GPU owner after installing and hashing
the frozen candidate and saving the previously installed component. The owner
must restore that component afterward. Each command can be bounded externally
to 300 seconds; a timeout is an incomplete result, not a pass.

```bash
for sanitizer_tool in memcheck racecheck synccheck; do
  env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$PWD" \
    /home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/compute-sanitizer/compute-sanitizer \
    --tool "$sanitizer_tool" --kernel-name kns=Mstep \
    --target-processes application-only --launch-timeout 120 --error-exitcode 86 \
    --log-file ".cache/cuda-autoresearch-2026-11-11/full-v6-${sanitizer_tool}.log" \
    /home/quentinwach/Code/.venv-cuda-pr233/bin/python -m pytest -q \
    'tests/hardware/test_cuda_mstep_publication.py::test_published_edges_preserve_initial_state_repeats_and_split_clocks[2-False-gaussian]' \
    'tests/hardware/test_cuda_mstep_publication.py::test_published_edges_rotate_byte_ids_after_mstep_first_preparation[201]'
done
```

The filter instruments temporal core and full edge kernels, including the
CPML12, seeded nonzero-state, replay/split/input-alias and cyclic-storage
exercises. It does not instrument unrelated reference/source/DFT kernels.
Racecheck primarily diagnoses shared-memory hazards; it does not prove absence
of global-memory races between field families. Synccheck does not prove a
numerical dependency closure. These tools complement the source topology proof,
taint negative controls and full returned-state comparisons. No sanitizer result
is claimed in this review until the GPU owner attaches the actual logs.

Sanitizer execution update, 2026-11-11, selected typed-plus-generated-cell v11:

- Selected binary SHA256 is
  `4cdde58fc618dad17b0c7f04ec6050e8ec356eacdf51d153c4b98ba714500d1a`;
  benchmark owner reports all 158 normal hardware gates and exact full-state
  equality on all three timed fixtures. A new tiny tool case brings the final
  suite to 159 cases: `test_mstep_small_cpml12_sanitizer`, shape 35x49x65,
  CPML12, no sources/monitors, seeded nonzero state, positive-core assertion,
  one independently compared streamed/mstep pair.
- The original larger two-case memcheck run passed both numerical tests but
  exited 86 with 140 CUDA API reports from XLA constant-symbol lookups
  (`cuModuleGetGlobal_v2`, `CUDA_ERROR_NOT_FOUND`). It is **not a sanitizer
  pass**. The raw log is retained under
  `.cache/cuda-autoresearch-2026-11-11/full-typed-traced-v11-sanitizers/`.
  Its racecheck exceeded 180 seconds without a diagnostic; synccheck was
  stopped at 149 seconds when root replaced the workload. Both are incomplete.
- Root authorized `--report-api-errors no` to isolate device-memory, race and
  synchronization diagnostics from those host symbol probes. This disables
  host API error reporting; it does **not** disable device instrumentation.
  On the tiny CPML12 case, memcheck completed in 5.2 seconds with one test pass,
  exit zero and `ERROR SUMMARY: 0 errors`. Tiny racecheck and synccheck each
  reached their 90-second bound without completing or emitting a finding.
  They are incomplete. Logs and commands are under
  `.cache/cuda-autoresearch-2026-11-11/full-typed-traced-v11-tiny-sanitizers/`;
  each log embeds the exact selected binary hash.
- Graph caching remains enabled for this attempt. The tool runner hashes the
  frozen manifest/binary before installation and explicitly restores validated
  fixed-control-v2 SHA256
  `792d511f87d4d5255d812d89e860faae9fc0e8a1742e6a784f6d9e044cd614fe`
  after completion or a bounded termination. The finally/restoration hash is
  recorded in each run's `report.json`.

Final bounded follow-up and handoff at 13:05:53 UTC:

| Binary and scope | Tool | Outcome |
| --- | --- | --- |
| v11 selected, tiny CPML12, all Mstep kernels, cache enabled | memcheck | 1 test passed; 0 errors; exit 0; 5.2 s |
| v11 selected, same tiny case | racecheck / synccheck | Each timed out at 90 s; incomplete |
| v11 selected, tiny case, cache disabled, one default byte-layout core launch | racecheck / synccheck | Each timed out at 60 s; incomplete |
| final-control-v11, tiny case, cache disabled, one default generic-layout core launch | racecheck | Timed out at 60 s; incomplete |

All these filtered attempts disabled host CUDA API error reporting and retained
device instrumentation. The final control SHA256 is
`fe78cbb39c9ae6d2275a826e7751278055abd5784b6482e24351288a791931da`.
The narrowed filters were respectively
`regex=MstepPackedColumnCore.*Li32ELi16ELi1E` and
`regex=MstepPackedColumnCore.*Li32ELi16ELi0E`, each with `--launch-count 1`.
`BEAMZ_CUDA_DISABLE_GRAPH_CACHE=1` disables cache reuse; `BeamzLaunchGraph`
still performs capture/instantiation, so these were **not graph-free runs**.
The matched control timeout shows the observed limitation is shared by that
control path; it does not prove the candidate has no race or synchronization
defect. No completed racecheck or synccheck pass is claimed.

Narrowed logs/reports are retained under
`.cache/cuda-autoresearch-2026-11-11/{full-typed-traced-v11,final-control-v11}-narrow-core-sanitizers/`.
All sanitizer processes were confirmed gone, fixed-control-v2 SHA256
`792d511f87d4d5255d812d89e860faae9fc0e8a1742e6a784f6d9e044cd614fe`
was verified installed, and GPU ownership was explicitly returned to the
benchmark agent before its final tuning queue. The 159th tiny test passed under
memcheck; the final default-build/full-159-suite gate remains root's separate
promotion check.
