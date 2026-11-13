The cosine candidate now runs its solver in **37.376 s**, versus **50.961 s** previously and **72.061 s** on freshly rerun current origin/main: **1.363×** the previous candidate and **1.928×** main. All 30 exported state leaves and all 10 exported observables are bitwise unchanged from the previous candidate. Both host and GPU process memory remain below main.

| Implementation | Warm solver median | Complete notebook | Peak host RSS | Peak GPU process residency |
|---|---:|---:|---:|---:|
| Current origin/main | 72.061 s | 93.698 s | 6.593 GB | 2.712 GB |
| Previous candidate | 50.961 s | 68.036 s | 5.829 GB | 2.665 GB |
| Material facts only | 42.653 s | 63.268 s | 5.829 GB | 2.678 GB |
| Selected material + source | 37.376 s | 53.953 s | 5.829 GB | 2.676 GB |

The fresh main revision is `387cc8fa8a48f3f1e69077bb50c299df21940916`, including preparation/memory [PR533](https://github.com/beamzorg/beamz/pull/533); the remote main SHA was checked during this run. Main warm samples were [72.25789603730664, 71.86438014730811]; selected samples were [37.334317576140165, 37.418410565005615]. Previous-candidate samples come from the prior confirmed run, rather than another new full rerun. Material-only was also executed as a complete notebook. The selected candidate also has identical values for all 134 captured input fingerprints and packet attributes versus the previous candidate. All four use the identical notebook hash `40331f6d8cf778df9c423766514f557a139867ee503166b7cdfb75f1069b45a9`.

The workload is the unchanged 481×481×90 30 nm isotropic regular-grid cosine crossing, with 20,822,490 physical Yee cells, 17,484 steps / 1 ps, FP32, 12 CPML cells on every face, 101 port frequencies and the required field/port monitors. Its native staggered supports have physical ZYX dimensions 90×481×481. Runs used the local RTX 3090 at 250 W, one GPU job at a time under the research lock. No RunPod or ABBA. A solver update here includes fields, CPML, source injection, DFT and clock advancement.

**Accepted changes.** The block material certificate decodes negative constant headers and direct literal blocks exactly, chooses common FP32 bits, and proves the entire depth 16 trajectory using a conservative inclusion/exclusion volume. It admits 1,462 bulk, 5,888 NoIO boundary and 813 observer tiles. Mixed interfaces retain their exact coefficients and generic lookup; no 256-entry approximation or material quantization was used. The uniform kernels retain the block format's terminal import/export behavior, so material-format independence does not silently change ownership.

The mode-source lowering recognizes the actual Hy/Hz/Ey/Ez mask 54 with six coincident contributions per active component. These are the real/imaginary contributions associated with three frequencies; all six ordered FP32 multiply/add operations are preserved. Bounds/time calculations are shared within a component, the contribution count is fixed at compilation, and the existing paired-plane register/shared schedule is generalized to this source class and exact block coefficients. The source kernel uses 64 registers/thread without spills, versus roughly 125–128 registers for the old generic kernels. Other source topologies and nonisotropic H retain fallback paths.

This extends fact extraction and lowering in the existing temporal CUDA compiler. It does not add a new Python/PyTorch operation-tracing frontend or change the physical discretization. The selected benchmark bridge remains `scripts/compare_cosine_cuda_notebook.py` with an explicit frozen native module; the installed default backend was not replaced.

**Experiments retained only as evidence.** Captured rectangular dimensions were lowered into the certified CPML kernel, with exact shape admission. Its prefix improvement versus material+source was about 0.2%, within the observed timing variation, so this extra shape class was not selected. Joining source-only with source+observer cohorts, or all three source-capable policy cohorts, preserved numerical results but changed prefix runtime by less than 1%. The joins retained antichain boundaries and updated launch and wait ranges together, including empty first cohorts. They were not selected. A later unchanged selected-prefix repeat was faster than either join screen, reinforcing that there is no demonstrated material gain.

All diagnostic screens execute 1,024 physical steps, use the same captured input packet, and compare all 21 field/CPML/DFT arrays bitwise. They are not substituted for complete notebook timings.

**Validation.** The full notebook repeats its field DFTs bitwise. Compared with the preceding candidate, all 30 state leaves and 10 observable exports are bitwise exact. Compared with main, the largest normalized transmission discrepancy remains approximately 4.55e-7; raw field Ez DFT peak-scaled maximum discrepancy remains approximately 1.55e-5. End-of-run transient fields differ by up to 0.5815% of their component peak, as before, because the candidate retains CPU preparation while main prepares on GPU. There is no added discrepancy from the changes here. This is implementation agreement on the same mesh, not a convergence study.

CPU tests: 17 passed, 1 intentionally skipped (more than 256 values cannot use a byte dictionary). They cover bit-exact constants/literals, aliases, one-ULP changes, signed zero, final partial blocks and rejected invalid pool addresses. Complete CUDA Program oracle gates cover lossless mixed/constant block encoding, source/observer sensitivity, nonisotropic fallback, odd tails, resets, continuation and private input capture. The selected observer suite has 16 records and mode suite has 22. The focused CUDA memory-sanitizer check passed with zero errors; its result is recorded in `memcheck.log` and `selected-sanitizer-correctness.json`; pass state is in `summary.json`.

Host memory is the Linux process maximum RSS across complete cold setup and repetitions. GPU memory is 50 ms sampled per-process NVML residency, including context, modules, graphs and allocator. Reclaimable filesystem page cache is outside process RSS. Full validation exports and warm repeats are excluded from the complete-notebook time; peak memory remains below main across those phases too. The selected GPU margin below main is 35.65 MB. It uses 10.49 MB more GPU residency than the previous candidate, while staying below main.

**What remains.** The selected CUPTI prefix trace has no dropped records and a 2.146 s kernel-interval union. Uniform NoIO CPML/boundary kernels account for about 29% of summed overlapping kernel durations, traced general-policy kernels 21%, and remaining generic interior-observer kernels 16%. These are not additive wall-time shares. The next substantial target is exact effect and ownership tracing for boundary/observer tiles and CPML histories, followed by register/shared lifetimes. Merely combining these existing source cohorts or specializing dimensions did not recover another large gain. No additional 2× improvement is claimed.

The selected module is `/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-block-uniform-2026-11-14-1255/mode-source/_cuda_native.cpython-311-x86_64-linux-gnu.so` with SHA256 `19d98d8f5f31d9acecdb6818f3eda388d3a138dd77d232a4c753c9a7da0df71c`. Its selection manifest is [selected.json](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-block-uniform-2026-11-14-1255/selected.json), complete evidence is [summary.json](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-block-uniform-2026-11-14-1255/summary.json), and experiments are in [the artifact directory](/home/quentinwach/.t3/worktrees/beamz/t3-273958bc/.cache/cuda-block-uniform-2026-11-14-1255). The installed default still hashes to `34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836`.

To reproduce the selected build from the prior bounded candidate:

```bash
python scripts/build_cuda_block_uniform.py \
  --parent .cache/cuda-memory-parity-2026-11-14-1204/lazy-plan-v2 \
  --output <new-candidate-directory> --mode-source
```

Then use the explicit module with `scripts/compare_cosine_cuda_notebook.py`, the same comparison notebook and pinned frontend, and run `scripts/check_cuda_memory_parity.py`. The frozen source and exact compile/link commands are retained in the candidate's `proof.json`.
