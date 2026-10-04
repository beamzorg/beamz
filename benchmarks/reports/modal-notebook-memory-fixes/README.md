# Modal notebook parity after CUDA preparation memory fixes

Executed 2026-09-25 on the **local NVIDIA GeForce RTX3090**, using
`cuda_streamed`, JAX 0.9.0, and CUDA extension 0.19.0.
Compared pre-fix `36ad890` with `1cbbfca` (includes `a0ea8a6` and `95488bb`).
Both used the same Python environment, CUDA binary, Rust rasterizer binary,
allocator (`cuda_async`), and notebook input.

## Result

- Both full notebooks executed top-to-bottom without cell errors.
- **37/37 exported arrays are bit-for-bit identical**.
- All 10 rendered PNG figures identical: **True**.
- All numerical outputs are finite. The comparison includes raw flux, complex
  forward/backward modal amplitudes, mode-monitor flux, raw and normalized Ey
  snapshots, normalized responses, plotted powers, modal accounting, effective
  indices, and frequency arrays. Comparing raw values prevents reference
  normalization from concealing a change.
- All ten figures from the updated run were visually inspected, including
  geometry, solved modes, field snapshots, broadband interpolation, power
  accounting, and junction mode decomposition.

The full tutorial uses the 180×77×56 rectilinear straight-waveguide grid,
12,605 steps, 17 monitor frequencies, three candidate modes, and seven broadband
source profiles. All three simulations ran: single-profile straight waveguide,
broadband straight waveguide, and the width-step junction. `BEAMZ_DOCS_TEST` was
unset; notebook simulation parameters and plotting cells were unchanged.

At the central frequency, the broadband forward fundamental-mode power is
0.999999177 W. The junction's forward modal powers are approximately
[0.710846230, 6.69e-16, 0.249555210] W, totaling 0.960401440 W under the notebook's
straight-reference normalization. These match the pre-fix outputs.

**Scope:** this notebook uses a nonuniform grid, whose dense source diagnostic
path was retained by the memory fix. This is a regression check for the complete
tutorial; the uniform-grid crop is separately covered by the tests and GPU
state parity recorded in [the memory report](../CUDA_PREPARATION_MEMORY.md).

## Artifacts and reproduction

The aggregate comparison is recorded above. Executed notebooks, HTML preview,
NPZ arrays, native hashes, logs, environment, and provenance are local artifacts in
`benchmarks/results/modal-notebook-memory-fixes/`.

- Current notebook: `current/modal_sources_monitors.ipynb`
- HTML preview: `current/modal_sources_monitors.html`
- Baseline notebook: `baseline/modal_sources_monitors.ipynb`
- Notebook input SHA256: `8022c0ccbeda914218f1455744174b761e602bebd58eaea4bb56981253e21574`
- Native CUDA SHA256: `541d6c6d27fb905fe27ed53b8711c0b56f37bd3b566c2b92d2b37f452077aca1`

Run each checkout sequentially in a fresh kernel, using the same native binaries:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python scripts/validate_modal_notebook.py \
  --root "$PWD" --output /tmp/modal-current --expected-device 'RTX 3090'
```

The runner saves the executed notebook and unnormalized exports without changing
the source notebook. A 14 GiB host-memory cgroup limit was used for both runs.
Validation/export wall times: current 145.3 s, baseline
145.6 s; these include mode projection and plotting and are not
FDTD-only performance measurements.
