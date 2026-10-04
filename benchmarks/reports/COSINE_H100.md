# Cosine-crossing H100 capacity benchmark

Run `scripts/benchmark_cosine_h100.py` from the repository root with GPU JAX and
the native BeamZ rasterizer/CUDA component installed. This is an exploratory
hardware experiment that deliberately attempts GPU allocation failures, not CI.

The workload is the three-dimensional cosine crossing from
`examples/notebooks/cosine_waveguide_crossing.ipynb`: silicon index 3.67, silica
index 1.45, 161 nm thickness, 350 nm straight waveguides, and 5.3 µm cosine
tapers. The nominal domain is 13.7 × 13.7 × 1.965 µm, rounded upward to whole
cells independently at every resolution. Polygon discretization stays fixed at
the notebook's 30 samples per taper. The source and monitor planes retain their
physical sizes and 0.5 µm clearance from the snapped domain edge.

Unlike changing the notebook's resolution parameter directly, this benchmark
holds CPML thickness fixed at 428.337875 nm (12 cells at its reference grid).
It keeps the TE mode source, one three-component field DFT plane, and two flux
planes with 101 frequencies from 1.26–1.36 µm. All fields and CPML recurrence
states use float32. No movie frames or full-volume history are retained.
The default 20-million-cell preflight limit is explicitly disabled for this
capacity experiment; per-axis bounds remain in place.

## Measurement protocol

- Each resolution/backend/run type gets a fresh child process.
- JAX preallocation and persistent compilation/raster caching are disabled.
  The JAX memory fraction is 0.95; report allocator capacity separately from
  physical VRAM. GPU OOM therefore bounds this execution configuration.
- `NUMPY_MADVISE_HUGEPAGE=0` prevents severe host allocation stalls observed on
  the rented machine. It changes memory allocation policy, not solver arithmetic.
  Preserve this setting when comparing setup latency with these records.
- Throughput: 256 timesteps, one discarded warm-up, five timed repetitions.
  Every repetition starts from a new zero state outside the timer; execution is
  synchronized. State donation matches the ownership policy used by `sim.run()`.
- GCUPS = grid cell count × timesteps / median seconds / 1e9.
- Full runs: the original 1 ps physical duration, with resolution-dependent
  timestep counts. These are single cold executable launches, not warm medians.
- Rasterization, mode/program setup, XLA lowering, XLA compilation, and execution
  are separated. Full runs also extract transmission/crosstalk. Whole-child wall
  time includes imports, initialization, verification, and measurement overhead;
  it is not a direct timing of the public `sim.run()` method.
- JAX allocator statistics and externally sampled total GPU memory (250 ms
  nominal interval) are recorded. Sampling can miss short spikes and includes
  the CUDA context. Linux peak process RSS describes host memory.
- The parent samples cgroup memory usage and checks `memory.events` to
  distinguish a host OOM kill from a GPU allocation failure.
- A GPU allocation failure stops the coarse sweep. Bisection then narrows the
  last-success/first-failure interval. Timeouts and non-GPU failures are recorded
  separately and must not be called GPU-capacity limits.

```sh
python scripts/benchmark_cosine_h100.py \
  --output-dir benchmarks/results/cosine-h100
```

Every child writes a log and JSON record; existing records are never reused.
Use a new output directory when changing protocol, source, hardware or software.
The parent writes `summary.json` and `measurements.csv` after its requested trials.
The coarse sweep defaults to 36, 30, 25, 21, 18, 15, 12.5, 10.5, 9, 7.5, and 6 nm,
with three bracket refinements and full 1 ps runs at 36, 25, and 18 nm.

Interpret throughput saturation separately from memory capacity. Optical
convergence must be supported by the full-run spectra; a short throughput run
does not establish optical accuracy or late-time numerical stability.

Use a fresh output directory for each code revision. The runner refuses to reuse
existing trial JSON files, preventing stale measurements from entering a new sweep.
