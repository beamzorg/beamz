# Larger canonical nanoparticle reference

`scripts/export_cuda_large_fixture.py --recipe mie` prepares the same physical
nanoparticle experiment at a larger grid, using bounded NPY files instead of a
large compressed NPZ. The existing guide recipe remains the default.

The exported case is a 680 nm cube including 12 CPML cells per face, a 100 nm
radius sphere with refractive index 2, a 500 nm Gaussian plane-wave source,
CFL 0.95, and six surface monitors gathering all six E/H components. The normal
BeamZ binary material raster and Yee coefficient averaging are retained. At
512³, the cell width is 1.328125 nm and the CPML thickness is 15.9375 nm.
The timestep is 2.429859859763711e-18 s; 12,347 updates reach approximately
30.00148 fs. The prepared waveform horizon is the full solve duration.

Both the 256-step prefix and the full solve were independently executed with
the ordinary `cuda_streamed` Program schedule through its JAX wrapper. This
avoids using the moving candidate as its own reference. The ordinary schedule
uses the normal 256-step chunk policy. All 30 reference leaves are finite;
the completed counters are exactly 256 and 12,347, respectively. All six full
DFT faces have nonzero signal. The early prefix has not yet reached the far
face and is insufficient evidence for full-pulse numerical equivalence.

All 134 prepared input arrays have identical hashes between the prefix and
full exports, including full-length source waveforms. Only the executed step
count differs. The prefix changes `Program.config.num_steps` after preparing
the full waveform horizon. It does not truncate or regenerate source data.

Reproduction on the recorded local environment:

```sh
PYTHON=.cache/fdtdx-comparison-2026-11-11/venv/bin/python
SHA=34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836
OUT=.cache/cuda-engine-autoresearch-2026-11-11-2032/mie512
$PYTHON scripts/export_cuda_large_fixture.py --recipe mie --shape 512 512 512 \
  --reference-steps 256 --reference-ownership owned \
  --expected-component-sha "$SHA" --output "$OUT/mie512-prefix256.json"
$PYTHON scripts/export_cuda_large_fixture.py --recipe mie --shape 512 512 512 \
  --reference-ownership owned --expected-component-sha "$SHA" \
  --output "$OUT/mie512-full30fs.json"
```

The pre-initialization admission checked approximately 18.39 GB of conservative
GPU capacity against 24.87 GB free, and 17.82 GB of host capacity including a
4 GiB reserve. During the full reference the JAX allocator peaked at
16,266,137,600 bytes; host peak RSS was 8,924,737,536 bytes. These are reference
export measurements, not native candidate memory claims or timing results.

`export-summary.json` in the output directory binds the two final fixture
hashes and exporter source hash. `exporter-source.py` preserves the exact
executed source. Each final fixture binds its NPY manifest and every input and
reference array. Use the final JSON files for validation; the intermediate
`.prepared.json` describes the recoverable pre-reference packet and its earlier
manifest state.

A candidate must pass strict comparison of all 30 reference leaves before
timing is accepted. Complete 30 fs measurements retain sources, CPML, DFT,
clock updates, and chunk tails. A fast 256-step prefix alone does not establish
a full-duration speedup.
