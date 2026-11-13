# Good research checkout: shared temporal CUDA candidate

This directory preserves the measured four-geometry checkpoint and the selected
native module identity in Git. See the [campaign report](../cuda-four-geometries-2026-11-14.md)
for accuracy, performance, memory, workload variations and remaining limitations.

`summary.json` and `report.html` contain the comparison; `captures.json` records
prepared-input fingerprints and workload facts. `selected-build.json` records
source hashes and build commands, and `evidence.json` identifies validation artifacts.

The selected native module has SHA256
`511dfe2a7651dae7bc23249e8a8b0b98e7885cca88e0e0393950a8c0a48d78dc`.
`selected-kernel.cuh` preserves the exact tested CUDA source. The canonical
`cuda/src/temporal/kernel.cuh` differs only by trailing-whitespace normalization.

This is a good checkout for continuing the validated research candidate. It is
not a standalone production release: the experimental build still requires the
frozen native ABI/build parent referenced in the campaign report. Large raw
captures, compiled objects and the native binary remain in the local `.cache/`.
