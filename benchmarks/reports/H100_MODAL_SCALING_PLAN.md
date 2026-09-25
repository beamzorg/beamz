# Modal H100 scaling acceptance plan

Targets: **150 GCUPS CUDA streamed**, **75 GCUPS pure JAX** on eight H100 SXMs.
One update is one complete Yee timestep per physical material cell. Padding,
component count and independent replicas never increase the numerator.

Run `scripts/benchmark_modal_scaling.py --output <new-directory>` for the full
matrix, or add `--dry-run` to inspect it without initializing JAX or renting GPUs.
The default matrix includes 1/2/4/8 GPUs; weak scaling with local volumes
256³/384³/512³; strong scaling with global cubes 512³/768³/1024³; and 3/101
frequencies. All cases use 12-cell CPML, a mode source, two mode monitors,
heterogeneous dielectric waveguide material and 80 nm resolution. Weak scaling
lengthens one coupled waveguide along x. Cubes are domain-size diagnostics, not
claims about the useful physical complexity of a device. Monitor apertures stay
fixed at the modal port size. A large full-plane monitor is a different workload.

A fresh process runs each configuration. Five synchronized timed samples follow
warmup. Setup and compilation are outside stepping GCUPS. Public-path latency,
per-device allocator statistics, native binary hashes and failed trials are
retained. The default 256-step timing window can precede pulse arrival at distant
monitors; it measures stepping cost, not a converged optical result. Spectral
acceptance must propagate through the actual device and compare complete spectra
and continuation state against JAX. No numerical tolerances are relaxed.

Implementation sequence:

1. Keep normal CUDA CPML recurrences owner-local during the scan; assemble the
   public packed continuation layout only at scan boundaries.
2. Measure distributed-local CUDA arithmetic separately from communication and
   retain only validated kernel improvements.
3. Explicit local JAX stencil/CPML ownership, retaining a pure JAX implementation
   and fallback for unsupported configurations.
4. Measure local-volume saturation and strong/weak scaling; inspect exposed halo
   communication and source/monitor ownership before extending decomposition.

Eight-device targets require actual eight-device measurements. The initial
RunPod budget remains $100 total; prior estimated compute is $36.93. Start with
same-node controls on two devices, currently available, and recheck eight-device
capacity before final validation. GPU allocation and teardown are deliberately
outside the sweep script; `--max-wall-seconds` limits only the sweep process.
