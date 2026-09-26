# Eight-H100 CUDA capacity and throughput study

## Question and protocol

Does increasing cell count beyond the previous 2.097-billion-cell benchmark
improve synchronized `cuda_streamed` throughput, and where does the memory
capacity limit occur? This study uses the accepted ABI-21 native binary from
`f98fdc4`, with SHA-256
`6753641e1a682751ad00c10abda53bd4cd33fd3569ccf048baea018bb94be00d`.
Runtime kernels are unchanged. The deployment manifest identifies the local
source commit and file hashes; the remote Git commit is an archive snapshot.

All measurements use eight H100 SXM 80 GB GPUs, x-axis partitioning, FP32 CPML,
fast math disabled, 12-cell CPML on all external faces, a TE mode source, and two
101-frequency mode monitors. Each fresh process initializes on CPU, distributes
the state, compiles, warms up once, then measures five synchronized 256-step
samples. Input state is retained, matching the prior benchmark; this capacity
limit is not necessarily the limit of a donating/evolving-state application.

The balanced domain-growth series uses `(z,y,x)=(n,n,8n)` at 80 nm spacing.
The planar series uses `(n,8n,64n)` at 80 nm. The fixed-physical-domain series
uses `(n,n,8n)` with spacing `80*512/n` nm, retaining a physical domain of
40.96 × 40.96 × 327.68 µm. Nominal core dimensions remain 320 × 480 nm,
with each half-width rounded to whole cells, so realized interface positions
change with discretization. This is a nominal fixed geometry, not an exact
subpixel-preserving rasterization. The source envelope has fixed physical time scale.
Source/monitor apertures are fixed in physical units; their sampled size grows
with refinement. Twelve CPML cells have varying physical thickness in that
series. These timing tests do not establish spatial convergence or comparable
absorption error across resolutions.

GCUPS counts physical cells times complete Yee timesteps, including source and
monitor arithmetic, divided by median synchronized runtime. Setup, compilation,
finite-state checks, and any profiling are excluded. The source pulse and ports
can be far apart relative to these short runs; nonzero monitor weights verify
observation execution, not pulse arrival or a converged spectrum.

Memory is reported as per-device allocator live/peak allocations separately from
allocator pool size and sampled `nvidia-smi` process/device residency. The latter
is sampled every 500 ms across setup, compilation, and timing and can miss short
peaks. GPU utilization and memory utilization from `nvidia-smi` are activity
metrics, not achieved HBM bandwidth or SM occupancy. Allocator fraction is 0.95,
with preallocation disabled (previous study used 0.80).

A throughput plateau requires two successive substantial volume increases with
less than 3% improvement and sample variability below that difference. The
largest successful probe is distinguished from the largest repeatedly validated
size and any failed allocation boundary. A failed probe is retained as evidence.

## Execution

RunPod pod `d5yvneey683trr`, AP-IN-1, created 2026-09-26 17:29:52.756 UTC at
$27.92/hour. Additional spending authorization: $50. Measurement cutoff 19:00 UTC;
independent MCP deletion guard 19:09 UTC. All-pair NV18 topology verified.

The initial baseline measurements are 146.10 GCUPS at 512³ cells/GPU and
152.71 GCUPS at 640³ cells/GPU. The latter is within 0.8% of the accepted
151.57 GCUPS result. Further measurements are in progress; no capacity or
saturation conclusion is claimed yet.

Nsight Compute is installed, but an isolated CUDA-kernel probe returned
`ERR_NVGPUCTRPERM`. The host restricts hardware counters and the container lacks
the required permission. Hardware bandwidth, occupancy, and stall metrics
are unavailable on this run. Sampled NVML activity must not be substituted for
those metrics. The probe ran between benchmark processes with all GPUs idle;
its log and the idle check are retained.
