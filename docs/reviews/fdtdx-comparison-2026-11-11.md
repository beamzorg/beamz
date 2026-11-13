# BeamZ CUDA versus released FDTDX: matched Mie benchmark

Status: **Stopped at the user’s request.** Three RTX3090 timing cases completed; the 2 nm case passed numerical validation but its timing was stopped. No RTX4090 simulation ran, no RunPod pod was created, and the account balance remained unchanged.

## Completed results

RTX3090, existing 250 W limit. Geometric means of two fresh-process medians, each from three synchronized warm solves; setup and compilation excluded.

| Resolution | Grid | Steps | BeamZ | FDTDX0.6.3 | FDTDX / BeamZ |
|---|---:|---:|---:|---:|---:|
| 5 nm | 136³ | 3280 | 1.76 s | 2.44 s | 1.39× |
| 4 nm | 170³ | 4100 | 3.53 s | 6.27 s | 1.77× |
| 2.5 nm | 272³ | 6560 | 17.20 s | 36.07 s | 2.10× |

BeamZ’s measured speedup grows from **1.39× to 2.10×** over the completed sweep. These measurements do not establish a 10× engine speedup.

| Resolution | BeamZ sampled peak | FDTDX sampled peak |
|---|---:|---:|
| 5 nm | 1.65 GiB | 1.38 GiB |
| 4 nm | 2.14 GiB | 1.62 GiB |
| 2.5 nm | 5.17 GiB | 3.26 GiB |

These are device-global compile/execution peaks, including the idle local GPU context, not isolated allocations. BeamZ uses more memory in this configuration.

All four local resolutions passed strong-signal field validation and the amended full-run spectrum/absolute-field checks. Maximum full-run phasor relative L2 error was **1.394e-4 (0.01394%)**; final field error relative to the fixed strong-signal witness was at most **2.057e-6**. Raw weak late-time field relative discrepancies remain **13–25%**; this benchmark does not establish accurate relative ringdown amplitudes. The original failed criterion and amendment are documented below.

RunPod Secure and Community creation attempts returned HTTP400: no instances available. Both attempts left the pod list empty and spend at $0/hr. The user then canceled all RunPod work. No endpoint, worker, or pod was provisioned.

Artifacts: [chart](../../.cache/fdtdx-comparison-2026-11-11/report/comparison.png), [standalone HTML](../../.cache/fdtdx-comparison-2026-11-11/report/comparison.html), [summary](../../.cache/fdtdx-comparison-2026-11-11/report/summary.json), and [raw campaign](../../.cache/fdtdx-comparison-2026-11-11/local-3090-final/comparison.json).


This campaign compares the installed BeamZ CUDA production-v13 component with
FDTDX **0.6.3**, the latest published release verified on 2026-11-11 through the
live PyPI and GitHub release APIs. That release uses JAX. It does not contain the
unreleased PyTorch custom-tracing rewrite described in the creator's post.

Primary release sources: [PyPI 0.6.3](https://pypi.org/project/fdtdx/0.6.3/),
[GitHub v0.6.3](https://github.com/ymahlau/fdtdx/releases/tag/v0.6.3).
Release commit: `59ec558484d709d511cd70e3a8a45c05e662cfda`.
BeamZ native SHA256:
`34de360a045eceb3343a9fc88bb94f4c4b2ee7e65a09b6fc8a8d66ef59e5f836`.

## Workload and fairness

The 680 nm cube includes twelve CPML cells on each face at every resolution.
It contains a radius-100 nm nondispersive n=2 sphere in vacuum, illuminated along
+x with the same Ey/Hz 500 nm Gaussian source. Six faces record one-frequency
complex E/H field DFTs every timestep. Precision is FP32; gradients, symmetry,
progress displays, and boundary recordings are disabled. The physical duration
is 30 fs with CFL fraction 0.95. No engine parameters are tuned during comparison.

These are **common-discrete-data adapters**, rather than two scenes whose public
API defaults differ. `export_fdtdx_comparison.py` exports prepared BeamZ
component material coefficients, exact source slab coefficients and waveforms,
CPML profiles, and detector interpolation sites/weights into a SHA256-bound
contract. The FDTDX adapter configures these through source/detector subclasses
and public array/object fields, while preserving the published run/update/curl
implementation. Both solve the same common discrete system. Native default
FDTDX detector interpolation and source pulse normalization are not measured.

BeamZ uses separate Yee component supports with terminal planes; FDTDX uses
three-component common N³ supports. The comparison crops/transposes to common
physical Yee sites. FDTDX H is normalized by the vacuum impedance and is converted
to SI for comparison. Pinned outer electric nodes and high-face CPML support
alignment are explicitly represented in the common data.

The published FDTDX step is E-first and BeamZ is H-first. The adapter uses
FD H source[n] = BeamZ H source[n] and FD E source[n] = BeamZ E source[n−1], with
E source[−1]=0. After N complete updates, FD E is compared against BeamZ E from
N−1 updates, and both final H fields correspond to the same half-step. Detector
E phasors are aligned analytically by a one-step phase rotation and removal of
BeamZ's final E sample; no fitted amplitude or phase is allowed. Extra validation
solves and state downloads are outside timing and device-memory sampling.

`validate_fdtdx_comparison.py` independently checks contract/data hashes, resolved
work metadata, common interior E and H, and each detector's E and H phasors
separately. The predeclared maximum relative L2 discrepancy is 1%. Interior field
validation excludes fourteen cells per face (CPML12 plus two guard cells).
Detector sites are all checked. A failed numerical gate prevents any timed
campaign for that contract.

## Timing method

Both engines use the same pinned Python/JAX environment. The driver executes
fresh BeamZ/FDTDX/FDTDX/BeamZ processes after validation. Each process resets the
same immutable initial state and synchronizes all returned device leaves.
Warm executable solve timings exclude setup and compilation. Cold scope includes
scene construction, lowering, compilation, and the first synchronized result,
but excludes Python imports and validation exports. Public-dispatch timings use warmed prepared JIT wrappers on both engines and
are reported separately from executable solve timings.

Device-global memory is sampled every 50 ms and includes driver reservations
and unrelated allocations; it can miss short spikes. JAX allocator statistics
are a separate measure and omit some native CUDA allocations. Ratios compare
engines on the same GPU. Two processes per engine do not establish a strong
confidence interval, and samples within a process are not independent trials.
The local RTX3090 remains at its existing 250 W power limit.

## Reproduction and evidence

Source recipes live in `scripts/benchmark_fdtdx_comparison.py`,
`scripts/export_fdtdx_comparison.py`, `scripts/beamz_release_adapter.py`,
`scripts/fdtdx_release_adapter.py`, and `scripts/validate_fdtdx_comparison.py`.
Generated release provenance, immutable contracts, exact environment pins, raw
validation states, gates, timing JSON, logs, and plots live under
`.cache/fdtdx-comparison-2026-11-11/`.

## Numerical protocol amendment before timing

The original strict final-field relative criterion rejected the 5 nm 30 fs
validation: E differed by 12.69%, H by 15.96%, while every full-run detector
E/H phasor differed by at most 2.67e-6. The failed gate and raw states remain in
`local-3090/case-00/`; no performance campaign was admitted by that gate.

Further validation established that the absolute discrepancy remains roughly
constant as the signal decays: at a fixed 1024-step (9.367 fs) sphere witness,
E's L2 norm is 7.67e9 and maximum absolute discrepancy is 78 V/m; at 30 fs its
norm is 7.30e4 and maximum discrepancy is 60.6 V/m. The strict 1% witness gate
passes with E/H relative discrepancies of 1.71e-6 / 2.09e-6. A 512-step sphere
check and a 512-step vacuum check also pass without changing the criterion.
Spatial Fourier diagnostics assign over 92% of final discrepancy power to
frequencies above half the grid Nyquist frequency. This supports accumulated
FP32 grid-scale noise in decayed fields; it is an inference, not a high-precision
reference solution or proof of the exact rounding mechanism.

Before any comparative timing was accepted, the protocol was amended to require
all of the following at each resolution, on each GPU:

- An independently validated, fixed 9–10 fs strong-signal witness must meet the
  original 1% relative criterion for interior E/H and every detector E/H phasor.
- That witness must use exactly the same material, CPML, and detector operators,
  identical source coefficient arrays and waveform prefixes, and the same
  adapter/native identities as the full-duration case. Hash bindings are checked
  and the strict witness gate is rerun by the full-duration validator.
- Full-duration detector E/H phasors must still meet 1% relative L2 error.
- Each full-duration interior field difference, normalized by that component
  family's strong-signal witness norm, must be at most **1e-5**. E and H remain
  separate. No amplitude or phase fitting is allowed.

Every amended gate retains the raw final-field relative discrepancies and whether
those passed the original criterion. Acceptance therefore concerns agreement on
strong-signal fields and full-run spectra, with a strict absolute floor relative
to the signal witness. It does **not** establish 1% relative agreement of weak
late-time residuals or high-Q ringdown. This distinction matters to anyone using
the benchmark to infer numerical accuracy beyond its measured observables.

Cold numbers are adapter setup-to-first-result scopes: BeamZ rebuilds the scene
and discretization, while FDTDX configures already-exported common arrays. They
must not be interpreted as an equal-work scene-construction speed comparison.

## Portable preparation

The first complete 5 nm ABBA comparison passed. The next resolution's strict
input audit rejected CPML coefficient preparation: the earlier environment's
CPML `a` coefficients differed from the shared comparison environment by only
one or two FP32 ulps. Material, source, detector, and other CPML arrays matched.
This is not grounds for waiving the common-input requirement. The portable
adapters therefore explicitly configure the frozen canonical CPML coefficient
arrays before lowering, preserving the native update code and exact input
hash audits. The original failed preparation record remains available.

Public dispatch measurements use the warmed prepared JIT wrapper on each side;
BeamZ's broader `Simulation.advance` API additionally assembles host results and
is not an equal-work dispatch comparison. Earlier diagnostic numbers with that
broader scope remain historical, not part of the final dispatch comparison.

The creator's post specifies the domain size and application but does not supply
all material, source, duration, detector, hardware, and software-version details
needed to reproduce its plotted timings exactly. This campaign uses the explicit
common recipe above; it does not reproduce the post's 19x or 3–6x claims.
