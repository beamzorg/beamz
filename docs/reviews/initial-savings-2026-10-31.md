# Initial implementation savings, 2026-10-31

Target: `dev/refactor_3110`. Audit baseline:
`153d7d3e8e7a8ef077e84c82e53c42e23abb8306`.

The initial implementation removes **1,537 net physical Python lines** from
`beamz/`: **50,211 → 48,674**. Every added helper, representation and final Ruff
formatting is included. No implementation moved outside the package, no supported
backend or physics capability was retired, and scientific documentation and
numerical tolerances were retained.

| Owned area | Before | After | Net removed |
| --- | ---: | ---: | ---: |
| Simulation | 16,723 | 16,434 | 289 |
| Devices | 13,821 | 13,236 | 585 |
| Design | 8,244 | 7,924 | 320 |
| Analysis, lattice and other shared code | 11,423 | 11,080 | 343 |
| Total | 50,211 | 48,674 | 1,537 |

Count with `sum(len(p.read_text().splitlines()) for p in Path('beamz').rglob('*.py'))`.
Tests, notebooks and this document are outside this metric. New test fixtures
construct canonical inputs; no production numerical algorithm was moved into them.
The source-backed architectural estimates were still optimistic: the completed
representations preserve distinctions that their proposed replacement-size caps
underestimated. The package remains **8,675 lines above 39,999**. This pass does
not achieve the below-40,000 goal.

## Implementation and API migration

- Request and compiled controls share `RunConfig`; `RunSpec` is removed. Cache keys
  contain the requested configuration and content-tokenized inputs. Fixed and
  progress execution share one controller. Runtime state no longer contains five
  scalar acquisition members that had no accumulation producer; physical DFT flux
  and recorder data remain. Continuation checks actual DFT arena shapes.
- Modes use canonical `Materials` input and array-native `Result` fields. Call
  `solve_grid(material_grid=Materials.from_components(...), ...)`; component kwargs
  belong to the material constructor. Labeled xarray views are lazy public views.
  The profile `solve_modes` helper always returns `(neffs, electric, magnetic, axis)`.
  SciPy lifecycle/PML and CPML polynomial mechanics are shared; distinct diagonal
  and tensor refinement acceptance/fallback policies remain.
- Source compilation uses one flat immutable injection entry and one emitter.
  Slab shapes, waveform quadrature/timing, launch power and native runtime source
  specifications retain their existing contracts.
- Geometry constructors return a frozen `Solid`, shared with native raster scenes.
  Inspect `shape_kind` instead of using constructor functions in `isinstance`.
  Exact primitives, holes, shape-specific transforms, mesh features and terminal
  padding remain explicit. Native boxes retain literal bounds to avoid thin-layer
  rounding changes. Material field validation/grid retention have one owner;
  `PreparedComponent` inherits the prepared imported value instead of proxying it.
- Cross-section rendering uses one entry point, projected device geometry uses one
  axis mapping, and modal analysis shares line-profile and port construction.
  Results plotting uses `plot_field(monitor_name, field, frequency=..., ...)`;
  `field_name`, `field_monitor_name` and `f` aliases are removed. Tracked callers
  and notebooks migrate to the canonical names. Independent numerical references
  remain in the package; the Vulture allowlist identifies their test/dynamic uses.

## Validation

Focused suites passed: 894 broad design/raster/grid/GDS cases, 318 simulation
contracts/invariants, 191 parent analysis/plotting/modal cases, and the device
source/mode/CPML suites. Six stale device assertions migrated from component kwargs
into the canonical material input and passed with unchanged physical assertions.
One new allocation test used a degenerate ARPACK fixture; a guided fixture replaced
it without changing solver fallback or tolerances.

Baseline differentials passed: 1,410 geometry transforms, 360 mesher cases,
128 line profiles, 224 CPML profiles, 20 source emissions, 16 sparse solves and
12 complete mode families. Array equality is exact where reported by those checks.
The broad checks caught and corrected a z-position sentinel regression and native
box-bound rounding before integration.

The full modal-sources notebook passed on CPU with JAX 0.9.0: 12,605 steps,
17 frequencies, seven broadband profiles and a 180×77×56 grid. The normalized
fundamental broadband power across its design band was 0.9999737–1.0001464;
center-frequency flux was 1.0000782. All exported diagnostic arrays were finite.

Whole-package Ruff, formatting, Pyright and Vulture pass. Focused automated suites
use the locked CI environment (JAX 0.10.2); the notebook uses the existing notebook
environment. CPU native-ABI/AD contracts do not establish GPU execution or
performance parity; no GPU run was performed and no native numerical algorithm
or buffer layout was changed.
