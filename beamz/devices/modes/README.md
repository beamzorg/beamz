# Native mode solver

`beamz.devices.modes` owns BeamZ's finite-difference frequency-domain mode
solver. It is shared by mode sources, mode monitors, ports, and direct
rasterized mode analysis; it does not participate in FDTD time stepping.

The high-level BeamZ workflow uses `ModeSpec` on a `ModeSource`, `ModeMonitor`,
or `Port`. For direct solver access, provide an already-rasterized transverse
permittivity grid:

```python
import numpy as np

from beamz import LIGHT_SPEED
from beamz.devices.modes import solve_grid

eps = np.ones((40, 30))
eps[15:25, 12:18] = 3.48**2
result = solve_grid(
    eps_xx=eps,
    x_edges=np.linspace(-2.0, 2.0, eps.shape[0] + 1),
    y_edges=np.linspace(-1.5, 1.5, eps.shape[1] + 1),
    freqs=[LIGHT_SPEED / 1.55e-6],
    num_modes=2,
    target_neff=3.0,
)
```

Coordinates passed to the direct raster API are measured in micrometres and
frequencies are measured in hertz. The result contains labeled effective
indices, field components, and solver diagnostics; BeamZ analysis owns plotting
and persistence.

For a uniformly bent cross-section, add `bend_radius=10.0` (micrometres) and
optionally `bend_axis="x"` or `"y"` (local radial coordinate). Radius is measured
from coordinate zero; positive radius puts the curvature center at `-R`.
`bend_radius=None` preserves straight solves. See
[circular-bend modes](../../../docs/bent-waveguide-modes.md) for the material
assumptions, field conventions, validation results, and limitations, and
[the runnable example](../../../examples/bent_waveguide_modes.py).

The discrete launch path converts solved modes onto BeamZ's component-specific
Yee supports, normalizes their signed power, and applies guarded refinement
only when field overlap, impedance, energy, power, and discrete-Maxwell
validation all accept the candidate.

On uniform grids, y-directed ports are reflected into the x-normal frame for
the same discrete solve and refinement, then transformed back with the axial
magnetic-field sign. Sources and modal analysis share these physical profiles;
this prevents port orientation from changing the mode contract.

`plane.py` owns the shared finite-plane extraction used by source launch and
modal projection, so both consumers solve identical material crops and Yee
supports.

See `UPSTREAM.md` for the imported MicroMode revision and source mapping.

## Optional mode symmetry

`ModeSpec(force_symmetry=False)` is the default for previews, sources, mode
monitors, and port projections. No mirror-parity projection is performed: a
solved off-center field is preserved, independent of approximate material
correlation or sampling thresholds. To request projection explicitly, use
`ModeSpec(force_symmetry=True)` on the source or monitor (or the port used to
create it).

Projection acts on the solver's collocated cell-center fields **before** Yee
interpolation, aperture windowing, guarded Maxwell refinement, and discrete
unit-power normalization. It uses a common vector mirror parity, with electric
fields treated as polar vectors and magnetic fields as axial vectors. It never
reverses a compact staggered field array or modifies an accepted refinement.
The previous approximate correlation detector and source-only post-refinement
projection have been removed.

Each transverse axis is checked independently. The physical mirror center is
the requested plane center; direct internal `solve_modes` calls use the aperture
midpoint unless `mirror_center` is provided in metres and profile storage order.
All grid edges, all component coordinates, and the complex diagonal epsilon/mu
profiles must be mirror symmetric about that center, to numerical tolerance.
Discrete solves also check the constitutive arrays on their individual Yee
supports, using edge or center coordinates according to each component's
stagger. A symmetric graded mesh is supported; an asymmetric graded mesh,
clipped/off-center crop, asymmetric tensor, or unavailable/non-closed component
support is skipped. There is no approximate mirror interpolation. Full-tensor
ports use the separate coupled path described below and reject forced symmetry.

`ModeSource.solve_modes()` defaults to solver fields, including optional
validated symmetry, before Yee interpolation, Maxwell refinement, and launch
power normalization. It retains the eigensolver's sampling and normalization.
For a colocated view of the prepared 3D fields used for injection, request
`source.solve_modes(simulation, field_representation="launch")`. This choice
affects the returned preview only; injection always uses prepared launch fields.
Prepared launch previews are available for 3D simulations.
`field_representation` and `processing_stage` in the returned diagnostics identify
the view. With no surrounding solve cells, Yee interpolation can omit an outer
sample from the launch view; the solver view retains the complete solve domain.
`ModeData.diagnostics` contains one immutable mapping per
frequency; `DiscreteMode.diagnostics` records the discrete processing result.
Both report `force_symmetry_requested`, `force_symmetry_applied`,
`force_symmetry_axes`, `force_symmetry_skipped`, and `symmetry_stage`. Refinement
and final power normalization remain independent of this option; an accepted
refinement need not retain exact seed parity. Raw `solve_grid` results remain
unprojected. One-dimensional line profiles used for 2D launches also remain
unprojected, with an explicit skip diagnostic when symmetry is requested.

## Full-tensor dielectric ports

Lossless real SPD, nonmagnetic 3D tensor ports use a coupled discrete operator
shared with propagation. Set `ModeSpec(polarization=None, force_symmetry=False)`.
Uniform and rectilinear grids are supported; normal spacing must be locally
uniform and the material longitudinally invariant. Unsupported combinations fail
explicitly. Source launch and monitor projection retain dielectric cross terms.

## 3D mode selection

Previews, source launch and modal projection share material extraction and mode
selection. Source launch, modal projection, and explicit launch previews also
share discrete field preparation. The aperture defines injection/integration support.
`ModeSpec(min_aperture_fraction=0.5)` prefers modes with at least half their
area-weighted transverse electric intensity inside that aperture, retaining
polarization/neff ordering. Heterogeneous planes search up to 64 candidates
(or the initial pool if larger), then fail if the requested confined mode is
unavailable. This fraction is a spatial diagnostic, not modal power or energy.
Set it to `None` for unrestricted radiation or coupled-guide mode inspection.

Resolve the complete requested family before assigning indices, and share the
final candidate pool per preview frequency. This prevents duplicate selections
and repeated eigensolves while preserving per-mode field preparation.

## Source apertures and solve domains

Three-dimensional mode sources, previews, and mode monitors solve on the requested
plane, snapped outward to the material grid. The default adds no transverse solve
padding and applies no aperture taper. Additional cells retained for component
sampling are not a request to solve on the entire simulation cross-section.

Place sources in a straight, longitudinally invariant section and make the plane
large enough for the desired guided field to decay at its edges. The solver logs
a warning when the raw electric edge-intensity sum exceeds one percent of the
whole-plane sum. `edge_intensity_fraction` and `fields_decay_at_boundary` retain
this sampling diagnostic; it is not a modal power or confinement fraction.
Warnings do not resize the solve domain or suppress radiation candidates.

`ModeSpec(solver_padding_cells=4)` explicitly adds four cells on each transverse
side. A pair specifies `(width, height)` padding: `(y, z)` for x-normal planes,
`(x, z)` for y-normal planes, and `(x, y)` for z-normal planes. `None` and zero
both mean no extra padding. Counts use actual cells on graded grids and clip to
the simulation boundary. `solver_bounds` records the actual solve region.

A padded solve still uses the declared aperture to select injection/integration
support, including component-sampling margins. Fields outside that support must
be negligible for clean directional injection. The default solver preview shows
the complete solve domain, including explicitly requested padding. Launch previews
show the prepared injection support. `min_aperture_fraction` is now opt-in; its default `None`
preserves polarization/effective-index candidate ordering.
