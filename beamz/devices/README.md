# Devices

Simulation devices define how fields enter, leave, and are sampled from a
simulation.

- `modes/`: native finite-difference mode solving, minimal labeled results,
  BeamZ-shaped discrete modes, and guarded Yee-grid refinement.
- `modes/plane.py`: shared finite-plane extraction and solving for source launch
  and modal analysis.
- `modes/specs.py`: immutable mode-selection and mode-data values shared by
  sources, monitors, and ports.
- `sources/specs.py`: immutable public source values.
- `sources/time.py`: sampled and analytic temporal waveforms.
- `sources/solve.py`: source-plane extraction and the 2D native-solver adapter.
- `sources/mode_launch.py`: 2D/3D launch planning, profile conversion, and power scaling.
- `sources/planar_tfsf.py`: discrete 3D total-field/scattered-field residuals.
- `sources/bloch.py`: narrowband oblique surface-current injection on physical Yee supports.
- `sources/compiler.py`: the single lowering boundary into executable source plans.
- `monitors/monitors.py`: immutable public monitor specifications.
- `monitors/compiler.py`: grid placement and packed acquisition plans. Runtime
  accumulation belongs to `simulation.observe`, not to device specifications.
- `ports.py`: named modal port metadata used by S-parameter analysis.
- `boundaries.py`: immutable periodic/Bloch, PEC, PMC, sponge `Absorber`, and PML specifications.
- `_placement.py`: shared grid-snapping rules for sources and monitors.
- `_boundary_compile.py`: grid-aware periodic/PEC/PMC/PML/absorber lowering kept separate from
  the public boundary values for the same reason as source and monitor compilation.
- `_immutable.py`: array freezing and canonicalization shared by every device spec.
- `visualization.py`: data-only visual descriptions consumed by analysis plotting.

These files separate public values, numerical planning, and runtime execution. Avoid
adding per-device facade modules; a new file should own a distinct numerical stage or
be folded into the nearest existing owner.

## 2D mode launch

TM and TE launches orient their E/H pair by signed Poynting flux for each of
`+x`, `-x`, `+y`, and `-y`. The source reference is an H cell centre; the electric
current occupies the adjacent node on the launched side. Its complex profile
includes the half-cell propagation phase and the half-timestep leapfrog phase,
lowered through the waveform's in-phase and quadrature components. TE retains
the mode solver's spatially varying impedance.

The directional regression measures both reflected power on the launched side
and unwanted outgoing power behind the source. It checks both polarizations,
sponge and CPML boundaries, and multiple resolutions/reference-plane placements.
A PEC-terminated guide is the positive control: it must return near-unit guided
power in every direction and polarization, including through the source plane.
