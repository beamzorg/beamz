"""3D SOI straight-guide controls for source and modal normalization."""

import numpy as np

from beamz import (
    LIGHT_SPEED,
    Absorber,
    Design,
    FieldMonitor,
    GaussianPulse,
    GridSpec,
    Material,
    ModeSpec,
    Port,
    Rectangle,
    Simulation,
    µm,
)
from beamz.design.raster import RasterOptions
from tests.differential.passive_soi.common import load_passive_soi_case
from tests.differential.passive_soi.experiments import (
    DEFAULT_OPTIONS,
    ExperimentOptions,
)


def build_straight_control(
    name, *, resolution_ppw=6, options: ExperimentOptions = DEFAULT_OPTIONS
):
    """Use the benchmark stack and source protocol on an independently meshed guide.

    Three output planes measure spatial variation without changing the grid.
    They are alternative measurements, never summed as distinct output ports.
    """
    widths = {"te0": 0.5, "wide_te0": 1.2, "te1": 1.2, "tm0": 0.405}
    if name not in widths:
        raise ValueError(f"unknown straight-guide control {name!r}")
    case = load_passive_soi_case(
        "polarization_splitter_rotator" if name == "tm0" else "mode_converter"
    )
    silicon = Material(case.materials["silicon_n_at_1p55_um"] ** 2)
    silica = Material(case.materials["silica_n_at_1p55_um"] ** 2)
    design = Design(width=12 * µm, height=7.5 * µm, depth=4 * µm, background=silica)
    if name == "tm0":
        design += Rectangle(
            position=(0, 0, 2 * µm),
            width=12 * µm,
            height=7.5 * µm,
            depth=2 * µm,
            material=Material(4),
        )
    width = widths[name] * µm
    design += Rectangle(
        position=(0, 3.75 * µm - width / 2, 2 * µm),
        width=12 * µm,
        height=width,
        depth=0.22 * µm,
        material=silicon,
    )
    frequencies = options.frequencies(case, 20)
    spec = ModeSpec(
        polarization="tm" if name == "tm0" else "te",
        mode_index=1 if name == "te1" else 0,
        num_modes=5,
    )
    span = (options.transverse_span_um or 3) * µm
    size = (0, span, 2 * µm)
    source_time = GaussianPulse(
        freq0=LIGHT_SPEED / (1.55 * µm),
        fwidth=float(np.ptp(frequencies)),
        offset=5 / (2 * np.pi),
    )
    source = Port(
        center=(1.5 * µm, 3.75 * µm, 2.11 * µm),
        size=size,
        name="source",
        direction="+",
        mode_spec=spec,
    ).to_source(
        freq0=source_time.freq0,
        fwidth=source_time.fwidth,
        num_freqs=options.source_profiles or 3,
        source_time=source_time,
    )
    offset = options.monitor_offset_um or 0
    ports = tuple(
        Port(
            center=((x + offset) * µm, 3.75 * µm, 2.11 * µm),
            size=size,
            name=label,
            direction="+" if label == "o1" else "-",
            mode_spec=spec,
        )
        for label, x in [("o1", 2), ("near", 8), ("middle", 9), ("far", 10)]
    )
    monitors = [p.to_monitor(frequencies) for p in ports]
    monitors.append(
        FieldMonitor(
            center=(6 * µm, 3.75 * µm, 2.11 * µm),
            size=(12 * µm, 7.5 * µm, 0),
            freqs=(source_time.freq0,),
            fields=("Ex", "Ey", "Ez"),
            name="straight_xy",
        )
    )
    simulation = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[
            Absorber(edges="all", thickness=options.boundary_thickness_um * µm)
        ],
        run_time=(options.run_time_ps or 6.4) * 1e-12,
        grid_spec=GridSpec.auto(
            min_steps_per_wvl=resolution_ppw,
            wavelength=1.55 * µm,
            courant=0.99,
            max_scale=1.4,
            max_total_cells=None,
        ),
        raster_options=RasterOptions(quality="balanced", smoothing=options.smoothing),
    )
    return simulation, ports, ("near", "middle", "far"), frequencies


def build_converter_grid_control(name, *, resolution_ppw=6, options=DEFAULT_OPTIONS):
    """Extrude one straight port on the converter's exact realized grid."""
    from beamz.design.discretization import build_material_grid
    from tests.differential.passive_soi.mode_conversion import (
        build_mode_conversion_simulation,
    )

    original, original_ports, _, frequencies = build_mode_conversion_simulation(
        "mode_converter", resolution_ppw=resolution_ppw, options=options
    )
    if name not in ("te0", "te1"):
        raise ValueError("converter grid control must be te0 or te1")
    template = next(
        p for p in original_ports if p.name == ("o1" if name == "te0" else "conversion")
    )
    width = (0.5 if name == "te0" else 1.2) * µm
    center_y, center_z = template.center[1:]
    case = load_passive_soi_case("mode_converter")
    design = Design(
        width=original.design.width,
        height=original.design.height,
        depth=original.design.depth,
        background=Material(case.materials["silica_n_at_1p55_um"] ** 2),
    )
    design += Rectangle(
        position=(0, center_y - width / 2, 2 * µm),
        width=design.width,
        height=width,
        depth=0.22 * µm,
        material=Material(case.materials["silicon_n_at_1p55_um"] ** 2),
    )
    material = build_material_grid(
        design, original.grid, quality="balanced", smoothing=options.smoothing
    )
    mode = ModeSpec(
        polarization="te", mode_index=0 if name == "te0" else 1, num_modes=5
    )
    source_x = original.sources[0].center[0] if name == "te0" else 1.5 * µm
    source_time = original.sources[0].source_time
    source = Port(
        center=(source_x, center_y, center_z),
        size=template.size,
        name="source",
        direction="+",
        mode_spec=mode,
    ).to_source(
        freq0=source_time.freq0,
        fwidth=source_time.fwidth,
        num_freqs=options.source_profiles or 3,
        source_time=source_time,
    )
    ports = tuple(
        Port(
            center=(x, center_y, center_z),
            size=template.size,
            name=label,
            direction="+" if label == "o1" else "-",
            mode_spec=mode,
        )
        for label, x in [
            ("o1", source_x + 0.5 * µm),
            ("near", source_x + 2 * µm),
            ("middle", 0.5 * design.width),
            ("far", design.width - 1.5 * µm),
        ]
    )
    monitors = [p.to_monitor(frequencies) for p in ports]
    monitors.append(
        FieldMonitor(
            center=(0.5 * design.width, center_y, center_z),
            size=(design.width, design.height, 0),
            freqs=(source_time.freq0,),
            fields=("Ex", "Ey", "Ez"),
            name="straight_xy",
        )
    )
    simulation = Simulation(
        material_grid=material,
        sources=[source],
        monitors=monitors,
        boundaries=original.boundaries,
        run_time=original.run_time,
    )
    return simulation, ports, ("near", "middle", "far"), frequencies


def build_ring_bus_control(*, resolution_ppw=6, options=DEFAULT_OPTIONS):
    """Remove the ring while preserving its bus, source, ports, grid, and absorber."""
    from beamz.design.discretization import build_material_grid
    from tests.differential.passive_soi.ring_resonator import (
        build_ring_resonator_simulation,
    )

    original, ports, frequencies = build_ring_resonator_simulation(
        resolution_ppw=resolution_ppw, options=options, diagnostics=True
    )
    case = load_passive_soi_case("ring_resonator")
    design = Design(
        width=original.design.width,
        height=original.design.height,
        depth=original.design.depth,
        background=Material(case.materials["silica_n_at_1p55_um"] ** 2),
    )
    design += Rectangle(
        position=(0, original.sources[0].center[1] - 0.25 * µm, 2 * µm),
        width=design.width,
        height=0.5 * µm,
        depth=0.22 * µm,
        material=Material(case.materials["silicon_n_at_1p55_um"] ** 2),
    )
    material = build_material_grid(
        design, original.grid, quality="balanced", smoothing=options.smoothing
    )
    simulation = Simulation(
        material_grid=material,
        sources=original.sources,
        monitors=original.monitors,
        boundaries=original.boundaries,
        run_time=original.run_time,
    )
    return simulation, ports, ("o2",), frequencies
