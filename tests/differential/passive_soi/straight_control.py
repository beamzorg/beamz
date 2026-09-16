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
