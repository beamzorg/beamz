"""Small vector-FDTD fixture for extruded topology derivatives."""

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianPulse,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Rectangle,
    Simulation,
)
from beamz.optimization import FilterProject, InverseDesign, TopologyDesignRegion


def make_3d_design(
    backend="autodiff", *, axis="x", polarization="te", direction="+", run_time=500e-15
):
    dx, frequency = 0.1e-6, LIGHT_SPEED / 1e-6
    order = {"x": (0, 1, 2), "y": (1, 0, 2), "z": (2, 1, 0)}[axis]

    def rotate(values):
        return tuple(values[i] for i in order)

    width, height, depth = rotate((3e-6, 2e-6, 1.6e-6))
    design = Design(width=width, height=height, depth=depth, material=Material(1))
    for x in (0, 1.8e-6):
        w, h, d = rotate((1.2e-6, 0.6e-6, 0.4e-6))
        design += Rectangle(
            position=rotate((x, 0.7e-6, 0.6e-6)),
            width=w,
            height=h,
            depth=d,
            material=Material(4),
        )
    modes = ModeSpec(num_modes=1, polarization=polarization)
    source = ModeSource(
        center=rotate((0.6e-6 if direction == "+" else 2.4e-6, 1e-6, 0.8e-6)),
        size=rotate((0, 1.4e-6, 1.2e-6)),
        direction=direction,
        mode_spec=modes,
        source_time=GaussianPulse(freq0=frequency, fwidth=0.1 * frequency, offset=2),
    )
    output = ModeMonitor(
        center=rotate((2.4e-6 if direction == "+" else 0.6e-6, 1e-6, 0.8e-6)),
        size=source.size,
        mode_spec=modes,
        freqs=[frequency],
        name="output",
    )
    sim = Simulation(
        design=design,
        resolution=dx,
        run_time=run_time,
        sources=[source],
        monitors=[output],
        boundaries=[PML(thickness=0.3e-6, formulation="cpml")],
    )
    region = TopologyDesignRegion(
        center=rotate((1.5e-6, 1e-6, 0.8e-6)),
        size=rotate((0.6e-6, 0.8e-6, 0.4e-6)),
        eps_bounds=(1, 4),
        pixel_size=dx,
        transformations=[FilterProject(0.15e-6, beta=2)],
    )
    return InverseDesign(
        simulation=sim,
        design_region=region,
        gradient_backend=backend,
        checkpoint_interval=64,
    )


def mode_power(data):
    return abs(data["output"].amps.sel(direction="+", mode_index=0).values[0]) ** 2
