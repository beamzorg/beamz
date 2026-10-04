"""Compare periodic material seams with the interior of a 3x3 tiled domain.

The reference uses the same local geometry, cells and initial fields, but its
central tile contains no computational x/y boundary. Eight steps cannot reach
that tile from the remote outer seams. Flat stripes integrate exactly; cylinders
also expose the finite accuracy of adaptive geometry quadrature.
"""

import numpy as np

import beamz as bz
from beamz.design.raster import (
    Cylinder,
    ExtrudedPolygon,
    Grid,
    Object,
    Polygon,
    RasterOptions,
    Scene,
)
from beamz.lattice import component_axis_offsets_3d

NX, NY, NZ = 16, 16, 4
DX, DT = 30e-9, 1e-17


def make_sim(
    tiles, *, dispersive, smoothing, geometry="flat", graded=False, quality="balanced"
):
    medium = (
        bz.PoleResidue.lorentz(
            2, strength=1, resonance=8e15, damping=6e14, frequency_range=(3e14, 9e14)
        )
        if dispersive
        else bz.Material(3)
    )
    length = NX * DX
    widths = (
        DX * (1 + 0.2 * np.sin(2 * np.pi * (np.arange(NX) + 0.25) / NX))
        if graded
        else np.full(NX, DX)
    )
    edges = np.r_[0, np.cumsum(widths)]
    if tiles == 3:
        edges = np.r_[edges[:-1] - length, edges[:-1], edges + length]
    grid = Grid(edges, edges, np.arange(NZ + 1) * DX)
    if geometry == "flat":
        objects = []
        extent = 5 * length
        for k in range(-4, 6):
            lower = k * length + 0.23 * DX
            upper = lower + 0.4 * length
            points = (
                (lower / 2 - extent, lower / 2 + extent),
                (lower / 2 + extent, lower / 2 - extent),
                (upper / 2 + extent, upper / 2 - extent),
                (upper / 2 - extent, upper / 2 + extent),
            )
            objects.append(
                Object(ExtrudedPolygon(Polygon(points), -DX, (NZ + 1) * DX), 1)
            )
    else:
        objects = [
            Object(
                Cylinder(
                    (i * length + 0.2 * DX, j * length + 2.3 * DX),
                    4.7 * DX,
                    -DX,
                    (NZ + 1) * DX,
                ),
                1,
            )
            for i in range(-1, 3)
            for j in range(-1, 3)
        ]
    sim = bz.Simulation(
        scene=Scene((bz.Material(1), medium), tuple(objects)),
        raster_grid=grid,
        raster_options=RasterOptions(smoothing=smoothing, quality=quality),
        boundaries=[bz.Periodic(axes=("x", "y"))],
        time=np.arange(9) * DT,
    )
    state = sim.initial_state()
    fields = {}
    for component in ("Ex", "Ey", "Ez"):
        field = getattr(state, component.lower())
        offsets = component_axis_offsets_3d(component)
        z = (np.arange(field.shape[0]) + offsets["z"])[:, None, None]
        y = (np.arange(field.shape[1]) + offsets["y"])[None, :, None]
        x = (np.arange(field.shape[2]) + offsets["x"])[None, None, :]
        fields[component.lower()] = field.at[:].set(
            0.3 * np.cos(np.pi * z / NZ)
            + np.sin(2 * np.pi * y / NY)
            + 0.2 * np.cos(2 * np.pi * x / NX)
        )
    return sim, state._replace(**fields)


def compare(**settings):
    small, state = make_sim(1, **settings)
    large, tiled_state = make_sim(3, **settings)
    small_program = small.compile(backend="jax")
    large_program = large.compile(backend="jax")
    record = dict(settings, epsilon_max_difference={}, field_max_difference={})
    for axis in "xyz":
        a = np.asarray(getattr(small_program.grid, "eps_" + axis))
        b = np.asarray(getattr(large_program.grid, "eps_" + axis))
        record["epsilon_max_difference"][axis] = float(
            np.max(abs(a - b[:, NY : NY + a.shape[1], NX : NX + a.shape[2]]))
        )
    for steps in (1, 8):
        a = small.advance(
            state=state, num_steps=steps, backend="jax", performance=False
        ).state
        b = large.advance(
            state=tiled_state, num_steps=steps, backend="jax", performance=False
        ).state
        record["field_max_difference"][str(steps)] = {}
        for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
            v = np.asarray(getattr(a, component))
            w = np.asarray(getattr(b, component))[
                :, NY : NY + v.shape[1], NX : NX + v.shape[2]
            ]
            assert np.isfinite(v).all() and np.isfinite(w).all()
            record["field_max_difference"][str(steps)][component] = float(
                np.max(abs(v - w))
            )
    return record
