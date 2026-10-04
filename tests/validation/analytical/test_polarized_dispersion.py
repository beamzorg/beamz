"""Constitutive and geometry contracts for causal polarized interfaces."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.dispersive_interfaces import DispersiveInterface
from beamz.lattice import component_shapes
from beamz.simulation.dispersion import (
    compile_dispersion,
    initial_polarization,
    update_dispersion,
)

BAND = (3e14, 9e14)


def material(kind):
    if kind == "static":
        return bz.PoleResidue(3, [(-1e15, 0)], frequency_range=BAND)
    if kind == "drude":
        return bz.PoleResidue.drude(
            1, plasma_frequency=7e15, damping=1e15, frequency_range=BAND
        )
    return bz.PoleResidue.lorentz(
        1, strength=2, resonance=6e15, damping=8e14, frequency_range=BAND
    )


@pytest.mark.parametrize("kind", ["lorentz", "drude", "static"])
@pytest.mark.parametrize("normal", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("second_dispersive", [False, True, "equal"])
def test_coupled_interface_matches_harmonic_transfer_function(
    kind, normal, second_dispersive
):
    media = (
        material(kind),
        material(kind)
        if second_dispersive == "equal"
        else (material("lorentz") if second_dispersive else bz.Material(2)),
    )
    f = np.array([[0.0, 0.1, 0.5, 0.9, 1.0], [1.0, 0.9, 0.5, 0.1, 0.0]])
    eps = np.array([float(m.permittivity) for m in media])[:, None]
    inf = (1 - normal) * np.sum(f * eps, axis=0) + normal / np.sum(f / eps, axis=0)
    shape = (4, 1)
    shapes = component_shapes(shape, "te")
    interface = DispersiveInterface("Ex", np.arange(5), f, np.full(5, normal), media)
    regions = tuple(
        (m, {"Ex": ((1 - normal) * f[j]).reshape(shapes["Ex"])})
        for j, m in enumerate(media)
        if isinstance(m, bz.PoleResidue)
    )
    mg = MaterialGrid(
        np.ones(shape),
        np.zeros(shape),
        np.ones(shape),
        1e-7,
        shape,
        dispersion=regions,
        dispersion_interfaces=(interface,),
        smoothing="farjadpour_diagonal",
        polarization="te",
    )
    grid = SimpleNamespace(
        material_grid=mg,
        geometry=mg.grid,
        component_shapes=shapes,
        eps_x=inf.reshape(shapes["Ex"]),
        sig_x=0.0,
    )
    dt = 1e-17
    plan = compile_dispersion(grid, dt)
    sim = bz.Simulation(material_grid=mg, time=np.arange(2) * dt, polarization="te")
    # Obtain the ordinary Yee state without compiling the deliberately isolated
    # constitutive fixture's material coefficients.
    from beamz.simulation.model import SimulationState

    state = sim.initial_state()._replace(polarization=initial_polarization(plan))
    assert isinstance(state, SimulationState)
    omega = 2 * np.pi * 5e14

    def step(state, k):
        delta = jnp.cos(omega * dt * (k + 1)) - jnp.cos(omega * dt * k)
        free = state._replace(
            ex=state.ex + delta / jnp.asarray(inf.reshape(shapes["Ex"]))
        )
        result = update_dispersion(state, free, plan)
        return result, result.ex.reshape(-1)

    _, trace = jax.jit(lambda s: jax.lax.scan(step, s, jnp.arange(8000)))(state)
    t = np.arange(1, 8001) * dt
    fit = np.c_[np.cos(omega * t[-3000:]), np.sin(omega * t[-3000:]), np.ones(3000)]
    coeff = np.linalg.lstsq(fit, np.asarray(trace)[-3000:], rcond=None)[0]
    measured = coeff[0] + 1j * coeff[1]
    mapped_frequency = np.tan(omega * dt / 2) / (np.pi * dt)
    spectra = np.array(
        [
            m.eps_model(mapped_frequency)
            if isinstance(m, bz.PoleResidue)
            else m.permittivity
            for m in media
        ]
    )[:, None]
    effective = (1 - normal) * np.sum(f * spectra, axis=0) + normal / np.sum(
        f / spectra, axis=0
    )
    np.testing.assert_allclose(measured, 1 / effective, rtol=3e-4, atol=3e-5)
    assert np.isfinite(trace).all()


def interface_sim(smoothing="farjadpour_diagonal", periodic=False):
    from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene

    m = material("lorentz")
    scene = Scene(
        materials=(bz.Material(1), m),
        objects=(Object(Box((0.225e-6, 0.0, 0.0), (0.5e-6, 0.4e-6, 0.1e-6)), 1),),
    )
    return bz.Simulation(
        scene=scene,
        raster_grid=Grid.uniform((0, 0, 0), (0.5e-6, 0.4e-6, 0.1e-6), (10, 8, 1)),
        raster_options=RasterOptions(smoothing=smoothing),
        polarization="te",
        time=np.arange(40) * 1e-17,
        boundaries=[bz.Periodic(axes=("y",))] if periodic else [],
    )


def test_equal_epsilon_infinity_still_has_interface_geometry_and_normal_response():
    sim = interface_sim()
    grid = sim.compile(backend="jax").grid.material_grid
    assert grid.smoothing == "farjadpour_diagonal"
    assert grid.dispersion_interfaces
    assert {i.component for i in grid.dispersion_interfaces} == {"Ex"}
    for i in grid.dispersion_interfaces:
        np.testing.assert_allclose(i.normal_squared, 1)
        np.testing.assert_allclose(i.fractions, 0.5)
    volume = interface_sim("volume").compile(backend="jax").grid.material_grid
    assert not volume.dispersion_interfaces
    assert grid.canonical_spec()[-1]


def test_polarized_interface_continuation_and_periodic_seam():
    sim = interface_sim(periodic=True)
    program = sim.compile(backend="jax")
    assert program.dispersion.interfaces
    state = sim.initial_state()
    state = state._replace(ex=jnp.ones_like(state.ex))
    full = sim.advance(
        state=state, num_steps=30, backend="jax", performance=False
    ).state
    half = sim.advance(
        state=state, num_steps=15, backend="jax", performance=False
    ).state
    resumed = sim.advance(
        state=half, num_steps=15, backend="jax", performance=False
    ).state
    np.testing.assert_allclose(full.ex, resumed.ex, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(full.ex[0], full.ex[-1], rtol=2e-6, atol=1e-7)
    for a, b in zip(full.polarization, resumed.polarization, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    assert (
        sim.memory_estimate()["compiled"]["totals_by_category"]["polarization_state"]
        > 0
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"component": "Hx"}, "electric"),
        ({"indices": [0, 0]}, "unique"),
        ({"indices": [-1, 0]}, "nonnegative"),
        ({"fractions": [[0.2], [0.2]]}, "shapes"),
        ({"fractions": [[0.2, 0.2], [0.2, 0.2]]}, "sum to one"),
        ({"normal_squared": [-0.1, 0.5]}, "normal weights"),
        ({"materials": (bz.Material(1), bz.Material(2))}, "at least one"),
        ({"materials": (material("lorentz"), bz.Material(2, conductivity=1))}, "Ohmic"),
    ],
)
def test_interface_rejects_inconsistent_constituent_data(change, match):
    spec = dict(
        component="Ex",
        indices=[0, 1],
        fractions=[[0.5, 0.5], [0.5, 0.5]],
        normal_squared=[1.0, 1.0],
        materials=(material("lorentz"), bz.Material(1)),
    )
    spec.update(change)
    with pytest.raises(ValueError, match=match):
        DispersiveInterface(**spec)


def test_interface_data_is_immutable_and_cache_tracks_constituents():
    fractions = np.full((2, 2), 0.5)
    interface = DispersiveInterface(
        "Ez", [0, 1], fractions, [1, 1], (material("drude"), bz.Material(1))
    )
    fractions[:] = 0
    np.testing.assert_array_equal(interface.fractions, 0.5)
    with pytest.raises(ValueError):
        interface.fractions[0, 0] = 0
    assert interface.cache_spec()[-1][0] == interface.materials[0].cache_spec()


def test_volume_dispersion_uses_native_yee_epsilon_infinity():
    """The instantaneous response and poles must occupy the same support."""
    from beamz.design.raster import Box, Grid, Object, RasterOptions, Scene

    medium = bz.PoleResidue.lorentz(
        3, strength=2, resonance=6e15, damping=8e14, frequency_range=BAND
    )
    scene = Scene(
        materials=(bz.Material(1), medium),
        objects=(Object(Box((0.0, 0.0, 0.0), (0.225e-6, 0.4e-6, 0.1e-6)), 1),),
    )
    sim = bz.Simulation(
        scene=scene,
        raster_grid=Grid.uniform((0, 0, 0), (0.5e-6, 0.4e-6, 0.1e-6), (10, 8, 1)),
        raster_options=RasterOptions(smoothing="volume"),
        polarization="te",
        time=np.arange(4) * 1e-17,
    )
    grid = sim.compile(backend="jax").grid
    assert grid.material_grid.uses_direct_yee_materials
    for component, fraction in grid.material_grid.dispersion[0][1].items():
        actual = np.asarray(getattr(grid, "eps_" + component[-1].lower()))
        np.testing.assert_allclose(actual, 1 + 2 * fraction, atol=1e-6)


def test_source_rejects_pure_normal_interface_when_bulk_weight_is_zero():
    from beamz.design.raster import Box, Grid, Object, Scene

    grid = Grid(
        np.arange(5) * 100e-9,
        np.arange(5) * 100e-9,
        np.array([0, 100, 210, 320, 430, 540]) * 1e-9,
    )
    # This thin strip has only mixed Ex supports, whose tangential pole weight
    # vanishes. Equal epsilon-infinity also makes the source look homogeneous.
    scene = Scene(
        materials=(bz.Material(1), material("lorentz")),
        objects=(Object(Box((20e-9, 0, 0), (30e-9, 400e-9, 540e-9)), 1),),
    )
    sim = bz.Simulation(
        scene=scene,
        raster_grid=grid,
        time=np.arange(4) * 1e-17,
        sources=[
            bz.PlaneWaveSource(
                center=(200e-9, 200e-9, 210e-9),
                size=(400e-9, 400e-9, 0),
                direction="+z",
                source_time=bz.GaussianPulse(5e14, 1e14),
            )
        ],
    )
    with pytest.raises(ValueError, match="nondispersive"):
        sim.compile(backend="jax")
