"""Independent Maxwell curl checks through public material compilation."""

import numpy as np
import pytest

import beamz as bz
from beamz.const import MU_0
from beamz.design.discretization import MaterialGrid


@pytest.mark.parametrize("coupled", [False, True])
def test_tensor_pole_preparation_reuses_full_inverses(monkeypatch, coupled):
    from beamz.design.raster import RasterOptions
    from beamz.simulation import anisotropic

    inverse_calls = []
    original = anisotropic.lossy_tensor_inverse

    def count_inverse(support, shared_tensor, losses, shared):
        inverse_calls.append(shared)
        return original(support, shared_tensor, losses, shared)

    monkeypatch.setattr(anisotropic, "lossy_tensor_inverse", count_inverse)
    poles = (
        bz.TensorPole.lorentz(0.25, resonance=1e15, damping=0.0),
        bz.GyrotropicPole.lorentz(
            0.25, resonance=1e15, damping=0.0, gyration=(0, 0, 1e14)
        ),
    )
    medium = bz.AnisotropicMedium(
        permittivity=(2, 2, 2, 0.1, 0, 0),
        permeability=(2, 3, 4, 0.1 if coupled else 0.0, 0, 0),
        electric_poles=poles,
        magnetic_poles=poles,
    )
    sim = bz.Simulation(
        domain=(400e-9,) * 3,
        background=medium,
        resolution=50e-9,
        raster_options=RasterOptions(smoothing="farjadpour_full"),
        time=np.arange(3) * 1e-17,
    )
    program = sim.compile(backend="jax")
    plan = program.dispersion.anisotropic
    assert len(plan.electric_regions) == len(plan.magnetic_regions) == 2
    assert plan.electric_regions[0].inverse is plan.electric_regions[1].inverse
    assert plan.magnetic_regions[0].inverse is plan.magnetic_regions[1].inverse
    if coupled:
        assert plan.magnetic is not None
        assert inverse_calls == ["Node", "Cell"]
    else:
        assert plan.magnetic is None
        assert inverse_calls == ["Node"]
        for region in plan.magnetic_regions:
            for inverse, mu in zip(region.inverse.diagonal, (2, 3, 4), strict=True):
                np.testing.assert_allclose(inverse, 1 / mu)
        differentiable = sim.differentiable(backend="jax")
        differentiated = differentiable.coefficients()
        for axis, mu in zip("xyz", (2, 3, 4), strict=True):
            expected = program.config.dt / (MU_0 * mu)
            for coefficients in (program.coefficients, differentiated):
                np.testing.assert_allclose(
                    getattr(coefficients, f"h_source_{axis}"),
                    expected,
                    rtol=1e-6,
                    atol=0,
                )


@pytest.mark.parametrize("cpml", [False, True])
@pytest.mark.parametrize("mu", [0.5, 1.0, 2.0, "interface"])
def test_public_magnetic_material_scales_one_step_curl(cpml, mu):
    shape = (32, 33, 34)
    permeability = (
        np.full(shape, 2, np.float32) if mu == "interface" else np.float32(mu)
    )
    if mu == "interface":
        permeability[:, :16] = 1
    grid = MaterialGrid(
        np.ones(shape, np.float32), np.float32(0), permeability, 20e-9, shape
    )
    simulation = bz.Simulation(
        material_grid=grid,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")] if cpml else [bz.PEC()],
    )
    state = simulation.initial_state()
    state = state._replace(
        ex=np.broadcast_to(
            np.arange(state.ex.shape[1], dtype=np.float32)[None, :, None],
            state.ex.shape,
        )
    )
    try:
        result = simulation.step(state, backend="jax")
        for y in (13, 19):
            material = (
                float(permeability[16, y, 16]) if mu == "interface" else float(mu)
            )
            expected = np.float32(1e-17 / (MU_0 * material * 20e-9))
            np.testing.assert_allclose(
                np.asarray(result.hz)[16, y, 16], expected, rtol=3e-6, atol=0
            )
    finally:
        simulation.clear_compiled_cache()


@pytest.mark.parametrize("cpml", [False, True])
def test_volume_raster_preserves_diagonal_magnetic_supports(cpml: bool) -> None:
    from beamz.design.raster import Grid, Material, RasterOptions, Scene, rasterize

    edges = np.arange(33) * 20e-9
    raster = rasterize(
        Scene((Material(epsilon_r=2, mu_r=(1, 2, 3)),)),
        Grid(edges, edges, edges),
        options=RasterOptions(smoothing="volume"),
    )
    grid = MaterialGrid.from_raster_result(raster, dimensions=3)
    simulation = bz.Simulation(
        material_grid=grid,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PML(formulation="cpml")] if cpml else [bz.PEC()],
    )
    state = simulation.initial_state()
    state = state._replace(
        ex=np.broadcast_to(
            np.arange(state.ex.shape[1], dtype=np.float32)[None, :, None],
            state.ex.shape,
        )
    )
    try:
        result = simulation.step(state, backend="jax")
        expected = np.float32(1e-17 / (MU_0 * 3 * 20e-9))
        np.testing.assert_allclose(
            np.asarray(result.hz)[16, 16, 16], expected, rtol=3e-6, atol=0
        )
    finally:
        simulation.clear_compiled_cache()


def test_cpml_preserves_broadcast_storage_for_graded_magnetic_layers():
    from beamz.devices._boundary_compile import lower_boundaries
    from beamz.lattice import component_shapes
    from beamz.simulation.compile import _permeable_h_coefficients

    shape = (97, 41, 33)
    dx, dt = 20e-9, 1e-17
    layers = np.linspace(1.0, 2.0, shape[0], dtype=np.float32)[:, None, None]
    material = np.broadcast_to(layers, shape)
    grid = MaterialGrid(material, 0.0, material, dx, shape)
    b = lower_boundaries(
        grid,
        component_shapes(shape),
        [bz.PML(formulation="cpml", thickness=12 * dx)],
        grid.grid.extent,
        dt,
        region_setup=True,
    )
    dense = layers.copy()
    dense[:12] = dense[12]
    dense[-12:] = dense[-13]
    for array in (b.permittivity, b.permeability):
        assert array.strides[1:] == (0, 0)
        np.testing.assert_array_equal(array[:, :1, :1], dense)
    decay, source = _permeable_h_coefficients(np.float32(0), b.permeability, dt)
    assert source.strides[1:] == (0, 0)
    np.testing.assert_array_equal(decay, 1.0)
    np.testing.assert_allclose(
        source[:, :1, :1], np.float32(dt / MU_0) / dense, rtol=3e-7
    )
