"""Volumetric, angled-beam inverse design checked against independent solves."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz import GaussianBeamSource
from beamz.optimization import FilterProject, ModePower
from tests.topology_3d_case import make_3d_design, mode_power


def beam_design(pol_angle=0.0, *, pixel_size=0.1e-6):
    base = make_3d_design(run_time=500e-15)
    source = GaussianBeamSource(
        center=(1.5e-6, 1e-6, 1.2e-6),
        size=(1.2e-6, 1.2e-6),
        source_time=base.simulation.sources[0].source_time,
        direction="-z",
        angle_theta=-np.deg2rad(10),
        pol_angle=pol_angle,
        waist_radius=0.45e-6,
        power=0.7,
    )
    region = base.design_region.updated_copy(
        parameterization="volumetric",
        pixel_size=pixel_size,
        transformations=[FilterProject(0.15e-6, beta=2)],
    )
    return base.updated_copy(
        simulation=base.simulation.updated_copy(sources=[source]), design_region=region
    )


@pytest.mark.parametrize("pol_angle", [0.0, np.pi / 2])
def test_beam_voxel_gradients_and_ordinary_modal_parity(pol_angle):
    design = beam_design(pol_angle)
    params = np.random.default_rng(42).uniform(
        0.3, 0.7, design.design_region.params_shape
    )
    fun = design.make_objective_fn(mode_power)
    value, gradient = jax.value_and_grad(fun)(params)
    assert value > 1e-9
    assert np.linalg.norm(gradient) > 1e-7
    direction = gradient / jnp.linalg.norm(gradient)
    fd = (fun(params + 0.003 * direction) - fun(params - 0.003 * direction)) / 0.006
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.008, atol=1e-7)
    adjoint = design.updated_copy(gradient_backend="adjoint")
    adj_value, adj_gradient = jax.value_and_grad(adjoint.make_objective_fn(mode_power))(
        params
    )
    np.testing.assert_allclose(adj_value, value, rtol=3e-5)
    assert np.linalg.norm(gradient - adj_gradient) / np.linalg.norm(gradient) < 0.005
    ordinary = (
        design.to_simulation(params)
        .run(backend="jax", performance=False)
        .mode("output")
    )
    np.testing.assert_allclose(
        design.to_simulation_data(params)["output"].amps.values,
        ordinary.amps.transpose("direction", "f", "mode_index").values,
        rtol=3e-5,
        atol=1e-7,
    )
    patch = np.asarray(design.to_simulation(params)._material_grid().permittivity)[
        design._problem._density_map.slices
    ]
    assert np.linalg.norm(patch[0] - patch[-1]) > 0.01


def test_fine_voxel_grid_interpolation_gradient():
    design = beam_design(pixel_size=0.05e-6)
    params = np.random.default_rng(7).uniform(
        0.3, 0.7, design.design_region.params_shape
    )
    fun = design.make_objective_fn(mode_power)
    _, gradient = jax.value_and_grad(fun)(params)
    assert gradient.shape == (8, 16, 12)
    direction = gradient / jnp.linalg.norm(gradient)
    fd = (fun(params + 0.005 * direction) - fun(params - 0.005 * direction)) / 0.01
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.008, atol=1e-7)


def test_beam_source_must_stay_outside_design():
    design = beam_design()
    source = design.simulation.sources[0].updated_copy(center=(1.5e-6, 1e-6, 0.9e-6))
    with pytest.raises(ValueError, match="two cells away"):
        design.updated_copy(simulation=design.simulation.updated_copy(sources=[source]))


def test_volumetric_export_preserves_different_layers_and_fixed_voxels():
    design = beam_design()
    region = design.design_region
    params = np.zeros(region.params_shape, dtype=np.float32)
    params[0, 2:4, 1:3] = 1
    params[-1, 4:6, 3:5] = 1
    solid = np.zeros_like(params, dtype=bool)
    solid[1, 1, 1] = True
    void = np.zeros_like(solid)
    void[0, 2, 1] = True
    region = region.updated_copy(transformations=[], fixed_solid=solid, fixed_void=void)
    design = design.updated_copy(design_region=region)
    exported = design.export_design(params, mode="pixel_exact")
    patch = np.asarray(exported.rasterize(design.simulation.resolution).permittivity)[
        design._problem._density_map.slices
    ]
    np.testing.assert_allclose(
        patch, 1 + 3 * region.material_density(params), atol=1e-5
    )


def test_coupling_efficiency_does_not_depend_on_beam_power():
    design = beam_design()
    params = design.design_region.initial_parameters
    doubled_source = design.simulation.sources[0].updated_copy(power=1.4)
    doubled = design.updated_copy(
        simulation=design.simulation.updated_copy(sources=[doubled_source])
    )
    objective = ModePower("output")
    original_ce = objective(design.to_simulation_data(params))
    doubled_ce = objective(doubled.to_simulation_data(params))
    np.testing.assert_allclose(doubled_ce, original_ce, rtol=1e-4, atol=1e-10)


def test_beam_source_clearance_accounts_for_wider_stencil():
    design = beam_design()
    source = design.simulation.sources[0].updated_copy(max_shift=3)
    with pytest.raises(ValueError, match="4 cells away"):
        design.updated_copy(simulation=design.simulation.updated_copy(sources=[source]))


def test_single_requested_mode_selects_distinct_polarization_families():
    from beamz import Port
    from beamz.analysis.data import analysis_data
    from beamz.analysis.mode_projection import _build_port_projection

    design = beam_design()
    fractions = []
    for polarization in ("te", "tm"):
        monitor = design.simulation.monitors[0]
        monitor = monitor.updated_copy(
            mode_spec=replace(monitor.mode_spec, polarization=polarization)
        )
        experiment = design.updated_copy(
            simulation=design.simulation.updated_copy(monitors=[monitor])
        )
        port = Port(
            center=monitor.center,
            size=monitor.size,
            name="mode",
            direction="+",
            mode_spec=monitor.mode_spec,
        )
        projection = _build_port_projection(
            analysis_data(experiment._problem._binding_results, "output"),
            port,
            monitor,
            monitor.freqs[0],
            {},
        )
        fields = projection["mode_components"]
        ey, ez = (np.sum(abs(fields[c]) ** 2) for c in ("Ey", "Ez"))
        fractions.append(ey / (ey + ez))
    assert fractions[0] > 0.9
    assert fractions[1] < 0.1


@pytest.mark.parametrize("device", ["source", "monitor"])
def test_beam_and_mode_planes_must_stay_outside_normal_absorber(device):
    design = beam_design()
    sim = design.simulation
    if device == "source":
        source = sim.sources[0].updated_copy(center=(1.5e-6, 1e-6, 1.5e-6))
        sim = sim.updated_copy(sources=[source])
    else:
        monitor = sim.monitors[0].updated_copy(center=(2.9e-6, 1e-6, 0.8e-6))
        sim = sim.updated_copy(monitors=[monitor])
    with pytest.raises(
        ValueError, match="outside the absorber along their normal axis"
    ):
        design.updated_copy(simulation=sim)
