"""Multi-material topology design: K >= 2 materials, nested density channels."""

import numpy as np
import pytest

import beamz as bz
from beamz._cache_tokens import cache_token
from beamz.constitutive import expand_packed
from beamz.design.raster.engine import RasterOptions
from beamz.optimization import TopologyProblem, TopologySpec
from beamz.optimization.anisotropic_design import material_weights
from tests.topology_case import make_problem

pytestmark = pytest.mark.optimization

OMEGA = 2 * np.pi * bz.LIGHT_SPEED / 1.55e-6


def _crystal(tilt_eps_inf=True):
    pole = bz.TensorPole.lorentz((0.8, 0.3, 0.0), resonance=2.2 * OMEGA, damping=0.0)
    if tilt_eps_inf:
        return bz.AnisotropicMedium((4.0, 3.0, 3.5), electric_poles=[pole]).rotated(
            (0.0, 0.0, 0.4)
        )
    # Diagonal high-frequency tensor with a tilted pole.
    return bz.AnisotropicMedium(
        (4.0, 3.0, 3.5), electric_poles=[pole.rotated((0.0, 0.0, 0.4))]
    )


def _problem(materials, *, mask=None, simulation=None, **options):
    base = make_problem(polarization="te", reference=False)
    spec = base.topology
    topology = TopologySpec(
        design=spec.design,
        region_mask=spec.region_mask if mask is None else mask,
        resolution=spec.resolution,
        filter_radius=options.pop("filter_radius", spec.filter_radius),
        learning_rate=spec.learning_rate,
        beta_schedule=spec.beta_schedule,
        materials=materials,
        **options,
    )
    return TopologyProblem(
        base.simulation if simulation is None else simulation,
        topology,
        base.objective,
        checkpoint_interval=base.checkpoint_interval,
    )


def _amplitude(results):
    return abs(results.mode("output").amps.sel(direction="+", mode_index=1).item()) ** 2


def test_weights_form_a_partition_and_reduce_to_two_materials():
    rng = np.random.default_rng(0)
    rho = rng.uniform(size=(3, 5, 4))
    weights = material_weights(rho, 4)
    assert len(weights) == 4
    assert min(np.min(w) for w in weights) >= 0.0
    np.testing.assert_allclose(sum(weights), 1.0, rtol=0, atol=1e-15)
    binary = (rho > 0.5).astype(float)
    selected = np.stack(material_weights(binary, 4))
    np.testing.assert_array_equal(np.sum(selected == 1.0, axis=0), 1)
    low, high = material_weights(rho[0], 2)
    np.testing.assert_array_equal(low, 1.0 - rho[0])
    np.testing.assert_array_equal(high, rho[0])


def test_materials_pair_is_the_material_min_max_shorthand():
    base = make_problem(polarization="te", reference=False).topology
    common = dict(
        design=base.design, region_mask=base.region_mask, resolution=base.resolution
    )
    crystal = _crystal()
    pair = TopologySpec(materials=(bz.Material(1.0), crystal), **common)
    shorthand = TopologySpec(
        material_min=bz.Material(1.0), material_max=crystal, **common
    )
    assert cache_token(pair) == cache_token(shorthand)
    assert pair.num_materials == 2
    assert pair.density_shape == base.region_mask.shape
    triple = TopologySpec(
        materials=(bz.Material(1.0), bz.Material(2.1), crystal), **common
    )
    assert triple.density_shape == (2, *base.region_mask.shape)
    assert triple.eps_min == 1.0 and triple.eps_max == pytest.approx(3.5)
    assert triple.initial_state().density.shape == triple.density_shape
    from dataclasses import replace

    assert replace(triple, learning_rate=0.2).materials == triple.materials
    with pytest.raises(ValueError, match="not both"):
        TopologySpec(
            materials=(bz.Material(1.0), crystal),
            material_max=bz.Material(2.0),
            **common,
        )
    with pytest.raises(ValueError, match="at least two"):
        TopologySpec(materials=(crystal,), **common)
    with pytest.raises(ValueError, match="scalar permittivity gradient"):
        triple.apply_gradient(
            triple.initial_state(), np.zeros(base.region_mask.shape), 1.0
        )


def test_three_material_gradient_and_ordinary_run_agree():
    """Air, a static tilted tensor and a tilted pole crystal share one region.

    The bound objective must equal an ordinary simulation of the same blended
    material map, and its autodiff gradient must match central finite
    differences and forward mode.
    """
    import jax
    import jax.numpy as jnp

    tilted = bz.AnisotropicMedium((2.5, 2.0, 2.2)).rotated((0.0, 0.0, -0.3))
    problem = _problem((bz.Material(1.0), tilted, _crystal()))
    plan = problem.program.dispersion.anisotropic
    assert problem.program.coefficients.e_inverse_offdiagonal.size
    assert plan is not None and plan.electric_regions
    mask = problem.topology.region_mask
    rng = np.random.default_rng(10)
    ramp = 0.25 * np.linspace(-1, 1, mask.shape[0])[:, None]
    density = np.stack(
        [
            (0.6 + ramp + 0.05 * rng.normal(size=mask.shape)) * mask,
            (0.5 - ramp + 0.05 * rng.normal(size=mask.shape)) * mask,
        ]
    )
    value, gradient = problem.value_and_grad(density, beta=2)
    assert value > 1e-4
    assert gradient.shape == density.shape
    assert np.all(np.abs(gradient[:, mask]).sum(axis=1) > 0)
    ordinary = problem.material_simulation(density, beta=2).run(
        backend="jax", performance=False
    )
    np.testing.assert_allclose(value, _amplitude(ordinary), rtol=1e-5)
    direction = np.random.default_rng(5).normal(size=density.shape) * mask
    direction += 0.1 * np.linalg.norm(direction) * gradient / np.linalg.norm(gradient)
    direction /= np.linalg.norm(direction)
    adjoint = float(np.sum(gradient * direction))
    h = 0.01
    difference = (
        problem.value(density + h * direction, beta=2)
        - problem.value(density - h * direction, beta=2)
    ) / (2 * h)
    np.testing.assert_allclose(difference, adjoint, rtol=1e-3)
    latent = problem._density(density)
    beta = problem._beta(2.0)
    _, forward = jax.jvp(
        lambda z: problem._evaluate(z, beta),
        (latent,),
        (jnp.asarray(direction, dtype=latent.dtype),),
    )
    np.testing.assert_allclose(adjoint, float(forward), rtol=1e-4)


def test_binarized_design_reproduces_static_multi_material_layout():
    """Binary channels select one material per cell, like painted geometry.

    Four materials (air, a scalar oxide, a static tensor and a pole crystal)
    are painted as pixel-aligned rectangles. Volume smoothing samples Yee
    sites like the design binding, and air surrounds the mask boundary, so the
    material grid and the simulated transmission must match the bound design.
    """
    oxide = bz.Material(2.1)
    tensor = bz.AnisotropicMedium((5.0, 4.0, 4.5))
    crystal = _crystal(tilt_eps_inf=False)
    materials = (bz.Material(1.0), oxide, tensor, crystal)
    base = make_problem(polarization="te", reference=False)
    simulation = base.simulation.updated_copy(
        raster_options=RasterOptions(smoothing="volume")
    )
    dx = base.topology.resolution
    y, x = np.indices(base.topology.region_mask.shape) * dx
    eps = 1e-15

    def box(x0, x1, y0, y1):
        return (x >= x0 - eps) & (x < x1 - eps) & (y >= y0 - eps) & (y < y1 - eps)

    # The region stays one air cell clear of the fixed guides.
    mask = box(2.1e-6, 3.9e-6, 1.2e-6, 2.8e-6)
    blocks = [
        (1, (2.2e-6, 1.4e-6, 0.4e-6, 1.2e-6)),
        (2, (2.6e-6, 1.6e-6, 0.8e-6, 0.4e-6)),
        (3, (2.6e-6, 2.0e-6, 0.8e-6, 0.4e-6)),
        (2, (3.4e-6, 1.3e-6, 0.4e-6, 0.6e-6)),
        (1, (3.4e-6, 2.1e-6, 0.4e-6, 0.5e-6)),
    ]
    index = np.zeros(mask.shape, dtype=int)
    design = simulation.design
    for material, (x0, y0, w, h) in blocks:
        index[box(x0, x0 + w, y0, y0 + h)] = material
        design += bz.Rectangle(
            position=(x0, y0), width=w, height=h, material=materials[material]
        )
    assert set(np.unique(index[mask])) == {0, 1, 2, 3}
    problem = _problem(
        materials,
        mask=mask,
        simulation=simulation,
        filter_radius=0.0,
        projection_type="identity",
    )
    channels = np.stack([index >= k for k in (1, 2, 3)]).astype(float) * mask
    static = simulation.updated_copy(design=design)
    expected = static._material_grid()
    bound = problem.material_simulation(channels).material_grid
    np.testing.assert_allclose(
        bound.tensors["epsilon"], expand_packed(expected.tensors["epsilon"]), atol=1e-6
    )
    for name in ("eps_x", "eps_y"):
        np.testing.assert_allclose(
            bound.yee_materials[name], expected.yee_materials[name], atol=1e-6
        )
    assert len(bound.tensor_dispersion) == len(expected.tensor_dispersion) == 1
    (medium, fraction), (reference, reference_fraction) = (
        bound.tensor_dispersion[0],
        expected.tensor_dispersion[0],
    )
    assert medium.cache_spec() == reference.cache_spec()
    np.testing.assert_array_equal(fraction, reference_fraction)
    value = problem.value(channels)
    static_value = _amplitude(static.run(backend="jax", performance=False))
    assert value > 1e-4
    np.testing.assert_allclose(value, static_value, rtol=1e-4)


def test_multi_material_export_paints_each_material():
    tensor = bz.AnisotropicMedium((5.0, 4.0, 4.5))
    problem = _problem(
        (bz.Material(1.0), bz.Material(2.1), tensor),
        filter_radius=0.0,
        projection_type="identity",
    )
    mask = problem.topology.region_mask
    columns = np.indices(mask.shape)[1]
    index = np.where(columns < 30, 1, 2) * mask
    channels = np.stack([index >= 1, index >= 2]).astype(float) * mask
    result = problem.run(steps=1, initial_density=channels, stop_after=0)
    design = problem.export_design(result, mode="pixel_exact")
    grid = design.rasterize(problem.topology.resolution)
    cells = np.asarray(grid.permittivity)
    np.testing.assert_allclose(cells[mask & (index == 1)], 2.1)
    np.testing.assert_allclose(cells[mask & (index == 2)], 4.5, rtol=1e-6)
