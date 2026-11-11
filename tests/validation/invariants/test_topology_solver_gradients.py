"""Full electromagnetic derivatives, including CPML, Yee sampling and modal DFTs."""

import numpy as np
import pytest

from beamz.optimization import TopologyProblem
from tests.topology_case import make_problem

pytestmark = pytest.mark.optimization


def initial_density(problem):
    density = problem.topology.initial_state().density.copy()
    mask = problem.topology.region_mask
    density[mask] += 0.05 * np.random.default_rng(10).normal(size=mask.sum())
    # Explicitly excite the odd target mode. Symmetry-preserving discretization
    # correctly suppresses it for a uniform design; small filtered noise alone
    # gives a weak signal and cancellation-dominated finite differences.
    density += 0.25 * np.linspace(-1, 1, density.shape[0])[:, None] * mask
    return density


@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("reference", [False, True])
def test_full_solver_gradient_matches_finite_difference(
    polarization, reference, validation_metrics
):
    problem = make_problem(polarization=polarization, reference=reference)
    density = initial_density(problem)
    value, gradient = problem.value_and_grad(density, beta=2)
    assert value > 1e-4  # A nontrivial signal, not a vanishing-pulse false positive.
    np.testing.assert_array_equal(gradient[~problem.topology.region_mask], 0)
    for seed in (1, 7):
        direction = np.random.default_rng(seed).normal(size=density.shape)
        direction *= problem.topology.region_mask
        direction /= np.linalg.norm(direction)
        # Keep a random component, but avoid nearly cancelling projections whose
        # small objective differences are dominated by float32 roundoff at h=.003.
        sign = 1.0 if np.sum(gradient * direction) >= 0 else -1.0
        direction += sign * 0.1 * gradient / np.linalg.norm(gradient)
        direction /= np.linalg.norm(direction)
        adjoint = float(np.sum(gradient * direction))
        assert abs(adjoint) > 1e-5
        for h in (0.03, 0.01, 0.003):
            difference = (
                problem.value(density + h * direction, beta=2)
                - problem.value(density - h * direction, beta=2)
            ) / (2 * h)
            validation_metrics.check(
                f"{polarization} modal gradient seed={seed} h={h}",
                measured=difference,
                reference=adjoint,
                tolerance="gradient_float32",
                metadata={"h": h, "seed": seed, "reference_monitor": reference},
            )
            # These centered differences are above cancellation noise. A 1% gate
            # accommodates float32 roundoff across supported JAX CPU/GPU targets.
            np.testing.assert_allclose(difference, adjoint, rtol=0.01, atol=3e-6)
    ordinary = problem.material_simulation(density, beta=2).run(
        backend="jax", performance=False
    )
    expected = (
        abs(ordinary.mode("output").amps.sel(direction="+", mode_index=1).item()) ** 2
    )
    if reference:
        expected /= (
            abs(ordinary.mode("input").amps.sel(direction="+", mode_index=0).item())
            ** 2
        )
    np.testing.assert_allclose(value, expected, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("interval", [1, 32, 1000])
def test_checkpointed_adjoint_matches_uncheckpointed_scan(interval):
    reference = make_problem(checkpoint_interval=None)
    checkpointed = TopologyProblem(
        reference.simulation,
        reference.topology,
        reference.objective,
        checkpoint_interval=interval,
    )
    density = initial_density(reference)
    reference_value, reference_gradient = reference.value_and_grad(density, beta=2)
    value, gradient = checkpointed.value_and_grad(density, beta=2)
    np.testing.assert_allclose(value, reference_value, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=5e-4, atol=1e-7)


@pytest.mark.parametrize("projection", ["ssp", "identity"])
def test_projection_full_solver_gradient_matches_finite_difference(projection):
    from dataclasses import replace

    base = make_problem()
    problem = TopologyProblem(
        base.simulation,
        replace(
            base.topology,
            projection_type=projection,
            filter_radius=0
            if projection == "identity"
            else base.topology.filter_radius,
        ),
        base.objective,
    )
    density = initial_density(problem)
    value, gradient = problem.value_and_grad(density, beta=2)
    direction = (
        np.random.default_rng(1).normal(size=density.shape)
        * problem.topology.region_mask
    )
    direction /= np.linalg.norm(direction)
    h = 0.01
    finite_difference = (
        problem.value(density + h * direction, beta=2)
        - problem.value(density - h * direction, beta=2)
    ) / (2 * h)
    assert value > 1e-4
    np.testing.assert_allclose(
        finite_difference, np.sum(gradient * direction), rtol=0.01, atol=3e-6
    )


def _tensor_block():
    import beamz as bz

    omega = 2 * np.pi * bz.LIGHT_SPEED / 1.55e-6
    medium = bz.AnisotropicMedium(
        2.0,
        electric_poles=[
            bz.TensorPole.lorentz((1.5, 0.5, 0.0), resonance=2 * omega, damping=0.0)
        ],
    ).rotated((0.0, 0.0, 0.5))
    return bz.Rectangle(
        position=(2.4e-6, 3.05e-6), width=1.2e-6, height=0.3e-6, material=medium
    )


def test_gradient_with_tensor_pole_media_outside_design_matches_finite_difference():
    """Autodiff differentiates through tensor-pole corrections next to the design."""

    problem = make_problem(
        polarization="te", reference=False, extra_structures=(_tensor_block(),)
    )
    assert problem.program.dispersion.anisotropic.electric_regions
    density = initial_density(problem)
    value, gradient = problem.value_and_grad(density, beta=2)
    assert value > 1e-4
    direction = (
        np.random.default_rng(3).normal(size=density.shape)
        * problem.topology.region_mask
    )
    direction += 0.1 * np.linalg.norm(direction) * gradient / np.linalg.norm(gradient)
    direction /= np.linalg.norm(direction)
    adjoint = float(np.sum(gradient * direction))
    h = 0.01
    difference = (
        problem.value(density + h * direction, beta=2)
        - problem.value(density - h * direction, beta=2)
    ) / (2 * h)
    np.testing.assert_allclose(difference, adjoint, rtol=0.01, atol=3e-6)
    # Reverse and forward mode must agree to float32 precision. Pole constants
    # are folded so no reverse-mode cotangent meets a ~1e-31 factor (underflow
    # once gave percent-level noise here while forward mode stayed exact).
    import jax
    import jax.numpy as jnp

    latent = problem._density(density)
    beta = problem._beta(2.0)
    _, forward = jax.jvp(
        lambda z: problem._evaluate(z, beta),
        (latent,),
        (jnp.asarray(direction, dtype=latent.dtype),),
    )
    np.testing.assert_allclose(adjoint, float(forward), rtol=1e-4)


def test_tensor_media_must_not_overlap_the_design_or_use_spectral_adjoint():
    import beamz as bz

    block = _tensor_block()
    inside = bz.Rectangle(
        position=(2.6e-6, 1.6e-6), width=0.8e-6, height=0.8e-6, material=block.material
    )
    with pytest.raises(ValueError, match="must not overlap"):
        make_problem(polarization="te", extra_structures=(inside,))
    problem = make_problem(
        polarization="te", reference=False, extra_structures=(block,)
    )
    with pytest.raises(ValueError, match="adjoint"):
        TopologyProblem(
            problem.simulation,
            problem.topology,
            problem.objective,
            gradient_backend="adjoint",
        )


def _anisotropic_problem():
    import beamz as bz
    from beamz.optimization import TopologyProblem, TopologySpec

    base = make_problem(polarization="te", reference=False)
    omega = 2 * np.pi * bz.LIGHT_SPEED / 1.55e-6
    crystal = bz.AnisotropicMedium(
        (4.0, 3.0, 3.5),
        electric_poles=[
            bz.TensorPole.lorentz((0.8, 0.3, 0.0), resonance=2.2 * omega, damping=0.0)
        ],
    ).rotated((0.0, 0.0, 0.4))
    spec = base.topology
    topology = TopologySpec(
        design=spec.design,
        region_mask=spec.region_mask,
        resolution=spec.resolution,
        filter_radius=spec.filter_radius,
        learning_rate=spec.learning_rate,
        beta_schedule=spec.beta_schedule,
        material_min=bz.Material(1.0),
        material_max=crystal,
    )
    return TopologyProblem(
        base.simulation,
        topology,
        base.objective,
        checkpoint_interval=base.checkpoint_interval,
    )


def test_anisotropic_design_material_gradient_and_ordinary_run_agree():
    """Density blends air and a tilted dispersive crystal inside the design.

    The bound objective must equal an ordinary simulation of the same material
    map, and its autodiff gradient must match finite differences and forward
    mode.
    """
    import jax
    import jax.numpy as jnp

    problem = _anisotropic_problem()
    plan = problem.program.dispersion.anisotropic
    assert problem.program.coefficients.e_inverse_offdiagonal.size
    assert plan is not None and plan.electric_regions
    density = initial_density(problem)
    value, gradient = problem.value_and_grad(density, beta=2)
    assert value > 1e-4
    ordinary = problem.material_simulation(density, beta=2).run(
        backend="jax", performance=False
    )
    expected = (
        abs(ordinary.mode("output").amps.sel(direction="+", mode_index=1).item()) ** 2
    )
    np.testing.assert_allclose(value, expected, rtol=1e-5)
    direction = np.random.default_rng(5).normal(size=density.shape)
    direction *= problem.topology.region_mask
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
