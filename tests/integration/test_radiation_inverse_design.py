"""Small 3D radiation objective through FDTD and the high-level inverse API."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz import FieldMonitor
from beamz.analysis import FarFieldProjection, FarFieldSurface
from beamz.analysis.far_field import bind_surface_data, surface_data
from beamz.lattice import component_coordinates_3d_um
from beamz.optimization import (
    CollectionConePower,
    DirectionalPower,
    TargetBeamOverlap,
    TopologyProblem,
)
from beamz.simulation.execute import (
    _compiled_source_launch_powers,
    _decode_monitor_results,
)
from beamz.simulation.results import SimulationResults
from tests.topology_3d_case import make_3d_design


def radiation_design(*, run_time=60e-15, backend="autodiff"):
    base = make_3d_design(run_time=run_time)
    frequency = base.simulation.monitors[0].freqs[0]
    monitor = FieldMonitor(
        center=(1.5e-6, 1e-6, 1.175e-6),
        size=(1.2e-6, 1.2e-6, 0),
        freqs=[frequency, 0.97 * frequency],
        name="radiation",
        fields=("Ex", "Ey", "Hx", "Hy"),
    )
    design = base.updated_copy(
        simulation=base.simulation.updated_copy(
            monitors=[*base.simulation.monitors, monitor]
        ),
        output_monitor_names=None,
        gradient_backend=backend,
    )
    directions = [(0, 0, 1), (0.6, 0, 0.8), (-0.6, 0, 0.8)]
    projection = FarFieldProjection(
        [FarFieldSurface("radiation")],
        directions,
        [0.1, 0.1, 0.1],
        frequencies=tuple(monitor.freqs[::-1]),
    )
    return design, projection


def test_high_level_radiation_gradients_and_ordinary_result_parity():
    design, projection = radiation_design()
    assert design.output_monitor_names == ("output", "radiation")
    params = jnp.asarray(design.design_region.initial_parameters)
    # A combined objective checks coherent overlap and finite-NA reductions too.
    target = np.array([[1, 0, 0], [0.8, 0, -0.6], [0.8, 0, 0.6]], complex)
    objective = (
        DirectionalPower(projection)
        + CollectionConePower(projection, 0.9)
        + TargetBeamOverlap(projection, target)
    )
    fun = jax.jit(design.make_objective_fn(objective))
    value, gradient = jax.value_and_grad(fun)(params)
    assert np.isfinite(value) and value > 0
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
    direction = gradient / jnp.linalg.norm(gradient)
    fd = (fun(params + 0.005 * direction) - fun(params - 0.005 * direction)) / 0.01
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.008, atol=1e-7)
    # Decode the same tiny forward state independently; no additional FDTD run.
    problem = design._problem
    state = problem._simulate(design._density(params), jnp.asarray(1.0))
    results = SimulationResults.from_run(
        design.simulation,
        runtime_fields=problem.program.grid,
        monitor_results=_decode_monitor_results(
            design.simulation, problem.program, state
        ),
        source_launch_powers=_compiled_source_launch_powers(
            problem.program, len(design.simulation.sources)
        ),
    )
    ordinary = projection(results)
    differentiable = projection(design.to_simulation_data(params))
    np.testing.assert_allclose(
        ordinary.electric, differentiable.electric, rtol=4e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        ordinary.intensity, differentiable.intensity, rtol=4e-5, atol=1e-9
    )
    # Also bind through the low-level composable-objective API.
    bound = objective.bind(results, problem.program)
    np.testing.assert_allclose(bound(state), value, rtol=4e-5)
    np.testing.assert_allclose(objective(results), value, rtol=4e-5)
    np.testing.assert_allclose(
        projection(results.renormalize(None)).electric,
        ordinary.electric,
        rtol=3e-6,
        atol=1e-8,
    )


def test_radiation_spectral_adjoint_matches_autodiff_and_finite_differences():
    design, projection = radiation_design(run_time=300e-15)
    # One output and frequency keep this a small CPU integration case.
    projection = replace(projection, frequencies=(projection.frequencies[0],))
    monitor = design.simulation.monitors[-1].updated_copy(freqs=projection.frequencies)
    design = design.updated_copy(
        simulation=design.simulation.updated_copy(monitors=[monitor]),
        output_monitor_names=("radiation",),
    )
    objective = CollectionConePower(projection, 0.9)
    params = jnp.asarray(design.design_region.initial_parameters)
    value, gradient = jax.value_and_grad(design.make_objective_fn(objective))(params)
    adjoint = design.updated_copy(gradient_backend="adjoint")
    adj_value, adj_gradient = jax.value_and_grad(adjoint.make_objective_fn(objective))(
        params
    )
    np.testing.assert_allclose(adj_value, value, rtol=3e-5)
    assert np.linalg.norm(gradient) > 0
    assert np.linalg.norm(adj_gradient - gradient) / np.linalg.norm(gradient) < 0.015
    direction = gradient / jnp.linalg.norm(gradient)
    fun = design.make_objective_fn(objective)
    fd = (fun(params + 0.005 * direction) - fun(params - 0.005 * direction)) / 0.01
    np.testing.assert_allclose(
        fd, jnp.sum(adj_gradient * direction), rtol=0.015, atol=1e-8
    )


def test_surface_binding_uses_temporal_phase_and_non_endpoint_area_weights():
    design, _ = radiation_design(run_time=5e-15)
    problem = design._problem
    results, program = problem._binding_results, problem.program
    spec = next(s for s in program.monitors if s.name == "radiation")
    count = 6 * spec.freq_count * spec.dft_point_count
    rng = np.random.default_rng(2)
    raw = rng.normal(size=count) + 1j * rng.normal(size=count)
    state = problem._initial._replace(
        dft_vec_re=problem._initial.dft_vec_re.at[
            spec.dft_value_offset : spec.dft_value_offset + count
        ].set(raw.real),
        dft_vec_im=problem._initial.dft_vec_im.at[
            spec.dft_value_offset : spec.dft_value_offset + count
        ].set(raw.imag),
        dft_weight_sum=problem._initial.dft_weight_sum.at[
            spec.dft_weight_offset : spec.dft_weight_offset + spec.freq_count
        ].set(jnp.array([4.0, 7.0])),
    )
    decoded = replace(
        results, monitors=_decode_monitor_results(design.simulation, program, state)
    )
    ordinary = surface_data(decoded, ["radiation"])["radiation"]
    bound = bind_surface_data(results, program, ["radiation"])(state)["radiation"]
    np.testing.assert_allclose(ordinary.electric, bound.electric, rtol=2e-6)
    np.testing.assert_allclose(ordinary.magnetic, bound.magnetic, rtol=2e-6)
    assert np.all(bound.points[:, 2] == 1.175e-6)
    np.testing.assert_allclose(np.sum(bound.area_weights), 1.2e-6**2, rtol=2e-6)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_surface_coordinates_match_native_yee_interpolation_on_off_grid_planes(axis):
    base = make_3d_design(run_time=5e-15)
    extent = np.full(3, 0.4e-6)
    extent[axis] = 0
    monitor = FieldMonitor(
        center=(1.455e-6, 1.055e-6, 1.155e-6),
        size=extent,
        freqs=base.simulation.monitors[0].freqs,
        name="surface",
    )
    design = base.updated_copy(
        simulation=base.simulation.updated_copy(monitors=[monitor]),
        output_monitor_names=None,
    )
    problem = design._problem
    program, results = problem.program, problem._binding_results
    spec = program.monitors[0]
    metadata = bind_surface_data(results, program, ["surface"])(problem._initial)[
        "surface"
    ]
    expected = 1 + metadata.points @ np.array([1e6, 2e6, -0.5e6])
    # Affine functions interpolate exactly. Evaluate on each independent native
    # Yee support, then compare its acquisition to the reported common plane.
    for i, component in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        coordinates = component_coordinates_3d_um(
            component,
            program.grid.material_grid.shape,
            design.simulation.resolution / 1e-6,
        )
        zz, yy, xx = np.meshgrid(
            coordinates["z"], coordinates["y"], coordinates["x"], indexing="ij"
        )
        native = 1 + xx + 2 * yy - 0.5 * zz
        actual = np.sum(
            native.ravel()[np.asarray(spec.dft_flat_idx[i])] * spec.dft_weights[i],
            axis=-1,
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)
    assert np.all(metadata.points[:, axis] == monitor.center[axis])
    np.testing.assert_allclose(np.sum(metadata.area_weights), 0.4e-6**2, rtol=2e-6)


def test_spectral_backend_plans_all_surface_frequencies_without_a_run():
    design, projection = radiation_design(run_time=5e-15)
    problem = TopologyProblem(
        design.simulation,
        design._problem.topology,
        DirectionalPower(projection),
        gradient_backend="adjoint",
    )
    expected = np.asarray(projection.frequencies, dtype=np.float32)
    np.testing.assert_array_equal(
        problem._spectral_adjoint.frequencies, np.sort(expected)
    )
    assert (
        problem.fingerprint
        != TopologyProblem(
            design.simulation,
            design._problem.topology,
            DirectionalPower(
                replace(projection, directions=[(0, 0, -1)], solid_angle_weights=None)
            ),
        ).fingerprint
    )


def test_surface_binding_rejects_two_dimensional_outputs():
    from tests.topology_case import make_problem

    base = make_problem(run_time=5e-15)
    monitor = FieldMonitor(
        center=(5e-6, 2e-6, 0),
        size=(0, 2e-6, 1e-6),
        freqs=base.simulation.monitors[0].freqs,
        name="surface",
    )
    problem = TopologyProblem(
        base.simulation.updated_copy(monitors=[monitor]),
        base.topology,
        output_monitor_names=["surface"],
    )
    with pytest.raises(ValueError, match="3D"):
        bind_surface_data(problem._binding_results, problem.program, ["surface"])
