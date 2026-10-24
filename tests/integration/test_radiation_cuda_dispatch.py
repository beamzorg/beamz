"""CPU contracts for CUDA inverse-design dispatch and dynamic coefficients.

Native compilation/launch is replaced by JAX here; physical native parity is
covered separately by tests/hardware/test_cuda_radiation.py.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz import Simulation
from beamz.const import EPS_0
from beamz.optimization import CollectionConePower, InverseDesign, TopologyProblem
from beamz.simulation.differentiable import (
    topology_coefficients,
    topology_yee_permittivity,
)
from beamz.simulation.differentiation import build_differentiable_scan
from tests.integration.test_radiation_inverse_design import radiation_design


def test_cuda_density_updates_replace_codebooks_and_match_portable_physics():
    design, _ = radiation_design(run_time=15e-15)
    problem = design._problem
    program = problem.program
    # Simulate an encoded CUDA coefficient pool; all three trainable component
    # arrays must replace it, including supports that straddle the design edge.
    encoded = program.coefficients._replace(
        **{
            f"e_{kind}_{axis}": jnp.array([0], dtype=jnp.int32)
            if kind == "source"
            else jnp.array([123.0], dtype=jnp.float32)
            for axis in "xyz"
            for kind in ("source", "decay")
        }
    )
    native_program = replace(
        program,
        config=replace(program.config, backend="cuda_streamed"),
        coefficients=encoded,
    )
    density = design._density(jnp.asarray(design.design_region.initial_parameters))
    eps = problem._permittivity(density, 1.0)
    native = topology_coefficients(native_program, eps, problem._mask)
    expected_yee = topology_yee_permittivity(program, eps, problem._mask)
    for axis, epsilon in zip("xyz", expected_yee, strict=True):
        source = getattr(native, f"e_source_{axis}")
        assert source.dtype == jnp.float32 and source.shape == epsilon.shape
        np.testing.assert_allclose(
            source, program.config.dt / (EPS_0 * epsilon), rtol=3e-6
        )
        np.testing.assert_array_equal(getattr(native, f"e_decay_{axis}"), 1.0)
    # Execute the real portable updates with the native coefficient representation,
    # then compare with the ordinary density-bound JAX material representation.
    scan = build_differentiable_scan(program, checkpoint_interval=7)
    expected = scan(
        problem._initial, topology_coefficients(program, eps, problem._mask)
    )
    actual = scan(problem._initial, native)
    for name in ("ex", "ey", "ez", "hx", "hy", "hz", "dft_vec_re", "dft_vec_im"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=5e-5, atol=1e-8
        )


def test_high_level_cuda_dispatch_preserves_radiation_values_and_gradients(monkeypatch):
    import beamz.optimization.problem as problem_module

    reference, projection = radiation_design(run_time=40e-15)
    original_compile = Simulation.compile
    calls = []

    def compile_simulation(sim, **kwargs):
        backend = kwargs.pop("backend", "jax")
        program = original_compile(sim, backend="jax", **kwargs)
        if backend == "cuda":
            return replace(
                program, config=replace(program.config, backend="cuda_streamed")
            )
        return program

    def native_scan(program, *, checkpoint_interval):
        calls.append((program.config.backend, checkpoint_interval))
        # Only the native execution boundary is substituted. Density mapping,
        # CUDA coefficient construction, monitor packing and projection are real.
        return build_differentiable_scan(
            replace(program, config=replace(program.config, backend="jax")),
            checkpoint_interval=checkpoint_interval,
        )

    monkeypatch.setattr(Simulation, "compile", compile_simulation)
    monkeypatch.setattr(problem_module, "build_differentiable_scan", native_scan)
    native = reference.updated_copy(backend="cuda", output_monitor_names=("radiation",))
    reference = reference.updated_copy(output_monitor_names=("radiation",))
    assert calls == [("cuda_streamed", reference.checkpoint_interval)]
    assert native._problem.program.config.backend == "cuda_streamed"
    assert native.backend == native._problem.backend == "cuda"
    objective = CollectionConePower(projection, 0.9)
    params = jnp.asarray(reference.design_region.initial_parameters)
    expected = jax.value_and_grad(reference.make_objective_fn(objective))(params)
    actual = jax.jit(jax.value_and_grad(native.make_objective_fn(objective)))(params)
    for value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, expected_value, rtol=2e-4, atol=1e-7)
    assert np.linalg.norm(actual[1]) > 0
    assert native._problem.fingerprint != reference._problem.fingerprint
    # The low-level objective API must choose the same native scan as the
    # high-level data-returning API.
    low_level = TopologyProblem(
        native.simulation,
        native._problem.topology,
        objective,
        backend="cuda",
        gradient_backend="autodiff",
    )
    np.testing.assert_allclose(
        low_level.value(native._density(params)), actual[0], rtol=2e-4
    )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "backend,gradient",
    [("auto", "autodiff"), ("typo", "autodiff"), ("cuda", "adjoint")],
)
def test_invalid_execution_selection_is_rejected_before_compilation(
    monkeypatch, backend, gradient
):
    # No GPU or simulation preparation should be needed to diagnose these errors.
    def unexpected_compile(*args, **kwargs):
        pytest.fail("Invalid execution selection reached compilation.")

    monkeypatch.setattr(Simulation, "compile", unexpected_compile)
    with pytest.raises(ValueError, match="backend"):
        InverseDesign(None, None, backend=backend, gradient_backend=gradient)
    with pytest.raises(ValueError, match="backend"):
        TopologyProblem(None, None, backend=backend, gradient_backend=gradient)
