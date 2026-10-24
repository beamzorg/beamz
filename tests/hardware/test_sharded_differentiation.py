"""Distributed reverse-mode contracts; run separately from static-only review."""

import os
import subprocess
import sys

import numpy as np
import pytest


def _exercise(axis, cpml, count, backend="jax"):
    import jax
    import jax.numpy as jnp

    from beamz.simulation import sharding
    from beamz.simulation.execute import initial_program_state
    from beamz.simulation.observe import MONITOR_FIELDS
    from tests.integration.test_differentiable_simulation import (
        _with_conductivity,
        make_simulation,
        seeded_state,
    )

    sim, epsilon = make_simulation(cpml=cpml, monitors=True, sources=True)
    sim = _with_conductivity(sim, 0)
    reference = sim.differentiable(checkpoint_interval=0)
    start = seeded_state(reference)

    def loss(solver, state):
        # Both a replicated observation and partitioned fields contribute once.
        fields = sum(jnp.sum(getattr(state, c) ** 2) for c in ("ex", "ey", "ez"))
        return 2.7 * (fields + jnp.sum(jnp.abs(solver.monitor_fields(state)) ** 2))

    expected, gradient = jax.value_and_grad(
        lambda eps: loss(reference, reference(eps, state=start))
    )(epsilon)
    cfg = dict(
        axis=axis, num_devices=count, backend="cpu" if backend == "jax" else "gpu"
    )
    previous = None
    for interval in (1, 3):
        solver = sim.differentiable(
            backend=backend, checkpoint_interval=interval, sharding=cfg
        )
        program = solver.program
        prepared = initial_program_state(
            program, t=0, current_step=0, continuation=start
        )
        prepared = sharding.prepare_state(
            program, prepared, replicated_fields=MONITOR_FIELDS
        )
        parameters = solver.prepare_permittivity(np.asarray(epsilon))
        objective = jax.jit(
            lambda eps, solver=solver, prepared=prepared: loss(
                solver, solver(eps, state=prepared)
            )
        )
        actual, derivative = jax.jit(jax.value_and_grad(objective))(parameters)
        np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)
        physical = tuple(slice(0, n) for n in epsilon.shape)
        np.testing.assert_allclose(
            np.asarray(derivative)[physical], gradient, rtol=3e-4, atol=2e-6
        )
        assert not derivative.sharding.is_fully_replicated
        padding = np.asarray(derivative).copy()
        padding[physical] = 0
        np.testing.assert_array_equal(padding, 0)
        output = solver(parameters, state=prepared)
        expected_state = reference(epsilon, state=start)
        for name in (
            "ex",
            "ey",
            "ez",
            "hx",
            "hy",
            "hz",
            "cpml_psi_e_terms",
            "cpml_psi_h_terms",
            "dft_vec_re",
            "dft_vec_im",
            "dft_weight_sum",
        ):
            cropped = sharding.crop_state(program, output)
            for value, target in zip(
                jax.tree.leaves(getattr(cropped, name)),
                jax.tree.leaves(getattr(expected_state, name)),
                strict=True,
            ):
                np.testing.assert_allclose(value, target, rtol=5e-5, atol=2e-6)
        continued = solver(parameters, state=output)
        target = reference(epsilon, state=expected_state)
        np.testing.assert_allclose(
            solver.monitor_fields(continued),
            reference.monitor_fields(target),
            rtol=6e-5,
            atol=2e-6,
        )
        direction = derivative / jnp.linalg.norm(derivative)
        for delta in (0.01, 0.005):
            fd = (
                objective(parameters + delta * direction)
                - objective(parameters - delta * direction)
            ) / (2 * delta)
            np.testing.assert_allclose(
                fd, jnp.sum(derivative * direction), rtol=5e-3, atol=2e-5
            )
        if previous is not None:
            np.testing.assert_allclose(derivative, previous, rtol=3e-5, atol=2e-6)
        previous = derivative
        if backend == "jax" and interval == 1:
            names = (
                "ex",
                "ey",
                "ez",
                "hx",
                "hy",
                "hz",
                "cpml_psi_h_terms",
                "cpml_psi_e_terms",
                "dft_vec_re",
                "dft_vec_im",
            )
            values = tuple(getattr(prepared, n) for n in names)

            def transition(
                values,
                solver=solver,
                parameters=parameters,
                prepared=prepared,
                names=names,
            ):
                result = solver(
                    parameters,
                    state=prepared._replace(**dict(zip(names, values, strict=True))),
                )
                return tuple(getattr(result, n) for n in names)

            tangent = jax.tree.map(lambda v: jnp.full_like(v, 0.003), values)
            output, pushed = jax.jvp(transition, (values,), (tangent,))
            cotangent = jax.tree.map(lambda v: jnp.full_like(v, 0.007), output)
            pulled = jax.vjp(transition, values)[1](cotangent)[0]

            def dot(a, b):
                return sum(
                    jnp.vdot(x, y)
                    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
                )

            np.testing.assert_allclose(
                dot(pushed, cotangent), dot(tangent, pulled), rtol=1e-4, atol=1e-6
            )
        # Parameter and Adam moments retain their partition on repeated updates.
        import optax

        optimizer = optax.adam(0.01)
        state = optimizer.init(parameters)
        for _ in range(2):
            changes, state = jax.jit(optimizer.update)(derivative, state, parameters)
            parameters = parameters + changes
            assert not parameters.sharding.is_fully_replicated
        assert all(
            not v.sharding.is_fully_replicated
            for v in jax.tree.leaves(state)
            if v.ndim == 3
        )


@pytest.mark.parametrize("axis", ("x", "y", "z"))
@pytest.mark.parametrize("cpml", (False, True))
@pytest.mark.parametrize("count", (2, 3))
def test_cpu_device_sharded_gradients(axis, cpml, count):
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={count}",
    )
    code = f"from tests.hardware.test_sharded_differentiation import _exercise; _exercise({axis!r}, {cpml!r}, {count})"
    subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_native_gpu_sharded_gradients(axis):
    import jax

    devices = [d for d in jax.devices() if d.platform == "gpu"]
    if len(devices) < 2:
        pytest.skip("requires two CUDA GPUs and the BeamZ extension")
    _exercise(axis, True, 2, "cuda_streamed")


def test_public_topology_sharding():
    code = """
import numpy as np
import tempfile
from pathlib import Path
from beamz.optimization import ModePower, TopologyProblem
from tests.topology_3d_case import make_3d_design

base = make_3d_design(run_time=20e-15)._problem
kwargs = dict(objective=ModePower('output'), gradient_backend='autodiff', checkpoint_interval=3)
reference = TopologyProblem(base.simulation, base.topology, **kwargs)
distributed = TopologyProblem(base.simulation, base.topology, sharding=dict(axis='x', num_devices=2, backend='cpu'), **kwargs)
density = base.topology.initial_state().density
expected, gradient = reference.value_and_grad(density)
actual, derivative = distributed.value_and_grad(density)
np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-8)
np.testing.assert_allclose(derivative, gradient, rtol=5e-4, atol=1e-8)
full = distributed.run(steps=2)
first = distributed.run(steps=2, stop_after=1)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'resume.npz'
    first.save(path)
    resumed = distributed.run(steps=2, resume=path)
np.testing.assert_allclose(resumed.final_params, full.final_params, rtol=0, atol=1e-7)
"""
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
