"""Native reverse-mode parity, independently checked against JAX autodiff."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation import kernels
from beamz.simulation.backend import cuda_backend_status
from tests.integration.test_differentiable_simulation import (
    field_loss,
    make_simulation,
    seeded_state,
)

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
)


def assert_close(actual, expected):
    scale = float(jnp.max(jnp.abs(expected), initial=0.0))
    np.testing.assert_allclose(
        actual, expected, rtol=2e-4, atol=max(1e-9, 2e-6 * scale)
    )


@pytest.mark.parametrize("cpml,mixed", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    "metric_kind", ["isotropic_uniform", "axis_uniform", "rectilinear"]
)
def test_cuda_material_and_initial_state_vjps_match_jax(cpml, mixed, metric_kind):
    sim, epsilon = make_simulation(cpml=cpml, mixed=mixed, metric_kind=metric_kind)
    reference = sim.differentiable(backend="jax", checkpoint_interval=0)
    native = sim.differentiable(backend="cuda", checkpoint_interval=3)
    state = seeded_state(reference)
    before = jax.tree.map(lambda v: np.array(v, copy=True), state)

    def objective(solver, eps, sig, ex, hpsi, epsi):
        initial = state._replace(ex=ex, cpml_psi_h_terms=hpsi, cpml_psi_e_terms=epsi)
        out = solver(eps, conductivity=sig, state=initial)
        return field_loss(out) + 0.001 * sum(
            jnp.sum(v**2) for v in (*out.cpml_psi_h_terms, *out.cpml_psi_e_terms)
        )

    args = (
        epsilon,
        jnp.float32(120.0),
        state.ex,
        state.cpml_psi_h_terms,
        state.cpml_psi_e_terms,
    )
    expected = jax.value_and_grad(
        lambda *a: objective(reference, *a), argnums=(0, 1, 2, 3, 4)
    )(*args)
    actual = jax.value_and_grad(
        lambda *a: objective(native, *a), argnums=(0, 1, 2, 3, 4)
    )(*args)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_close(a, e)
    for a, e in zip(jax.tree.leaves(state), jax.tree.leaves(before), strict=True):
        np.testing.assert_array_equal(a, e)
    assert sum(jnp.linalg.norm(g) for g in jax.tree.leaves(actual[1][0])) > 0


def test_cuda_source_and_monitor_gradient_matches_jax():
    sim, epsilon = make_simulation(cpml=True, sources=True, monitors=True)
    solvers = [
        sim.differentiable(backend=b, checkpoint_interval=3) for b in ("jax", "cuda")
    ]

    def objective(solver, eps, amp):
        state = solver(eps, source_amplitudes=amp)
        return jnp.sum(jnp.abs(solver.monitor_fields(state, "output")) ** 2)

    outcomes = [
        jax.value_and_grad(
            lambda eps, amp, solver=s: objective(solver, eps, amp), argnums=(0, 1)
        )(epsilon, jnp.ones((1,), dtype=jnp.float32))
        for s in solvers
    ]
    for a, e in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(a, e)
    # An actual inverse-design step must improve this objective.
    value, gradients = outcomes[1]
    gradient = gradients[0]
    candidate = epsilon - 0.01 * gradient / jnp.linalg.norm(gradient)
    assert objective(solvers[1], candidate, jnp.ones((1,), dtype=jnp.float32)) < value


def test_cuda_native_timestep_clock_gradient_matches_jax():
    sim, epsilon = make_simulation(cpml=True, sources=True, monitors=True)
    outcomes = []
    for backend in ("jax", "cuda"):
        solver = sim.differentiable(backend=backend, checkpoint_interval=3)
        state = seeded_state(solver)

        def objective(time_in_steps, state=state, solver=solver):
            initial = state._replace(t=time_in_steps * solver.program.config.dt)
            out = solver(epsilon, state=initial)
            return jnp.sum(jnp.abs(solver.monitor_fields(out, "output")) ** 2)

        outcomes.append(jax.value_and_grad(objective)(jnp.float32(0.37)))
    for actual, expected in zip(outcomes[1], outcomes[0], strict=True):
        assert_close(actual, expected)


def test_cuda_native_timestep_raw_coefficients_match_phase_vjps(monkeypatch):
    """Exercise every material column of the native timestep linearization."""
    from beamz.simulation.cuda import differentiation

    sim, epsilon = make_simulation(cpml=True, mixed=True, sources=True, monitors=True)
    native = sim.differentiable(backend="cuda", checkpoint_interval=3)
    monkeypatch.setattr(differentiation, "build_native_step", lambda *args: None)
    reference = sim.differentiable(backend="cuda", checkpoint_interval=3)
    state = seeded_state(native)
    coefficients = native.coefficients(epsilon)
    names = tuple(
        f"{phase}_{kind}_{axis}"
        for phase in ("h", "e")
        for kind in ("decay", "source")
        for axis in "xyz"
    )

    def objective(solver, factors, amplitude):
        dynamic = coefficients._replace(
            **{
                name: getattr(coefficients, name) * factors[index]
                for index, name in enumerate(names)
            }
        )
        out = solver.run_coefficients(state, dynamic, amplitude)
        return (
            field_loss(out)
            + 0.01 * jnp.sum(jnp.abs(solver.monitor_fields(out, "output")) ** 2)
            + 0.001
            * sum(
                jnp.sum(value**2)
                for value in (*out.cpml_psi_h_terms, *out.cpml_psi_e_terms)
            )
        )

    arguments = (jnp.ones((12,), dtype=jnp.float32), jnp.ones((1,), dtype=jnp.float32))
    outcomes = [
        jax.value_and_grad(
            lambda f, a, solver=solver: objective(solver, f, a), argnums=(0, 1)
        )(*arguments)
        for solver in (reference, native)
    ]
    for actual, expected in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(actual, expected)


def test_cuda_native_timestep_ragged_windows_and_dft_state_vjps_match_jax():
    import beamz as bz
    from beamz.simulation.differentiation import DifferentiableSimulation

    sim, epsilon = make_simulation(cpml=True, sources=True, monitors=True, timesteps=11)
    sim = sim.updated_copy(
        monitors=[
            *sim.monitors,
            bz.FieldMonitor(
                center=(6 * 80e-9, 4 * 80e-9, 4 * 80e-9),
                size=(0, 3 * 80e-9, 2 * 80e-9),
                fields=("Ez", "Hy"),
                freqs=(1.8e14, 2.1e14, 2.8e14),
                interval=3,
                name="auxiliary",
            ),
        ]
    )
    outcomes = []
    for backend in ("jax", "cuda"):
        program = sim.compile(backend=backend)
        auxiliary = replace(
            program.monitors[1], dft_t_start=2e-16, dft_t_end=9e-16, dft_window_code=1
        )
        program = replace(program, monitors=(program.monitors[0], auxiliary))
        solver = DifferentiableSimulation(program, checkpoint_interval=3)
        state = seeded_state(solver)

        def objective(eps, initial_dft, solver=solver, state=state):
            initial = state._replace(
                dft_vec_re=initial_dft[0],
                dft_vec_im=initial_dft[1],
                dft_weight_sum=initial_dft[2],
            )
            out = solver(eps, state=initial)
            return jnp.sum(out.dft_vec_re**2 + out.dft_vec_im**2) + (
                0.001 * jnp.sum(out.dft_weight_sum**2)
            )

        initial = tuple(
            jnp.full_like(value, 0.03)
            for value in (state.dft_vec_re, state.dft_vec_im, state.dft_weight_sum)
        )
        outcomes.append(jax.value_and_grad(objective, argnums=(0, 1))(epsilon, initial))
    for actual, expected in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(actual, expected)


def test_cuda_material_gradient_matches_finite_difference():
    sim, epsilon = make_simulation(cpml=True, mixed=True)
    solver = sim.differentiable(backend="cuda", checkpoint_interval=3)
    state = seeded_state(solver)

    def objective(eps):
        return field_loss(solver(eps, state=state))

    direction = jnp.asarray(
        np.random.default_rng(82).normal(size=epsilon.shape), dtype=jnp.float32
    )
    direction /= jnp.linalg.norm(direction)
    gradient = jax.grad(objective)(epsilon)
    delta = 0.02
    fd = (
        objective(epsilon + delta * direction) - objective(epsilon - delta * direction)
    ) / (2 * delta)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), fd, rtol=0.02, atol=2e-6)


def test_cuda_autodiff_rejects_bf16_cpml(monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "bf16")
    sim, _ = make_simulation(cpml=True)
    with pytest.raises(ValueError, match="fp32"):
        sim.differentiable(backend="cuda")


@pytest.mark.parametrize("phase", ["h", "e"])
def test_cuda_phase_state_vjps_match_jax(phase):
    """Check every uniquely written field/psi gradient without timestep graphs."""
    sim, epsilon = make_simulation(cpml=True, mixed=True)
    solver = sim.differentiable(backend="jax")
    program, state = solver.program, seeded_state(solver)
    coefficients = solver.coefficients(epsilon)
    names = ("ex", "ey", "ez", "hx", "hy", "hz", "cpml_psi_h_terms", "cpml_psi_e_terms")
    inputs = tuple(getattr(state, name) for name in names)

    def objective(backend, values):
        ctx = kernels.CompiledStepContext(
            config=replace(program.config, backend=backend),
            boundary=program.boundary,
            source_batches={},
            metrics=program.metrics,
            resolution=program.config.resolution,
            dt=program.config.dt,
            dt_scalar=jnp.float32(program.config.dt),
            is_3d=True,
            differentiable=True,
        )
        initial = state._replace(**dict(zip(names, values, strict=True)))
        kernel = kernels.select_update_kernel(ctx)
        out = getattr(kernel, f"update_{phase}")(initial, ctx, coefficients)
        fields = tuple(getattr(out, f"{phase}{axis}") for axis in "xyz")
        masks = tuple(
            getattr(program.boundary.metallic, f"{phase}{axis}_mask") for axis in "xyz"
        )
        fields = kernels.apply_post_source_boundaries(fields, masks)
        return sum(jnp.sum(value**2) for value in fields) + 0.001 * sum(
            jnp.sum(value**2) for value in getattr(out, f"cpml_psi_{phase}_terms")
        )

    outcomes = [
        jax.value_and_grad(lambda values, backend=backend: objective(backend, values))(
            inputs
        )
        for backend in ("jax", "cuda_streamed")
    ]
    for actual, expected in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(actual, expected)


@pytest.mark.parametrize("phase", ["h", "e"])
@pytest.mark.parametrize("metric_kind", ["axis_uniform", "rectilinear"])
def test_cuda_cpml_coefficient_and_metric_vjps_match_jax(phase, metric_kind):
    sim, epsilon = make_simulation(cpml=True, mixed=True, metric_kind=metric_kind)
    solver = sim.differentiable(backend="jax")
    program = solver.program
    state = seeded_state(solver)
    coeffs = solver.coefficients(epsilon)
    terms = getattr(program.boundary.cpml, f"{phase}_terms")
    parameters = tuple((term.a, term.b, term.inv_kappa) for term in terms)

    def objective(backend, parameters, metric_scale):
        dynamic_terms = tuple(
            replace(term, a=a, b=b, inv_kappa=k)
            for term, (a, b, k) in zip(terms, parameters, strict=True)
        )
        cpml = replace(program.boundary.cpml, **{f"{phase}_terms": dynamic_terms})
        ctx = kernels.CompiledStepContext(
            config=replace(program.config, backend=backend),
            boundary=replace(program.boundary, cpml=cpml),
            source_batches={},
            metrics=type(program.metrics)(*(v * metric_scale for v in program.metrics)),
            resolution=program.config.resolution,
            dt=program.config.dt,
            dt_scalar=jnp.float32(program.config.dt),
            is_3d=True,
            differentiable=True,
        )
        kernel = kernels.select_update_kernel(ctx)
        out = getattr(kernel, f"update_{phase}")(state, ctx, coeffs)
        fields = tuple(getattr(out, f"{phase}{axis}") for axis in "xyz")
        masks = tuple(
            getattr(program.boundary.metallic, f"{phase}{axis}_mask") for axis in "xyz"
        )
        fields = kernels.apply_post_source_boundaries(fields, masks)
        return sum(jnp.sum(v**2) for v in fields) + 0.001 * sum(
            jnp.sum(v**2) for v in getattr(out, f"cpml_psi_{phase}_terms")
        )

    outcomes = [
        jax.value_and_grad(
            lambda p, m, backend=b: objective(backend, p, m), argnums=(0, 1)
        )(parameters, jnp.float32(1.0))
        for b in ("jax", "cuda_streamed")
    ]
    for actual, expected in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(actual, expected)


def test_cuda_checkpointing_reduces_compiled_gradient_workspace():
    sim, epsilon = make_simulation(timesteps=65)
    workspace = []
    for interval in (0, 8):
        solver = sim.differentiable(backend="cuda", checkpoint_interval=interval)
        state = seeded_state(solver)
        objective = jax.jit(
            jax.value_and_grad(
                lambda eps, run=solver, initial=state: field_loss(
                    run(eps, state=initial)
                )
            )
        )
        analysis = objective.lower(epsilon).compile().memory_analysis()
        workspace.append(analysis.temp_size_in_bytes)
    assert workspace[1] < 0.8 * workspace[0]


@pytest.mark.parametrize("phase", ["h", "e"])
@pytest.mark.parametrize("layout", ["scalar", "profile", "grid"])
def test_cuda_trainable_broadcast_and_grid_coefficients_match_jax(phase, layout):
    """Fixed-gradient pruning must preserve reductions for trainable inputs."""
    sim, epsilon = make_simulation(cpml=True, mixed=True)
    solver = sim.differentiable(backend="jax")
    program = solver.program
    state = seeded_state(solver)
    coefficients = solver.coefficients(epsilon)
    names = tuple(
        f"{phase}_{kind}_{axis}" for kind in ("decay", "source") for axis in "xyz"
    )
    parameters = []
    for name in names:
        field = getattr(state, phase + name[-1])
        shape = (
            ()
            if layout == "scalar"
            else ((1, field.shape[1], 1) if layout == "profile" else field.shape)
        )
        parameters.append(jnp.full(shape, jnp.mean(getattr(coefficients, name))))

    def objective(backend, values):
        if backend == "jax":
            # The ordinary JAX planner derives these coefficients from physical
            # eps/sigma. Check arbitrary raw coefficients against its canonical
            # curl/CPML primitives instead of varying unused planner leaves.
            fields = tuple(getattr(state, phase + axis) for axis in "xyz")
            axes = (1, 0, 0, 2, 2, 1)
            if phase == "h":
                sources = (state.ez, state.ey, state.ex, state.ez, state.ey, state.ex)
                raw = tuple(
                    jnp.diff(v, axis=a) / program.config.resolution
                    for v, a in zip(sources, axes, strict=True)
                )
            else:
                views = kernels.build_h_boundary_views_for_e_3d(
                    state.hx,
                    state.hy,
                    state.hz,
                    program.boundary.cpml.metallic_edges,
                    logical_shapes=program.boundary.logical_component_shapes,
                )
                raw = tuple(
                    kernels._adjacent_difference(
                        views[name], a, program.config.resolution
                    )
                    for name, a in zip(
                        ("hz_y", "hy_z", "hx_z", "hz_x", "hy_x", "hx_y"),
                        axes,
                        strict=True,
                    )
                )
            corrected = tuple(
                kernels.correct_cpml_term(
                    kernels.fit_array_to_shape(d, fields[i // 2].shape), psi, term
                )
                for i, (d, psi, term) in enumerate(
                    zip(
                        raw,
                        getattr(state, f"cpml_psi_{phase}_terms"),
                        getattr(program.boundary.cpml, f"{phase}_terms"),
                        strict=True,
                    )
                )
            )
            sign = -1 if phase == "h" else 1
            updated = tuple(
                decay * old
                + sign * source * (corrected[2 * i][0] + corrected[2 * i + 1][0])
                for i, (old, decay, source) in enumerate(
                    zip(fields, values[:3], values[3:], strict=True)
                )
            )
            masks = tuple(
                getattr(program.boundary.metallic, phase + axis + "_mask")
                for axis in "xyz"
            )
            updated = kernels.apply_post_source_boundaries(updated, masks)
            return sum(jnp.sum(v**2) for v in updated) + sum(
                0.001 * jnp.sum(v[1] ** 2) for v in corrected
            )
        coeffs = coefficients._replace(**dict(zip(names, values, strict=True)))
        context = kernels.CompiledStepContext(
            config=replace(program.config, backend=backend),
            boundary=program.boundary,
            source_batches={},
            metrics=program.metrics,
            resolution=program.config.resolution,
            dt=program.config.dt,
            dt_scalar=jnp.float32(program.config.dt),
            is_3d=True,
            differentiable=True,
        )
        out = getattr(kernels.select_update_kernel(context), f"update_{phase}")(
            state, context, coeffs
        )
        return sum(jnp.sum(getattr(out, phase + axis) ** 2) for axis in "xyz") + sum(
            0.001 * jnp.sum(v**2) for v in getattr(out, f"cpml_psi_{phase}_terms")
        )

    outcomes = [
        jax.jit(jax.value_and_grad(lambda p, backend=b: objective(backend, p)))(
            tuple(parameters)
        )
        for b in ("jax", "cuda_streamed")
    ]
    for actual, expected in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(actual, expected)
    assert sum(float(jnp.linalg.norm(g)) for g in outcomes[1][1]) > 0


@pytest.mark.parametrize("cpml,mixed", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("layout", ["scalar", "profile", "grid"])
def test_native_material_chunk_matches_phase_path_with_fixed_losses(
    monkeypatch, cpml, mixed, layout
):
    """Exercise aliased cotangents, bank parity, and broadcast accumulation."""
    from beamz.simulation.cuda import differentiation

    sim, epsilon = make_simulation(
        cpml=cpml, mixed=mixed, sources=True, monitors=True, timesteps=7
    )
    native = sim.differentiable(backend="cuda", checkpoint_interval=3)
    monkeypatch.setattr(differentiation, "build_native_step", lambda *args: None)
    reference = sim.differentiable(backend="cuda", checkpoint_interval=3)
    state = seeded_state(native)
    coefficients = native.coefficients(epsilon)
    updates = {}
    for axis in "xyz":
        shape = getattr(state, f"e{axis}").shape
        source = getattr(coefficients, f"e_source_{axis}")
        value = jnp.mean(source)
        selected_shape = (
            ()
            if layout == "scalar"
            else (1, shape[1], 1)
            if layout == "profile"
            else shape
        )
        updates[f"e_source_{axis}"] = jnp.full(selected_shape, value)
        updates[f"h_decay_{axis}"] = jnp.float32(0.93)
        updates[f"e_decay_{axis}"] = jnp.linspace(0.91, 0.99, shape[1]).reshape(
            (1, shape[1], 1)
        )
    coefficients = coefficients._replace(**updates)

    def objective(solver, factors, ex, hx, hpsi, epsi):
        initial = state._replace(
            ex=ex, hx=hx, cpml_psi_h_terms=hpsi, cpml_psi_e_terms=epsi
        )
        dynamic = coefficients._replace(
            **{
                f"e_source_{axis}": getattr(coefficients, f"e_source_{axis}")
                * factors[index]
                for index, axis in enumerate("xyz")
            }
        )
        out = solver.run_coefficients(initial, dynamic)
        return (
            field_loss(out)
            + 0.001
            * sum(jnp.sum(p**2) for p in (*out.cpml_psi_h_terms, *out.cpml_psi_e_terms))
            + jnp.sum(out.dft_vec_re**2 + out.dft_vec_im**2)
        )

    args = (
        jnp.ones(3, jnp.float32),
        state.ex,
        state.hx,
        state.cpml_psi_h_terms,
        state.cpml_psi_e_terms,
    )
    outcomes = [
        jax.value_and_grad(
            lambda *args, solver=solver: objective(solver, *args),
            argnums=(0, 1, 2, 3, 4),
        )(*args)
        for solver in (reference, native)
    ]
    for actual, expected in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(actual, expected)
    direction = jnp.array([0.3, -0.4, 0.5], jnp.float32)
    derivative = jnp.vdot(outcomes[1][1][0], direction)
    delta = 0.01
    high = objective(native, args[0] + delta * direction, *args[1:])
    low = objective(native, args[0] - delta * direction, *args[1:])
    np.testing.assert_allclose(
        derivative, (high - low) / (2 * delta), rtol=0.01, atol=2e-5
    )


@pytest.mark.parametrize("cpml,mixed", [(False, False), (True, False), (True, True)])
def test_native_fixed_material_state_chunk_needs_no_primal_tape(
    monkeypatch, cpml, mixed
):
    from beamz.simulation.cuda import differentiation

    sim, epsilon = make_simulation(
        cpml=cpml, mixed=mixed, sources=True, monitors=True, timesteps=7
    )
    native = sim.differentiable(backend="cuda", checkpoint_interval=2)
    monkeypatch.setattr(differentiation, "build_native_step", lambda *args: None)
    reference = sim.differentiable(backend="cuda", checkpoint_interval=2)
    state = seeded_state(native)
    coefficients = native.coefficients(epsilon)._replace(
        **{
            f"{phase}_decay_{axis}": jnp.float32(0.93 if phase == "h" else 0.91)
            for phase in "he"
            for axis in "xyz"
        }
    )

    def objective(solver, initial):
        ex, hx, hpsi, epsi, re, im, weights = initial
        dynamic = state._replace(
            ex=ex,
            hx=hx,
            cpml_psi_h_terms=hpsi,
            cpml_psi_e_terms=epsi,
            dft_vec_re=re,
            dft_vec_im=im,
            dft_weight_sum=weights,
        )
        out = solver.run_coefficients(dynamic, coefficients)
        return (
            field_loss(out)
            + 0.001
            * sum(jnp.sum(p**2) for p in (*out.cpml_psi_h_terms, *out.cpml_psi_e_terms))
            + jnp.sum(out.dft_vec_re**2 + out.dft_vec_im**2)
        )

    initial = (
        state.ex,
        state.hx,
        state.cpml_psi_h_terms,
        state.cpml_psi_e_terms,
        state.dft_vec_re,
        state.dft_vec_im,
        state.dft_weight_sum,
    )
    outcomes = [
        jax.value_and_grad(lambda initial, solver=solver: objective(solver, initial))(
            initial
        )
        for solver in (reference, native)
    ]
    for actual, expected in zip(
        jax.tree.leaves(outcomes[1]), jax.tree.leaves(outcomes[0]), strict=True
    ):
        assert_close(actual, expected)


@pytest.mark.parametrize("cpml", [False, True])
@pytest.mark.parametrize("metric_kind", ["isotropic_uniform", "rectilinear"])
def test_pmc_material_and_initial_field_gradients_match_jax(cpml, metric_kind):
    from beamz import PMC

    sim, epsilon = make_simulation(cpml=cpml, metric_kind=metric_kind, timesteps=4)
    sim = sim.updated_copy(boundaries=[*sim.boundaries, PMC()])
    reference = sim.differentiable(backend="jax", checkpoint_interval=0)
    native = sim.differentiable(backend="cuda", checkpoint_interval=2)
    state = seeded_state(reference)

    def objective(solver, eps, ez):
        return field_loss(solver(eps, state=state._replace(ez=ez)))

    expected = jax.value_and_grad(
        lambda eps, ez: objective(reference, eps, ez), argnums=(0, 1)
    )(epsilon, state.ez)
    actual = jax.value_and_grad(
        lambda eps, ez: objective(native, eps, ez), argnums=(0, 1)
    )(epsilon, state.ez)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_close(a, e)
