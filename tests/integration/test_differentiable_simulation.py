"""End-to-end inverse-design execution and checkpoint semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid


def _with_conductivity(sim, sigma):
    from dataclasses import replace

    return sim.updated_copy(
        material_grid=replace(
            sim.material_grid,
            conductivity=np.full(sim.material_grid.shape, sigma, dtype=np.float32),
        )
    )


@pytest.mark.parametrize("dimensions,polarization", [(3, "tm"), (2, "tm"), (2, "te")])
@pytest.mark.parametrize("absorber", ["pec", "cpml", "sponge"])
def test_conductivity_binding_matches_ordinary_materials(
    dimensions, polarization, absorber
):
    sim, _ = make_simulation(
        dimensions=dimensions,
        polarization=polarization,
        cpml=absorber == "cpml",
        sponge=absorber == "sponge",
    )
    solver = sim.differentiable()
    state = seeded_state(solver)
    for sigma in (120.0, 240.0):
        ordinary = _with_conductivity(sim, sigma)
        program = ordinary.compile(backend="jax")
        reference_coefficients = ordinary.differentiable().coefficients()
        expected = ordinary.advance(state=state, backend="jax", performance=False).state
        # Yee tuples are total electric losses, so include the ordinary sponge.
        values = (
            jnp.float32(sigma),
            jnp.full(sim.material_grid.shape, sigma),
        )
        if sigma == 120.0:
            values += (None, tuple(getattr(program.grid, f"sig_{a}") for a in "xyz"))
        for conductivity in values:
            coefficients = solver.coefficients(conductivity=conductivity)
            for axis in "xyz":
                for prefix in ("e_decay", "e_source", "h_decay", "h_source"):
                    name = f"{prefix}_{axis}"
                    np.testing.assert_allclose(
                        getattr(coefficients, name),
                        getattr(reference_coefficients, name),
                        rtol=2e-6,
                        atol=1e-12,
                    )
            actual = solver(conductivity=conductivity, state=state)
            for name in ("ex", "ey", "ez", "hx", "hy", "hz"):
                np.testing.assert_allclose(
                    getattr(actual, name), getattr(expected, name), rtol=2e-5, atol=2e-8
                )


@pytest.mark.parametrize("absorber", ["pec", "cpml", "sponge"])
def test_conductivity_gradient_matches_recompiled_forward_model(absorber):
    sim, _ = make_simulation(cpml=absorber == "cpml", sponge=absorber == "sponge")
    solver = sim.differentiable()
    state = seeded_state(solver)

    def energy(out):
        return sum(
            jnp.sum(getattr(out, a) ** 2) for a in ("ex", "ey", "ez")
        ) + 1e5 * sum(jnp.sum(getattr(out, a) ** 2) for a in ("hx", "hy", "hz"))

    gradient = jax.grad(lambda sigma: energy(solver(conductivity=sigma, state=state)))(
        jnp.float32(120)
    )
    for delta in (1.0, 2.0):
        values = [
            energy(
                _with_conductivity(sim, sigma)
                .advance(state=state, backend="jax", performance=False)
                .state
            )
            for sigma in (120 + delta, 120 - delta)
        ]
        finite_difference = (values[0] - values[1]) / (2 * delta)
        np.testing.assert_allclose(gradient, finite_difference, rtol=0.02, atol=2e-7)
    assert abs(float(gradient)) > 1e-7


def make_simulation(
    *,
    dimensions=3,
    cpml=False,
    sponge=False,
    mixed=False,
    monitors=False,
    sources=False,
    timesteps=7,
    metric_kind="isotropic_uniform",
    polarization="tm",
):
    shape = (8, 9, 10) if dimensions == 3 else (9, 10)
    dx = 80e-9
    rng = np.random.default_rng(18)
    epsilon = (1.5 + rng.random(shape)).astype(np.float32)
    if sponge:
        # Keep absorber-normal material constant so this conductivity test does
        # not trigger the separate PML geometry-quality warning.
        epsilon.fill(2.0)
    grid = None
    if metric_kind != "isotropic_uniform":
        from beamz.design.grid import RectilinearGrid
        from beamz.lattice import sample_voxel_grid_at_component_3d

        sizes = shape[::-1]
        widths = [np.full(n, dx * (1.0 + 0.2 * axis)) for axis, n in enumerate(sizes)]
        if metric_kind == "rectilinear":
            widths = [w * np.linspace(1.0, 1.15, w.size) for w in widths]
        grid = RectilinearGrid(*(np.r_[0.0, np.cumsum(w)] for w in widths))
    yee = (
        {}
        if grid is None
        else {
            f"eps_{axis}": np.asarray(
                sample_voxel_grid_at_component_3d(epsilon, f"E{axis}")
            )
            for axis in "xyz"
        }
    )
    materials = MaterialGrid(
        epsilon,
        np.full(shape, 120.0, dtype=np.float32),
        np.float32(1.0),
        dx,
        shape,
        grid=grid,
        yee_materials=yee,
        polarization=polarization if dimensions == 2 else None,
    )
    boundaries = [bz.PEC(edges="all")]
    if cpml or sponge:
        boundaries = [
            bz.PML(
                edges=("left", "top", "back") if mixed else "all",
                thickness=2 * dx,
                formulation="cpml" if cpml else "sponge",
            )
        ]
        if mixed:
            boundaries.append(bz.PEC(edges=("right", "bottom", "front")))
    time = np.arange(timesteps, dtype=np.float64) * 1e-16
    devices = []
    if monitors:
        devices = [
            bz.FluxMonitor(
                center=(5 * dx, 4 * dx, 4 * dx),
                size=(0.0, 4 * dx, 4 * dx),
                freqs=(2e14, 2.5e14),
                name="output",
            )
        ]
    drives = []
    if sources:
        # Use an explicit field-unit source so material changes do not alter an
        # independently solved mode profile or launch normalization.
        probe = bz.Simulation(material_grid=materials, time=time)
        target = probe.initial_state().ez.shape
        index = tuple(slice(n // 2, n // 2 + 1) for n in target)
        drives = [
            bz.CustomSource(
                component="Ez",
                timing="e",
                index=index,
                coeff=np.full((1,) * dimensions, 0.03, dtype=np.float32),
                waveform=np.sin(np.linspace(0.3, 2.0, timesteps)).astype(np.float32),
                target_shape=target,
            )
        ]
    return bz.Simulation(
        material_grid=materials,
        time=time,
        boundaries=boundaries,
        sources=drives,
        monitors=devices,
        polarization=polarization,
    ), (
        jnp.asarray(epsilon)
        if not yee
        else tuple(jnp.asarray(yee[f"eps_{axis}"]) for axis in "xyz")
    )


def seeded_state(run):
    rng = np.random.default_rng(21)
    state = run.initial_state
    updates = {
        name: jnp.asarray(
            rng.normal(size=getattr(state, name).shape), dtype=jnp.float32
        )
        * (0.03 if name.startswith("e") else 0.0001)
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    }
    for name in ("cpml_psi_h_terms", "cpml_psi_e_terms"):
        updates[name] = tuple(
            jnp.asarray(rng.normal(size=v.shape), dtype=jnp.float32) * 1e-3
            for v in getattr(state, name)
        )
    return state._replace(**updates)


def field_loss(state):
    return sum(jnp.sum(getattr(state, name) ** 2) for name in ("ex", "ey", "ez"))


@pytest.mark.parametrize("dimensions,polarization", [(3, "tm"), (2, "tm"), (2, "te")])
@pytest.mark.parametrize("cpml", [False, True])
def test_checkpointed_material_gradient_matches_finite_difference(
    dimensions, polarization, cpml
):
    sim, epsilon = make_simulation(
        dimensions=dimensions, polarization=polarization, cpml=cpml
    )
    solver = sim.differentiable(checkpoint_interval=3)
    state = seeded_state(solver)

    def objective(eps):
        return field_loss(solver(eps, state=state))

    gradient = jax.grad(objective)(epsilon)
    direction = jnp.asarray(
        np.random.default_rng(42).normal(size=epsilon.shape), dtype=jnp.float32
    )
    direction /= jnp.linalg.norm(direction)
    # This random direction nearly cancels in 3D. Two larger centered steps
    # resolve its change above float32 summation noise while keeping individual
    # material perturbations small. Both must agree with the analytic derivative.
    for delta in (0.1, 0.2):
        finite_difference = (
            objective(epsilon + delta * direction)
            - objective(epsilon - delta * direction)
        ) / (2 * delta)
        np.testing.assert_allclose(
            jnp.vdot(gradient, direction), finite_difference, rtol=0.015, atol=2e-6
        )
    assert jnp.linalg.norm(gradient) > 0


@pytest.mark.parametrize("interval", [None, 1, 3, 7, 100])
def test_checkpointing_preserves_values_and_gradients(interval):
    sim, epsilon = make_simulation(cpml=True)
    plain = sim.differentiable(checkpoint_interval=0)
    checkpointed = sim.differentiable(checkpoint_interval=interval)
    state = seeded_state(plain)
    expected = jax.value_and_grad(lambda eps: field_loss(plain(eps, state=state)))(
        epsilon
    )
    actual = jax.value_and_grad(lambda eps: field_loss(checkpointed(eps, state=state)))(
        epsilon
    )
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=2e-7)


def test_material_inputs_reuse_executable_and_are_differentiable():
    sim, epsilon = make_simulation()
    solver = sim.differentiable()
    state = seeded_state(solver)
    traces = []

    @jax.jit
    def objective(eps, sig):
        traces.append(None)
        return field_loss(solver(eps, conductivity=sig, state=state))

    # Count tracing directly rather than depending on JAX's private cache size.
    with jax.disable_jit(False):
        first = objective(epsilon, jnp.float32(120.0))
        second = objective(epsilon * 1.1, jnp.float32(130.0))
        assert first != second
        assert len(traces) == 1
        gradient = jax.grad(objective, argnums=1)(epsilon, jnp.float32(120.0))
        delta = jnp.float32(0.5)
        fd = (objective(epsilon, 120.0 + delta) - objective(epsilon, 120.0 - delta)) / (
            2 * delta
        )
        np.testing.assert_allclose(gradient, fd, rtol=0.015, atol=2e-7)


def test_monitor_and_source_objectives_have_end_to_end_gradients():
    sim, epsilon = make_simulation(cpml=True, sources=True, monitors=True)
    solver = sim.differentiable(checkpoint_interval=3)

    def objective(eps, amplitude):
        state = solver(eps, source_amplitudes=amplitude)
        fields = solver.monitor_fields(state, "output")
        return jnp.sum(jnp.abs(fields) ** 2)

    value, gradients = jax.value_and_grad(objective, argnums=(0, 1))(
        epsilon, jnp.ones((1,), dtype=jnp.float32)
    )
    assert value > 0
    assert jnp.linalg.norm(gradients[0]) > 0
    np.testing.assert_allclose(gradients[1][0], 2 * value, rtol=2e-5)
    state = solver(epsilon)
    assert solver.monitor_flux(state, "output").shape == (2,)


@pytest.mark.parametrize("interval", [-1, 2.5, True])
def test_invalid_checkpoint_interval_is_rejected(interval):
    sim, _ = make_simulation()
    with pytest.raises(ValueError, match="checkpoint_interval"):
        sim.differentiable(checkpoint_interval=interval)


@pytest.mark.parametrize("dimensions,polarization", [(3, "tm"), (2, "tm"), (2, "te")])
def test_scalar_conductivity_matches_uniform_material_array(dimensions, polarization):
    from dataclasses import replace

    sim, epsilon = make_simulation(dimensions=dimensions, polarization=polarization)
    scalar_grid = replace(sim.material_grid, conductivity=np.float32(120.0))
    scalar_sim = bz.Simulation(
        material_grid=scalar_grid,
        time=sim.time,
        boundaries=sim.boundaries,
        polarization=polarization,
    )
    reference = sim.differentiable()
    actual = scalar_sim.differentiable()
    state = seeded_state(reference)
    np.testing.assert_allclose(
        field_loss(actual(epsilon, state=state)),
        field_loss(reference(epsilon, state=state)),
        rtol=2e-6,
    )


def test_monitor_flux_matches_analysis_result():
    sim, epsilon = make_simulation(sources=True, monitors=True)
    solver = sim.differentiable()
    state = seeded_state(solver)
    out = solver(epsilon, state=state)
    from beamz.simulation.observe import monitor_dft_flux
    from beamz.simulation.results import MonitorResults

    result = MonitorResults.from_compiled_state(
        sim.monitors[0], solver.program.monitors[0], out, solver.program.config
    )
    np.testing.assert_allclose(
        solver.monitor_flux(out, "output"),
        monitor_dft_flux(result),
        rtol=2e-5,
        atol=1e-18,
    )


def test_partitioned_monitor_scan_preserves_ragged_windows_and_gradients():
    from dataclasses import replace

    from beamz.simulation.differentiation import DifferentiableSimulation
    from beamz.simulation.execute import build_scan

    sim, epsilon = make_simulation(cpml=True, monitors=True, sources=True, timesteps=11)
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
    program = sim.compile(backend="jax")
    auxiliary = replace(
        program.monitors[1], dft_t_start=2e-16, dft_t_end=9e-16, dft_window_code=1
    )
    program = replace(program, monitors=(program.monitors[0], auxiliary))
    solver = DifferentiableSimulation(program, checkpoint_interval=3)
    state = seeded_state(solver)
    packed_run = build_scan(program)
    expected = packed_run(state, solver.coefficients(epsilon))
    actual = solver(epsilon, state=state)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        # Layout constraints in the ordinary scan can change float32 stencil
        # reassociation. Scale the absolute tolerance by the array amplitude so
        # near-cancelled cells do not dominate the comparison.
        amplitude = float(jnp.max(jnp.abs(e), initial=0.0))
        np.testing.assert_allclose(a, e, rtol=3e-5, atol=max(2e-7, 2e-6 * amplitude))

    def objective(out):
        return jnp.sum(jnp.abs(solver.monitor_fields(out, "output")) ** 2)

    plain_grad = jax.grad(
        lambda eps: objective(packed_run(state, solver.coefficients(eps)))
    )(epsilon)
    split_grad = jax.grad(lambda eps: objective(solver(eps, state=state)))(epsilon)
    np.testing.assert_allclose(split_grad, plain_grad, rtol=3e-5, atol=2e-7)
    assert isinstance(actual.dft_vec_re, jax.Array)


def test_differentiable_clock_uses_initial_origin_without_accumulated_drift():
    sim, epsilon = make_simulation(timesteps=257)
    solver = sim.differentiable(backend="jax", checkpoint_interval=16)
    dt = jnp.float32(solver.program.config.dt)
    initial_time = jnp.float32(0.37) * dt
    initial = solver.initial_state._replace(t=initial_time, current_step=jnp.int32(19))

    def terminal_time(origin):
        return solver(epsilon, state=initial._replace(t=origin)).t

    value, gradient = jax.value_and_grad(terminal_time)(initial_time)
    expected = initial_time + jnp.float32(solver.program.config.num_steps) * dt
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(gradient, jnp.float32(1.0))
