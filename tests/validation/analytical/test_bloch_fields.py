"""Phase seams against extended-domain Yee updates, including complex media."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.lattice import component_axis_offsets_3d
from beamz.simulation.compile.material_blocks import update_dispersion


def _seed(sim, vector):
    state = sim.initial_state()
    program = sim.compile(backend="jax")
    geometry = program.grid.geometry
    updates = {}
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        value = getattr(state, component.lower())
        if value.size <= 1:
            continue
        axes = "zyx" if value.ndim == 3 else "yx"
        offsets = component_axis_offsets_3d(component)
        phase = np.zeros(value.shape)
        for axis_index, axis in enumerate(axes):
            coordinates = (
                geometry.centers(axis)
                if offsets[axis] == 0.5
                else geometry.axis_edges(axis)
            )
            shape = [1] * value.ndim
            shape[axis_index] = coordinates.size
            phase += vector["xyz".index(axis)] * coordinates.reshape(shape)
        scale = (
            0.1 * (1 + "xyz".index(component[-1])) / (377 if component[0] == "H" else 1)
        )
        from beamz.simulation.compile.physics import apply_zero_mask

        updates[component.lower()] = apply_zero_mask(
            jnp.asarray(scale * np.exp(1j * phase), dtype=jnp.complex64),
            getattr(program.boundary.metallic, component.lower() + "_mask"),
        )
    return state._replace(**updates)


@pytest.mark.validation
@pytest.mark.parametrize("polarization", ["tm", "te", "3d"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_bloch_step_matches_extended_complex_fourier_field(polarization, sign):
    # A larger domain with ordinary walls is independent of the seam machinery.
    # At its center, multiple steps reproduce the Bloch cell with arbitrary phase.
    dl = 40e-9
    ny, nx, nz = 4, 5, 4
    vector = (sign * 2.3e6, -1.7e6, 0.0)

    def make_sim(tiled):
        factor = 3 if tiled else 1
        size = (
            nx * dl * factor,
            ny * dl * factor,
            nz * dl if polarization == "3d" else 0,
        )
        boundaries = [bz.PEC(edges=("front", "back"))] if polarization == "3d" else []
        if not tiled:
            boundaries.append(bz.Bloch(axes=("x", "y"), wavevector=vector))
        return bz.Simulation(
            size=size,
            resolution=dl,
            polarization="tm" if polarization == "3d" else polarization,
            boundaries=boundaries,
            run_time=1e-15,
        )

    small, large = make_sim(False), make_sim(True)
    actual, expected = _seed(small, vector), _seed(large, vector)
    # Shift the extended field so its center has the cell's physical phase origin.
    shift = np.exp(-1j * (vector[0] * nx * dl + vector[1] * ny * dl))
    expected = expected._replace(
        **{
            c: getattr(expected, c) * shift
            for c in ("ex", "ey", "ez", "hx", "hy", "hz")
        }
    )
    for _ in range(2):
        actual = small.step(actual, backend="jax")
        expected = large.step(expected, backend="jax")
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        value = getattr(actual, component)
        if value.size <= 1:
            continue
        slices = (slice(None),) if value.ndim == 3 else ()
        slices += (slice(ny, ny + value.shape[-2]), slice(nx, nx + value.shape[-1]))
        np.testing.assert_allclose(
            value, getattr(expected, component)[slices], rtol=8e-6, atol=1e-7
        )


@pytest.mark.validation
@pytest.mark.parametrize("polarization", ["tm", "te"])
def test_complex_ade_response_and_split_continuation(polarization):
    medium = bz.PoleResidue.lorentz(
        1.0, strength=2, resonance=6e15, damping=8e14, frequency_range=(4e14, 8e14)
    )
    dt, frequency, steps = 1e-17, 5e14, 6000
    sim = bz.Simulation(
        design=bz.Design(width=160e-9, height=160e-9, background=medium),
        resolution=40e-9,
        time=np.arange(steps) * dt,
        polarization=polarization,
        boundaries=[bz.Bloch(wavevector=(2e6, -1e6, 0))],
    )
    program = sim.compile(backend="jax")
    state = sim.initial_state()
    name = "ez" if polarization == "tm" else "ex"

    def step(state, i):
        old = state
        delta = jnp.exp(-1j * 2 * np.pi * frequency * dt * (i + 1)) - jnp.exp(
            -1j * 2 * np.pi * frequency * dt * i
        )
        free = state._replace(**{name: getattr(state, name) + delta})
        state = update_dispersion(old, free, program.dispersion)
        return state, jnp.mean(getattr(state, name))

    _, trace = jax.jit(lambda s: jax.lax.scan(step, s, jnp.arange(steps)))(state)
    t = np.arange(1, steps + 1) * dt
    measured = np.mean(
        np.asarray(trace)[-2000:] * np.exp(1j * 2 * np.pi * frequency * t[-2000:])
    )
    np.testing.assert_allclose(
        measured, 1 / medium.eps_model(frequency), rtol=0.003, atol=0.001
    )
    initial = _seed(sim, (2e6, -1e6, 0))
    full = sim.advance(
        state=initial, num_steps=20, backend="jax", performance=False
    ).state
    half = sim.advance(
        state=initial, num_steps=10, backend="jax", performance=False
    ).state
    resumed = sim.advance(
        state=half, num_steps=10, backend="jax", performance=False
    ).state
    for x, y in zip(jax.tree.leaves(full), jax.tree.leaves(resumed), strict=True):
        np.testing.assert_allclose(x, y, rtol=3e-6, atol=1e-6)
