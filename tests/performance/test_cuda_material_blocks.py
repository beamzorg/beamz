"""Native constant-block updates must preserve the preceding CUDA operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    ExtrudedPolygon,
    Grid,
    Material,
    Object,
    Polygon,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.compile import _pack_cuda_lossless_e_coefficients
from beamz.simulation.cuda.material_blocks import pack_material_blocks
from beamz.simulation.execute import (
    build_program_scan,
    build_step_context,
    forward_step,
)
from beamz.simulation.kernels import select_update_kernel
from beamz.simulation.sharding import place_tree
from tests.jax_compat import enable_x64
from tests.performance.test_cuda_tensor import compare_states, seeded_state


def simulation(size, steps, *, full=False, graded=False, bulk=False, pec=False):
    length = size * 100e-9
    u = np.linspace(0, 1, size + 1)
    edges = length * (u + 0.08 * np.sin(2 * np.pi * u)) if graded else length * u
    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    vertices = (
        np.array([[-0.5, -1 / 3], [0.5, -1 / 3], [0.5, 1 / 3], [-0.5, 1 / 3]])
        * 0.25
        @ rotation.T
        + 0.5
    ) * length
    scene = (
        Scene((Material(((6, 0.2, 0.1), (0.2, 5, 0.3), (0.1, 0.3, 4))),), ())
        if bulk
        else Scene(
            (Material(), Material(6)),
            (
                Object(
                    ExtrudedPolygon(Polygon(tuple(map(tuple, vertices))), 0, length),
                    1,
                ),
            ),
        )
    )
    material = MaterialGrid.from_raster_result(
        rasterize(
            scene,
            Grid(edges, edges, length * u),
            options=RasterOptions(
                smoothing="farjadpour_full" if full or bulk else "farjadpour_diagonal"
            ),
        )
    )
    return bz.Simulation(
        material_grid=material,
        time=np.arange(steps + 1) * material.grid.cfl_time_step(0.6),
        boundaries=(
            [bz.PEC(edges="all")]
            if pec
            else [bz.PML(thickness=300e-9 if size < 64 else 800e-9, formulation="cpml")]
        ),
        monitors=[
            bz.FluxMonitor(
                center=(0.5 * length, 0.5 * length, z * length),
                size=(0.7 * length, 0.7 * length, 0),
                freqs=[bz.LIGHT_SPEED / 1.55e-6],
                name=f"flux{i}",
            )
            for i, z in enumerate((0.35, 0.65))
        ],
        setup_device="cpu",
    )


def restore_previous_layout(coeffs, state):
    """Independent host reconstruction, also used for native equality checks."""
    changes = {}
    full = bool(coeffs.e_inverse_offdiagonal.size)
    dense = []
    for c in "xyz":
        shape = getattr(state, "e" + c).shape
        source = getattr(coeffs, "e_source_" + c)
        value = getattr(coeffs, ("e_inverse_diagonal_" if full else "e_decay_") + c)
        if source.ndim == 2 and source.dtype == jnp.int32:
            headers = np.asarray(source).ravel()
            linear = np.arange(np.prod(shape))
            header = headers[linear // 32]
            offsets = np.where(header < 0, ~header, header + linear % 32)
            value = np.asarray(value)[offsets].reshape(shape)
        elif full:
            value = np.broadcast_to(np.asarray(value), shape).copy()
        elif source.ndim == 1 and source.dtype == jnp.int32:
            words = np.asarray(source).view(np.uint32)
            linear = np.arange(np.prod(shape))
            ids = (words[linear // 4] >> (8 * (linear % 4))) & 255
            value = np.asarray(value)[ids].reshape(shape)
        else:
            value = source
        if full:
            changes["e_inverse_diagonal_" + c] = jnp.asarray(value)
            changes["e_source_" + c] = jnp.float32(1)
        else:
            dense.append(jnp.asarray(value))
    if not full:
        packed = _pack_cuda_lossless_e_coefficients(
            (jnp.float32(1),) * 3, dense, blocks=False
        )
        decays, sources = (
            packed if packed is not None else ((jnp.float32(1),) * 3, dense)
        )
        for c, decay, source in zip("xyz", decays, sources, strict=True):
            changes["e_decay_" + c] = decay
            changes["e_source_" + c] = source
    return coeffs._replace(**changes)


@pytest.fixture(autouse=True)
def require_cuda():
    status = cuda_backend_status()
    if not status.available:
        pytest.skip(status.reason)
    with enable_x64(False):
        yield


@pytest.mark.parametrize("full,bulk", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("pec", [False, True])
def test_blocks_are_bitwise_equal_to_previous_cuda_and_match_jax(
    full, bulk, graded, pec
):
    sim = simulation(66, 5, full=full, bulk=bulk, graded=graded, pec=pec)
    p = sim.compile(num_steps=5, backend="cuda_streamed")
    state = seeded_state(p)
    coeffs = place_tree(p, p.coefficients)
    previous = place_tree(p, restore_previous_layout(p.coefficients, state))
    if bulk:
        assert all(getattr(coeffs, "e_inverse_diagonal_" + c).ndim == 0 for c in "xyz")
    elif full:
        assert all(getattr(coeffs, "e_source_" + c).ndim == 2 for c in "xyz")
    else:
        # Byte IDs can be smaller for small interfaces; both layouts are exact.
        assert all(getattr(coeffs, "e_source_" + c).ndim in (1, 2) for c in "xyz")
    run = build_program_scan(p)
    a, b = jax.block_until_ready((run(state, previous), run(state, coeffs)))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    reference = sim.compile(num_steps=5, backend="jax")
    output = build_program_scan(reference)(
        state, place_tree(reference, reference.coefficients)
    )
    compare_states(jax.block_until_ready(output), b)


@pytest.mark.parametrize("full", [False, True])
def test_blocks_phase_path_and_dynamic_coefficient_values(full):
    p = simulation(66, 3, full=full, graded=True).compile(
        num_steps=3, backend="cuda_streamed"
    )
    state = seeded_state(p)
    coeffs = place_tree(p, p.coefficients)
    if not full:
        # Exercise the block reader even when the conservative size heuristic
        # prefers byte IDs on this small graded grid.
        reference = restore_previous_layout(coeffs, state)
        values = []
        for c in "xyz":
            ids = np.asarray(getattr(reference, "e_source_" + c)).view(np.uint32)
            linear = np.arange(getattr(state, "e" + c).size)
            codes = (ids[linear // 4] >> (8 * (linear % 4))) & 255
            values.append(
                np.asarray(getattr(reference, "e_decay_" + c))[codes].reshape(
                    getattr(state, "e" + c).shape
                )
            )
        pools, headers = pack_material_blocks(values, max_bytes=np.inf)
        coeffs = coeffs._replace(
            **{
                name + c: jnp.asarray(v)
                for name, items in (("e_decay_", pools), ("e_source_", headers))
                for c, v in zip("xyz", items, strict=True)
            }
        )
    assert coeffs.e_source_x.ndim == 2
    context = build_step_context(p)
    kernel = select_update_kernel(context)

    def steps(state, coeffs):
        for _ in range(3):
            state = forward_step(
                state, program=p, ctx=context, coeffs=coeffs, update_kernel=kernel
            )
        return state

    phase, graph = jax.jit(steps), build_program_scan(p)
    prefix = "e_inverse_diagonal_" if full else "e_decay_"
    for scale in (1.0, 0.5, 1.2):
        changed = coeffs._replace(
            **{prefix + c: scale * getattr(coeffs, prefix + c) for c in "xyz"}
        )
        reference = place_tree(p, restore_previous_layout(changed, state))
        a, b, c = jax.block_until_ready(
            (phase(state, changed), graph(state, changed), graph(state, reference))
        )
        compare_states(a, b)
        for x, y in zip(jax.tree.leaves(b), jax.tree.leaves(c), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_blocks_temporal_pair(monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_TEMPORAL_STEPS", "2")
    p = simulation(128, 6).compile(num_steps=6, backend="cuda_streamed")
    state = seeded_state(p)
    assert p.coefficients.e_source_x.ndim == 2
    previous = restore_previous_layout(p.coefficients, state)
    run = build_program_scan(p)
    a, b = jax.block_until_ready(
        tuple(run(state, place_tree(p, c)) for c in (previous, p.coefficients))
    )
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_tuned_rotation_restores_supported_material_storage():
    from beamz.simulation.cuda.tuning import _apply

    p = simulation(128, 4).compile(num_steps=4, backend="cuda_streamed")
    assert p.coefficients.e_source_x.ndim == 2
    rotated = _apply(p, dict(axes=(1, 2, 0), shell_tile="32x8"))
    assert rotated.coefficients.e_source_x.ndim in (1, 3)
    state = seeded_state(p)
    a, b = jax.block_until_ready(
        tuple(
            build_program_scan(plan)(state, place_tree(plan, plan.coefficients))
            for plan in (p, rotated)
        )
    )
    compare_states(a, b)


def test_diagonal_block_pool_gradients_match_previous_encoding():
    from beamz.simulation.differentiation import build_differentiable_scan

    p = simulation(128, 2).compile(num_steps=2, backend="cuda_streamed")
    state = seeded_state(p)
    coeffs = place_tree(p, p.coefficients)
    assert coeffs.e_source_x.ndim == 2
    previous = place_tree(p, restore_previous_layout(coeffs, state))
    run = build_differentiable_scan(p, checkpoint_interval=1)

    def loss(scale, c):
        # Both encodings carry their source coefficient pool in e_decay_*.
        c = c._replace(
            **{
                f"e_decay_{axis}": scale * getattr(c, f"e_decay_{axis}")
                for axis in "xyz"
            }
        )
        result = run(state, c)
        return jnp.mean(result.ex**2)

    value_gradient = jax.jit(jax.value_and_grad(loss))
    a, b = jax.block_until_ready(
        (
            value_gradient(jnp.float32(1), coeffs),
            value_gradient(jnp.float32(1), previous),
        )
    )
    np.testing.assert_array_equal(a[0], b[0])
    # Pool gradients sum millions of contributions in different FP32 scatter
    # orders for byte IDs and constant blocks; primals remain bitwise equal.
    np.testing.assert_allclose(a[1], b[1], rtol=2e-4, atol=1e-12)
    primal = jax.jit(loss)
    step = 5e-3
    high, low = jax.block_until_ready(
        (primal(jnp.float32(1 + step), coeffs), primal(jnp.float32(1 - step), coeffs))
    )
    finite_difference = (float(high) - float(low)) / (2 * step)
    np.testing.assert_allclose(a[1], finite_difference, rtol=5e-4, atol=1e-12)
    np.testing.assert_allclose(b[1], finite_difference, rtol=5e-4, atol=1e-12)
    assert float(a[1]) != 0
