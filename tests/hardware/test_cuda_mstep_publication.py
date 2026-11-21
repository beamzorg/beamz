"""Public ownership/observation gates for in-place edge publication schedules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.simulation.compile import clear_program_cache
from tests.hardware.test_cuda_mstep import STATUS, _compare, _run, _seed, _simulation

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not STATUS.available, reason=STATUS.reason or "CUDA unavailable"
    ),
]


def _snapshot(state):
    return [np.array(leaf, copy=True) for leaf in jax.tree.leaves(state)]


def _assert_unchanged(state, expected):
    for actual, saved in zip(jax.tree.leaves(state), expected, strict=True):
        np.testing.assert_array_equal(actual, saved)


def test_mstep_small_cpml12_sanitizer():
    """One seeded pair exercises a real partial core with bounded tool cost."""
    from beamz.simulation.cuda.mstep import build_mstep_plan

    clear_program_cache()
    sim = _simulation(shape=(35, 49, 65), steps=2)
    sim = sim.updated_copy(
        sources=(),
        monitors=(),
        boundaries=(
            bz.PML(edges="all", thickness=12 * sim.resolution, formulation="cpml"),
        ),
    )
    program = sim.compile(backend="cuda_mstep", num_steps=2)
    shapes = tuple(
        program.boundary.logical_component_shapes[c]
        for c in ("Hx", "Hy", "Hz", "Ex", "Ey", "Ez")
    )
    plan = build_mstep_plan(shapes, program.boundary.cpml, (None,) * 9)
    assert plan.core_cells > 0
    state = _seed(program)
    _compare(_run(sim, state, "cuda_streamed", 2), _run(sim, state, "cuda_mstep", 2))


@pytest.mark.parametrize("source", ["gaussian", "mode"])
@pytest.mark.parametrize("steps", [2, 3, 4, 5, 17])
def test_published_edges_preserve_initial_state_repeats_and_split_clocks(source, steps):
    sim = _simulation(shape=(59, 81, 113), steps=33, source=source)
    sim = sim.updated_copy(
        boundaries=(
            bz.PML(edges="all", thickness=12 * sim.resolution, formulation="cpml"),
        )
    )
    # Deliberately prepare streamed first to exercise shared constant-block
    # material metadata in the mstep fast path as well as public input aliases.
    initial = _seed(sim.compile(backend="cuda_streamed", num_steps=steps))
    initial = jax.tree.map(jnp.asarray, initial)
    saved = _snapshot(initial)

    def advance(state, count, backend="cuda_mstep"):
        return jax.block_until_ready(
            sim.advance(
                state=state, num_steps=count, backend=backend, performance=False
            ).state
        )

    whole = advance(initial, steps)
    _assert_unchanged(initial, saved)
    repeat = advance(initial, steps)
    _assert_unchanged(initial, saved)
    for a, b in zip(jax.tree.leaves(whole), jax.tree.leaves(repeat), strict=True):
        np.testing.assert_array_equal(a, b)
    _compare(advance(initial, steps, "cuda_streamed"), whole)
    _assert_unchanged(initial, saved)
    first = advance(initial, 1 if steps == 2 else 3 if steps > 3 else 1)
    count = 1 if steps == 2 else 3 if steps > 3 else 1
    first_saved = _snapshot(first)
    split = advance(first, steps - count)
    _assert_unchanged(first, first_saved)
    _compare(whole, split)


@pytest.mark.parametrize("axes", ["120", "201"])
def test_published_edges_rotate_byte_ids_after_mstep_first_preparation(
    monkeypatch, axes
):
    clear_program_cache()
    sim = _simulation(shape=(59, 81, 113), steps=7)
    sim = sim.updated_copy(
        boundaries=(
            bz.PML(edges="all", thickness=12 * sim.resolution, formulation="cpml"),
        )
    )
    program = sim.compile(backend="cuda_mstep", num_steps=5)
    assert all(getattr(program.coefficients, "e_source_" + c).ndim == 1 for c in "xyz")
    state = _seed(program)
    reference = _run(sim, state, "cuda_streamed", 5)
    monkeypatch.setenv("BEAMZ_CUDA_STORAGE_AXES", axes)
    _compare(reference, _run(sim, state, "cuda_mstep", 5))
