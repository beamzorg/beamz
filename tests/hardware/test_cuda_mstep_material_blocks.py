"""Native packed-core regression for exact constant-block coefficient storage."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beamz.simulation.cuda.material_blocks import pack_material_blocks

from beamz.simulation.compile import clear_program_cache
from beamz.simulation.execute import build_scan
from tests.hardware.test_cuda_mstep import STATUS, _compare, _run, _seed, _simulation

pytestmark = pytest.mark.skipif(
    not STATUS.available, reason=STATUS.reason or "CUDA component unavailable"
)


def test_mstep_mixed_psi_precision_matches_dynamic_streamed():
    """Typed edge dispatch must retain per-term decoding for mixed psi state."""
    import beamz as bz

    clear_program_cache()
    sim = _simulation(shape=(59, 81, 113), steps=5)
    sim = sim.updated_copy(
        boundaries=(
            bz.PML(edges="all", thickness=12 * sim.resolution, formulation="cpml"),
        )
    )
    state = _seed(sim.compile(backend="cuda_streamed", num_steps=5))
    state = state._replace(
        cpml_psi_h_terms=tuple(
            jnp.asarray(a, dtype=jnp.bfloat16 if term % 2 else jnp.float32)
            for term, a in enumerate(state.cpml_psi_h_terms)
        ),
        cpml_psi_e_terms=tuple(
            jnp.asarray(a, dtype=jnp.float32 if term % 2 else jnp.bfloat16)
            for term, a in enumerate(state.cpml_psi_e_terms)
        ),
    )
    _compare(_run(sim, state, "cuda_streamed", 5), _run(sim, state, "cuda_mstep", 5))


@pytest.mark.parametrize("steps", [2, 5])
@pytest.mark.parametrize("ranks", [(2, 2, 2), (1, 2, 1)])
def test_mstep_constant_block_headers_match_byte_codebook(steps, ranks):
    clear_program_cache()
    sim = _simulation(shape=(43, 65, 97), steps=steps)
    program = sim.compile(backend="cuda_mstep", num_steps=steps)
    shape = tuple(getattr(program.grid, "E" + c).shape for c in "xyz")
    dense = []
    for c, dims in zip("xyz", shape, strict=True):
        pool = np.asarray(getattr(program.coefficients, "e_decay_" + c))
        words = np.asarray(getattr(program.coefficients, "e_source_" + c))
        assert pool.ndim == 1 and words.ndim == 1
        linear = np.arange(np.prod(dims), dtype=np.int64)
        codes = (words.view(np.uint32)[linear >> 2] >> (8 * (linear & 3))) & 255
        dense.append(pool[codes].reshape(dims))
    blocks = pack_material_blocks(tuple(dense), max_bytes=float("inf"))
    assert blocks is not None
    pools, headers = blocks
    changes = {}
    for c, rank, pool, header in zip("xyz", ranks, pools, headers, strict=True):
        if rank == 2:
            changes["e_decay_" + c] = pool
            changes["e_source_" + c] = header
    candidate = replace(program, coefficients=program.coefficients._replace(**changes))
    state = _seed(program)
    reference_state = jax.tree.map(lambda a: np.array(a, copy=True), state)
    actual_state = jax.tree.map(lambda a: np.array(a, copy=True), state)
    reference = jax.block_until_ready(
        build_scan(program)(reference_state, program.coefficients)
    )
    actual = jax.block_until_ready(
        build_scan(candidate)(actual_state, candidate.coefficients)
    )
    _compare(reference, actual)


@pytest.mark.parametrize("first_backend", ["cuda_streamed", "cuda_mstep"])
def test_mstep_material_preparation_cache_order_preserves_results(first_backend):
    """Shared preparation may reuse the first schedule's coefficient encoding."""
    clear_program_cache()
    sim = _simulation(shape=(43, 65, 97), steps=5)
    second_backend = (
        "cuda_mstep" if first_backend == "cuda_streamed" else "cuda_streamed"
    )
    first = sim.compile(backend=first_backend, num_steps=5)
    second = sim.compile(backend=second_backend, num_steps=5)
    programs = {first_backend: first, second_backend: second}
    expected_rank = 2 if first_backend == "cuda_streamed" else 1
    for program in (first, second):
        assert all(
            getattr(program.coefficients, "e_source_" + c).ndim == expected_rank
            for c in "xyz"
        )
    state = _seed(programs["cuda_mstep"])
    reference_program, candidate_program = (
        programs["cuda_streamed"],
        programs["cuda_mstep"],
    )
    reference = jax.block_until_ready(
        build_scan(reference_program)(
            jax.tree.map(lambda a: np.array(a, copy=True), state),
            reference_program.coefficients,
        )
    )
    actual = jax.block_until_ready(
        build_scan(candidate_program)(
            jax.tree.map(lambda a: np.array(a, copy=True), state),
            candidate_program.coefficients,
        )
    )
    _compare(reference, actual)
