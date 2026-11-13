"""Independent small-grid enumeration checks the finite payload factorization."""

import itertools
import math

import pytest

from scripts.cuda_stencil_trace import trace_yee_step
from scripts.model_cuda_systolic_traffic import (
    finite_case,
    geometry,
    grid_for,
    verify_interior,
)
from scripts.prove_cuda_circular_asymmetric import prove as prove_asymmetric
from scripts.prove_cuda_circular_cache import prove
from scripts.prove_cuda_circular_geometry import prove as prove_index
from scripts.prove_cuda_circular_half_shared import prove as prove_half_shared
from scripts.prove_cuda_circular_warp_cache import prove as prove_warp_cache
from scripts.prove_cuda_moving_cpml_inplace import prove as prove_cpml
from scripts.prove_cuda_moving_inplace import prove as prove_inplace
from scripts.prove_cuda_systolic_schedule import component_shapes


@pytest.mark.parametrize("tile", ((4, 8, 32), (8, 8, 16), (15, 15, 31)))
def test_interior_formula_against_traced_operand_enumeration(tile):
    actual = verify_interior(tile)
    assert actual["requests"] > actual["unique"]


@pytest.mark.parametrize("n,tile,steps", ((3, (2, 3, 4), 2), (4, (3, 2, 3), 3)))
def test_finite_requested_payload_matches_independent_brute_force(n, tile, steps):
    shapes = component_shapes((n, n, n))
    grid = grid_for(n, steps, tile)
    reads = valid_publications = 0

    def valid(field, p):
        return all(0 <= a < b for a, b in zip(p, shapes[field], strict=True))

    for version in range(1, steps + 1):
        for block in itertools.product(*(range(v) for v in grid)):
            origin = tuple(a * b for a, b in zip(block, tile, strict=True))
            for local in itertools.product(*(range(v) for v in tile)):
                p = tuple(a + b - version for a, b in zip(origin, local, strict=True))
                for update in trace_yee_step(version - 1):
                    if not valid(update.output.field, p):
                        continue
                    if any(a == b - 1 for a, b in zip(local, tile, strict=True)):
                        valid_publications += 1
                    for read in update.reads:
                        q = tuple(a + b for a, b in zip(p, read.offset, strict=True))
                        if not valid(read.field, q):
                            continue
                        source_u = tuple(a + read.version for a in q)
                        source_block = tuple(a // b for a, b in zip(source_u, tile, strict=True))
                        reads += read.version == 0 or source_block != block
    model = finite_case(n, steps, tile, 700)
    assert model["scenarios"]["current_direct_requested"]["field_read_floats"] == reads
    assert model["valid_publication_floats"] == valid_publications
    assert model["publication_floats"] == math.prod(grid) * steps * 6 * geometry(tile)["outgoing_high_union"]


def test_circular_physical_slots_and_required_barriers():
    prove(steps=4)
    for option in ("early_import", "stale_h", "publish_all_owned"):
        with pytest.raises(AssertionError):
            prove(steps=4, **{option: True})


def test_owned_inplace_initial_antidependencies_and_unadmitted_halo():
    prove_inplace(steps=1)
    with pytest.raises(AssertionError):
        prove_inplace(steps=1, initial_low_halo=2)


def test_paper_scaled_cache_partial_tail_and_nonpower_owned_indices():
    result = prove_index((7, 7, 15), steps=64)
    assert result["owned_cells"] == 735
    assert result["skipped_tail_slots"] == 33
    for option in ("old_owned_mask", "omit_tail_guard", "wrong_backward_sign", "missing_head_roll"):
        with pytest.raises(AssertionError):
            prove_index((7, 7, 15), **{option: True})


def test_paper_scaled_cache_version_dependencies():
    prove((7, 7, 15), (23, 25, 47), steps=8)


def test_global_packed_cpml_local_recurrence_order_and_alias_controls():
    prove_cpml(padded=True)
    for option in ("reverse_ranks", "reverse_time", "alias_faces", "alias_phases"):
        with pytest.raises(AssertionError):
            prove_cpml(**{option: True})


def test_half_shared_register_ownership_and_phase_controls():
    prove_half_shared(local_centers=True, local_z=True)
    for option in ("overwrite_E_early", "overwrite_H_early", "stale_import_H", "register_neighbors"):
        with pytest.raises(AssertionError):
            prove_half_shared(**{option: True})
    with pytest.raises(AssertionError):
        prove_half_shared(local_z=True, wrong_z_direction=True)
    # Two XY planes per thread group break local-register Z ownership.
    with pytest.raises(AssertionError, match="Cross-thread register read"):
        prove_half_shared(threads=32, local_z=True)


def test_warp_x_full_participation_versions_and_two_component_frame():
    result = prove_warp_cache(steps=2)
    assert result["shared_read_scalars"] == 4 * 6975 * 2
    assert result["shuffle_read_scalars"] == 4 * 6975 * 2
    assert result["shuffle_reads_from_unowned_slots"] > 0
    assert result["shared_frame_bytes"] == 65536
    for option in ("branch_shuffle", "bad_wrap", "stale_e_shuffle",
                   "stale_h_shuffle", "omit_shared_component"):
        with pytest.raises(AssertionError):
            prove_warp_cache(steps=1, **{option: True})


def test_asymmetric_permanent_cache_lifetimes_and_nonpower_z():
    result = prove_asymmetric(steps=2)
    assert result["state_register_scalars_per_thread"] == 12
    assert result["total_shared_bytes"] == 49220
    assert result["shuffle_reads_from_unowned_slots"] > 0
    for option in ("early_import", "stale_h", "branch_shuffle",
                   "wrong_z_mask", "publish_all_owned", "counter_before_b1",
                   "omit_e_minus_one"):
        with pytest.raises(AssertionError):
            prove_asymmetric(steps=2, **{option: True})
