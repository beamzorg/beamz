"""Four contiguous-Z groups require real predecessor snapshots and16-lane rows."""

import pytest

from scripts.prove_cuda_cube_paired_geometry import Geometry, Hazard, simulate


@pytest.mark.parametrize("g", [Geometry(), Geometry(8, 16, 32)])
def test_mapping_is_a_complete4096_point_bijection(g):
    seen = {g.point(tid, j) for tid in range(1024) for j in range(4)}
    assert len(seen) == 4096
    for p in seen:
        tid, j = g.owner(p)
        assert g.point(tid, j) == p
    assert g.local_z == 4 and 2 * g.threads == 2048


@pytest.mark.parametrize("version", [1, 2, 7, 8, 15, 16, 31, 32])
@pytest.mark.parametrize("g", [Geometry(), Geometry(8, 16, 32)])
def test_actual_affine_reads_survive_wraps_ghosts_and_early_next_pair_preparation(g, version):
    proof = simulate(g, version=version, seed=37)
    assert proof["barriers"] == 5 and proof["field_register_lower_bound"] == 20
    assert proof["cross_chunk_snapshot_reads"] > 0
    assert proof["checked_reads"] > 100_000


@pytest.mark.parametrize("negative", ["old-xor1-snapshot", "stale-snapshot", "wrong-shuffle-width", "shuffle-only-owned", "save-old-after-H", "restore-before-P23", "restore-before-RL"])
def test_wrong_generalization_has_actual_read_or_collective_failure(negative):
    with pytest.raises(Hazard):
        simulate(version=2, seed=13, negative=negative)


def test_predecessor_group_wrap_is_mod4_and_width16_keeps_upperwarp_row():
    g = Geometry()
    assert [(group - 1) % g.groups for group in range(g.groups)] == [3, 0, 1, 2]
    for tid in (0, 15, 16, 31, 32, 511, 512, 1023):
        z, y, x = g.point(tid, 0)
        lane = tid % 32
        peer = tid // 32 * 32 + lane // 16 * 16 + ((x - 1) % 16)
        assert g.point(peer, 0) == (z, y, (x - 1) % 16)


@pytest.mark.parametrize("shape", [(16, 16, 16, 512), (8, 32, 16, 1024), (16, 16, 32, 1024)])
def test_oracle_does_not_silently_admit_other_mapping_classes(shape):
    with pytest.raises(ValueError):
        Geometry(*shape)
