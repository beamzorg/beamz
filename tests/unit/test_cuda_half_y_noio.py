"""Half-Y NoIO keeps the interleaved four-slot reader protocol."""

import pytest

from scripts.prove_cuda_half_y_noio import Hazard, Topology, simulate


@pytest.mark.parametrize("t", [Topology(8, 512), Topology(16, 1024)])
def test_point_owner_bijection_and_constant_four_private_slots(t):
    points = {t.point(tid, j) for tid in range(t.threads) for j in range(4)}
    assert len(points) == t.pad
    for p in points:
        tid, j = t.owner(p)
        assert t.point(tid, j) == p
    assert t.point(0, 1)[0] == 2


@pytest.mark.parametrize("version", [1, 2, 7, 8, 15, 16, 31, 32])
@pytest.mark.parametrize("t", [Topology(8, 512), Topology(16, 1024)])
def test_actual_field_version_offsets_at_wraps_ghosts_and_independent_warp_orders(t, version):
    proof = simulate(t, version=version, seed=37)
    assert proof["checks"] > 40_000 and proof["slots"] == 4
    assert proof["private_field_scalars"] == 24 and proof["barriers"] == 4
    assert proof["shared_bytes"] == (25828 if t.y == 8 else 51748)


@pytest.mark.parametrize("negative", ["publish-H-before-old-E-retirement", "omit-H-publication", "publish-E-before-H-reader-retirement"])
def test_missing_family_retirement_or_publication_has_actual_reader_failure(negative):
    with pytest.raises(Hazard):
        simulate(Topology(8, 512), version=2, negative=negative)


@pytest.mark.parametrize("shape", [(8, 1024), (16, 512), (4, 256)])
def test_proof_does_not_silently_generalize_a_different_mapping(shape):
    with pytest.raises(ValueError):
        Topology(*shape)
