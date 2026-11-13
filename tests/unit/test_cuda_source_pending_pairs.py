"""Paired plane lifetime failures and guarded Policy5 superset checks."""

import pytest

from scripts.prove_cuda_partial_ey_lifetimes import Hazard, Layout, simulate
from scripts.prove_cuda_source_pending_pairs import policy5_superset


@pytest.mark.parametrize("version", [1, 2, 7, 16, 31, 64])
def test_two_publication_pairs_preserve_all_traced_versions(version):
    r = simulate(
        Layout(8, 16, 1024),
        version=version,
        seed=37,
        shared_ey_private_pending=True,
        paired_planes=True,
    )
    assert r["barriers_per_step"] == 5 and r["shared_bytes"] == 58680
    assert (
        r["persistent_field_floats_per_thread"] + r["saved_current_ExEz_floats"] == 20
    )


@pytest.mark.parametrize(
    "negative",
    [
        "paired-early-restore",
        "paired-forget-first-pending",
        "paired-overwrite-second-old",
        "paired-missing-H3",
        "missing-snapshot-publication",
        "previous-version-snapshot",
    ],
)
def test_invalid_pair_lifetime_has_actual_reader_failure(negative):
    with pytest.raises(Hazard):
        simulate(
            Layout(8, 16, 1024),
            version=2,
            seed=37,
            shared_ey_private_pending=True,
            paired_planes=True,
            negative=negative,
        )


def test_only_four_plane_1024_chunks_are_admitted():
    with pytest.raises(ValueError):
        simulate(Layout(8, 16, 512), shared_ey_private_pending=True, paired_planes=True)


def test_policy5_superset_uses_actual_flags_and_existing_strict_interior_fact():
    for flags in range(256):
        if flags & 63 in (16, 17):
            r = policy5_superset(flags, (5, 14, 46), (26, 43, 91), (64, 96, 128))
            assert r["source_guard"] and r["observe_guard"] == bool(flags & 1)
        else:
            with pytest.raises(ValueError):
                policy5_superset(flags, (5, 14, 46), (26, 43, 91), (64, 96, 128))
    with pytest.raises(ValueError):
        policy5_superset(16, (0, 14, 46), (26, 43, 91), (64, 96, 128))
    with pytest.raises(ValueError):
        policy5_superset(17, (5, 14, 46), (63, 43, 91), (64, 96, 128))
