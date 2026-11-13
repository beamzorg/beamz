"""Independent direct-snapshot, ghost/version and private pending tests."""

import pytest

from scripts.prove_cuda_partial_ey_lifetimes import (
    Hazard,
    Layout,
    prove_late_bootstrap,
    simulate,
    traced_reads,
)


@pytest.mark.parametrize("threads", [1024, 512])
@pytest.mark.parametrize("version", [1, 2, 7, 16, 31, 64])
def test_full_shared_Ey_and_private_pending_route_exact_versions(threads, version):
    r = simulate(
        Layout(8, 16, threads), version=version, seed=37, shared_ey_private_pending=True
    )
    assert r["shared_bytes"] == 58680
    assert r["barriers_per_step"] == (7 if threads == 1024 else 11)
    assert r["persistent_field_floats_per_thread"] + r["saved_current_ExEz_floats"] == (
        16 if threads == 1024 else 28
    )


@pytest.mark.parametrize(
    "negative",
    [
        "missing-snapshot-publication",
        "previous-version-snapshot",
        "wrong-snapshot-slot",
        "missing-old-H-retirement",
        "early-Ey-ghost-import",
        "missing-plane-retirement",
        "missing-last-retirement",
    ],
)
def test_actual_reader_witnesses_reject_stale_snapshot_or_early_storage_reuse(negative):
    with pytest.raises(Hazard):
        simulate(
            Layout(8, 16, 1024),
            version=2,
            seed=37,
            shared_ey_private_pending=True,
            negative=negative,
        )


def test_snapshot_has_unique_opposite_chunk_same_XY_consumer_and_only_two_components():
    layout = Layout(8, 16, 1024)
    keys = set()
    for tid in range(1024):
        z, y, x = layout.point(tid, 0)
        producer, slot = layout.owner(((z - 1) % 8, y, x))
        assert producer == tid ^ 512 and slot == 3
        assert layout.scratch(tid) == producer
        keys.add(producer)
    assert len(keys) == 1024
    reads = traced_reads()
    assert {r.field for f in (3, 4, 5) for r in reads[f] if r.offset[0]} == {"Hx", "Hy"}


@pytest.mark.parametrize("threads", [1024, 512])
@pytest.mark.parametrize(
    "origin", [(70, 40, 40), (40, 70, 40), (40, 40, 70), (70, 70, 70)]
)
def test_late_bootstrap_and_reset_private_pending_never_need_uninitialized_old_values(
    threads, origin
):
    assert prove_late_bootstrap(Layout(8, 16, threads), origin=origin)["first"] == 6
    first = simulate(
        Layout(8, 16, threads), version=6, seed=13, shared_ey_private_pending=True
    )
    repeated = simulate(
        Layout(8, 16, threads), version=6, seed=13, shared_ey_private_pending=True
    )
    assert first == repeated
