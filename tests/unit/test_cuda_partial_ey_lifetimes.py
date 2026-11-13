"""Versioned cache lifetime checks, including four Y groups and negatives."""

import pytest

from scripts.prove_cuda_partial_ey_lifetimes import (
    REVIEWED_BODY_SHA256,
    Hazard,
    Layout,
    bind_reviewed_body,
    collective_proof,
    direct_witnesses,
    prove_late_bootstrap,
    simulate,
    traced_reads,
)


def test_review_binding_records_exact_body_and_rejects_later_source_edits():
    from pathlib import Path

    source = Path("scripts/cuda_native_partial_ey_frame.cuh").read_bytes()
    assert bind_reviewed_body(source)["body_sha256"] == REVIEWED_BODY_SHA256
    with pytest.raises(ValueError, match="source changed"):
        bind_reviewed_body(source + b"\n// changed\n")


@pytest.mark.parametrize("layout", [Layout(16, 16, 1024), Layout(16, 16, 512), Layout(8, 32, 1024),
                                    Layout(8, 32, 512), Layout(8, 32, 256)])
def test_all_layouts_route_wrapped_versions_with_warp_run_ahead(layout):
    result = simulate(layout, version=15, seed=37)
    assert result["checked_field_reads"] > 100000
    assert result["barriers_per_step"] == (19 if layout == Layout(16, 16, 512) else 11)
    assert result["cross_Z_snapshots"] == (layout.zgroups == 2)


@pytest.mark.parametrize("name", ["missing-old-H-retirement", "early-Ey-ghost-import", "save-old-after-H-publish", "early-snapshot-overwrite",
                                  "wrong-snapshot-slot", "shared-previous-Z", "missing-plane-retirement", "missing-last-retirement"])
def test_deliberate_lifetime_breaks_are_detected(name):
    with pytest.raises(Hazard):
        simulate(Layout(16, 16, 1024), negative=name)


def test_actual_stencil_z_and_y_cross_warp_reader_matrix():
    reads = traced_reads()
    z_reads = {f: {(r.field, r.offset) for r in items if r.offset[0]} for f, items in reads.items()}
    assert z_reads[0] == {("Ey", (1, 0, 0))}
    assert z_reads[1] == {("Ex", (1, 0, 0))}
    assert not z_reads[2]
    assert z_reads[3] == {("Hy", (-1, 0, 0))}
    assert z_reads[4] == {("Hx", (-1, 0, 0))}
    assert not z_reads[5]
    for layout in [Layout(16, 16, 1024), Layout(8, 32, 512), Layout(8, 32, 256)]:
        source, _ = layout.owner((0, 0, 2))
        target, _ = layout.owner((0, 1, 2))
        assert source // 32 != target // 32


def test_snapshot_consumer_is_unique_and_pending_storage_is_thread_local_afterP0():
    layout = Layout(16, 16, 1024)
    keys = {}
    for tid in range(layout.threads):
        key = layout.scratch(tid)
        assert key not in keys
        keys[key] = tid
        z, y, x = layout.point(tid, 0)
        previous = ((z - 1) % 16, y, x)
        producer, si = layout.owner(previous)
        assert producer ^ 512 == tid and si == 7
        assert key == producer
    assert len(keys) == 1024


@pytest.mark.parametrize("layout", [Layout(16, 16, 1024), Layout(16, 16, 512), Layout(8, 32, 1024),
                                    Layout(8, 32, 512), Layout(8, 32, 256)])
def test_actual_partial_Ey_positiveZ_private_index_and_H_negativeZ_mapping(layout):
    for tid in range(layout.threads):
        for gy in range(layout.ygroups):
            for j in range(layout.local_z):
                z, y, x = layout.point(tid, j, gy)
                si = gy * layout.local_z + j
                q = ((z + 1) % layout.z, y, x)
                if q[0] % 8 >= 6:
                    owner, other_si = layout.owner(q)
                    assert owner == tid
                    assert ((si + 1) // 8) * 2 + (si + 1) % 2 == (other_si // 8) * 2 + other_si % 2
                previous, previous_si = layout.owner(((z - 1) % layout.z, y, x))
                if j:
                    assert previous == tid and previous_si == si - 1
                elif layout.zgroups == 1:
                    assert previous == tid and previous_si == si + layout.local_z - 1
                else:
                    assert previous == tid ^ layout.plane and previous_si == layout.local_z - 1


def test_warp_ghost_lanes_cannot_be_predicated_out_and_end_barriers_have_readers():
    assert len(collective_proof()["inside-owned-branch_negative_witnesses"]) == 32
    assert len(direct_witnesses()) == 4


@pytest.mark.parametrize("layout", [Layout(16, 16, 1024), Layout(8, 32, 1024), Layout(8, 32, 512), Layout(8, 32, 256)])
@pytest.mark.parametrize("origin", [(70, 40, 40), (40, 70, 40), (40, 40, 70), (70, 70, 70)])
def test_late_clipped_start_imports_exact_versions_and_zeroes_only_invalid_cells(layout, origin):
    result = prove_late_bootstrap(layout, origin=origin)
    assert result["first"] == 6 and result["checked"] > 10000
