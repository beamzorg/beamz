"""Actual Yee producer footprints and honest scalar-warp coalescing costs."""

import itertools
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.cuda_traced_warp_layout import (
    FIELDS, LAYOUTS, SHAPE, Layout, audit_lowering, canonical_reads, eval_peer, field_sectors, footprint,
    instruction_points, low_x_stage_sectors, prove_routes, publication_model, psi_sectors,
    required_axes, sector_summary, trail_sectors,
)


@pytest.mark.parametrize("shape", LAYOUTS)
def test_thread_lane_ownership_is_bijective_and_all_Z_neighbors_stay_private(shape):
    layout = Layout(*shape)
    points = {layout.point(tid, z) for tid in range(512) for z in range(8)}
    assert points == set(itertools.product(range(8), range(16), range(32)))
    for tid in range(512):
        assert all(layout.owner(layout.point(tid, z)) == tid for z in range(8))
    assert instruction_points(layout).shape == (128, 32, 3)
    assert len(set(map(tuple, instruction_points(layout).reshape(-1, 3)))) == 4096


@pytest.mark.parametrize("shape,expected", [((1, 32), 68136), ((2, 16), 38440),
                                          ((4, 8), 26152), ((8, 4), 26152), ((16, 2), 35368)])
def test_trace_derived_publisher_footprints_match_independent_inverse_consumers(shape, expected):
    layout = Layout(*shape)
    _, reads = canonical_reads()
    requirements = required_axes(reads, layout)
    for field in FIELDS:
        inverse = set()
        for source in itertools.product(*(range(s) for s in SHAPE)):
            for dst, values in reads.items():
                for r in values:
                    if r.field != field or not any(r.offset):
                        continue
                    consumer = tuple((a - b) % s for a, b, s in zip(source, r.offset, SHAPE))
                    if layout.owner(source) // 32 != layout.owner(consumer) // 32:
                        inverse.add(source)
        assert footprint(field, layout, requirements) == inverse
    assert publication_model(layout)["shared_total_bytes"] == expected


def test_four_by_eight_faces_and_intersections_are_not_double_counted():
    model = publication_model(Layout(4, 8))
    values = {x["field"]: x for x in model["fields"]}
    assert values["Ex"]["allocated_words"] == values["Hx"]["allocated_words"] == 1024
    assert values["Ey"]["allocated_words"] == values["Hy"]["allocated_words"] == 512
    assert values["Ez"]["allocated_words"] == values["Hz"]["allocated_words"] == 1408
    assert values["Ez"]["intersection_words"] == 128
    assert model["shared_field_words"] == 5888
    assert model["private_field_words_per_thread"] == 48


@pytest.mark.parametrize("mutation", ["opposite-face", "omit-union-intersection", "wrong-forward-version"])
def test_unsafe_publisher_or_time_version_changes_have_actual_read_counterexamples(mutation):
    with pytest.raises(AssertionError):
        prove_routes(Layout(4, 8), depth=2, negative=mutation)


def test_actual_two_step_ring_and_halfstep_versions_have_exact_source_coordinates():
    proof = prove_routes(Layout(4, 8), depth=2)
    assert proof["actual_read_checks"] == 195300
    assert all(n > 0 for n in proof["route_counts"].values())


def test_sector_sets_respect_alignment_and_broadcasts():
    aligned = sector_summary(np.arange(32).reshape(1, 32))
    unaligned = sector_summary((1 + np.arange(32)).reshape(1, 32))
    assert aligned["issued_sector_lower_bound"] == 4
    assert unaligned["issued_sector_lower_bound"] == 5
    repeated = sector_summary(np.zeros((1, 32), np.int64))
    assert repeated["active_lane_accesses"] == 32
    assert repeated["distinct_warp_word_accesses"] == repeated["issued_sector_lower_bound"] == 1
    # Four8-float rows are not four sectors when their stride is N+1.
    rows = np.arange(4)[:, None] * 513 + np.arange(8)[None, :]
    assert sector_summary(rows.reshape(1, 32))["issued_sector_lower_bound"] == 7


@pytest.mark.parametrize("shape", LAYOUTS)
def test_trail_layout_changes_issue_sectors_not_physical_published_or_imported_values(shape):
    layout = Layout(*shape)
    for version, field in ((1, 0), (4, 3), (16, 5)):
        for imports, count in ((False, 735), (True, 736)):
            old = trail_sectors(None, version, field, imports=imports)
            new = trail_sectors(layout, version, field, imports=imports)
            assert old["active_lane_accesses"] == new["active_lane_accesses"] == count
            assert old["CTA_unique_sector_addresses"] == new["CTA_unique_sector_addresses"]
        assert low_x_stage_sectors(version, field)["active_lane_accesses"] == 105


@pytest.mark.parametrize("n", [512, 564])
@pytest.mark.parametrize("packed", [False, True])
def test_actual_staggered_dense_and_packed_pitches_preserve_global_address_sets(n, packed):
    shape = (n + 1, n + 1, n)
    origin = (56, 75, 155)
    old = field_sectors(None, shape, origin, version=16, owned=True, packed_codes=packed)
    for dims in LAYOUTS:
        new = field_sectors(Layout(*dims), shape, origin, version=16, owned=True, packed_codes=packed)
        assert new["active_lane_accesses"] == old["active_lane_accesses"] == 3255
        assert new["CTA_unique_sector_addresses"] == old["CTA_unique_sector_addresses"]


@pytest.mark.parametrize("shape", [(3, 8), (32, 1), (4, 16), (0, 32)])
def test_unsupported_warp_shape_fails_closed(shape):
    with pytest.raises(ValueError, match="complete32-lane"):
        Layout(*shape)


@pytest.mark.parametrize("expression", ["lane.__class__", "sum([lane])", "lane*2", "(lambda: 0)()"])
def test_shuffle_peer_parser_rejects_noninteger_lowering_grammar(expression):
    with pytest.raises(ValueError, match="expression grammar"):
        eval_peer(expression, 0)


def test_actual_cpp_peers_and_generated_packed_header_match_independent_read_ownership(tmp_path):
    source = Path(".cache/cuda-five-hour-2026-11-13-1059/warp-frontier-512-v1/source")
    if not source.exists() or not shutil.which("c++"):
        pytest.skip("Frozen copied CUDA body and host C++ compiler required")
    proof = audit_lowering(source, tmp_path / "actual")
    assert proof["shuffle_peer_cases"] == 256
    assert proof["header_index_cases"] == 24576
    assert proof["shared_producer_words_exactly_once"] == 5888
    assert proof["shared_total_bytes"] == 26152
    bad = tmp_path / "bad-source" / "src"
    bad.mkdir(parents=True)
    body = (source / "src/asymmetric_native.cuh").read_text()
    body = body.replace("(lane&24)|((lane+1)&7)", "(lane&24)|((lane+2)&7)")
    (bad / "asymmetric_native.cuh").write_text(body)
    shutil.copy2(source / "src/warp_frontier_layout_generated.cuh", bad)
    with pytest.raises(ValueError, match=r"Actual C\+\+ shuffle peer differs"):
        audit_lowering(bad.parent, tmp_path / "bad-proof")


@pytest.mark.parametrize("domain", [512, 564])
@pytest.mark.parametrize("family", [0, 1])
def test_cpml12_packed_slab_accesses_match_independent_interval_counts_and_all_layout_address_sets(domain, family):
    from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
    owned = (7, 15, 31)

    def intersect(a, b, c, d):
        return max(0, min(b, d) - max(a, c) + 1)

    for term, version in itertools.product(range(6), (1, 16)):
        axis = (1, 0, 0, 2, 2, 1)[term]
        extent = domain + family
        target = tuple(domain + d for d in SUPPORT_DELTA[family * 3 + term // 2])
        for b in (0, 1, 3, (domain + owned[axis] - 1) // owned[axis] - 1):
            origin = [56, 60, 124]
            origin[axis] = b * owned[axis]
            low = [o - version for o in origin]
            high = [o - version + n - 1 for o, n in zip(origin, owned)]
            count = 1
            for a in range(3):
                if a == axis:
                    count *= intersect(low[a], high[a], 0, 11) + intersect(low[a], high[a], extent - 12, extent - 1)
                else:
                    count *= intersect(low[a], high[a], 0, target[a] - 1)
            old = psi_sectors(None, domain, family, term, origin, version)
            assert old["active_lane_accesses"] == count
            for wy, wx in LAYOUTS:
                new = psi_sectors(Layout(wy, wx), domain, family, term, origin, version)
                assert new["active_lane_accesses"] == count
                assert new["CTA_unique_sector_addresses"] == old["CTA_unique_sector_addresses"]
