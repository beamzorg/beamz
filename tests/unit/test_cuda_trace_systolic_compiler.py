"""Dependency, storage and warp-participation contracts for column scheduling."""
import dataclasses

import pytest

from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_trace_systolic_compiler import (
    ThreadMap, compile_column, derive_routes, prove_routes, prove_compact_providers,
    compact_publication,
    prove_z_providers,
)


def test_full_cpml_trace_derives_the_two_y_components():
    plan = compile_column()
    assert plan['shared_fields'] == ('Hx', 'Hz', 'Ex', 'Ez')
    assert plan['barriers_per_step'] == 2
    assert plan['private_field_floats_per_thread'] == 48
    assert len([e for e in plan['effects'] if e['role'] == 'local-state']) == 12
    assert {e['role'] for e in plan['effects']} >= {'source', 'observer', 'ordered-reduction'}


def test_aliased_storage_requires_reader_retirement():
    assert compile_column(separate_families=False)['barriers_per_step'] == 4
    with pytest.raises(IllegalSchedule, match='retirement'):
        compile_column(separate_families=False, requested_barriers=2)


@pytest.mark.parametrize('steps', [4, 16, 35])
def test_rotating_route_uses_the_correct_field_version(steps):
    proof = prove_routes(steps=steps)
    assert proof['checked_routed_neighbours'] > 100000
    assert proof['result'] == 'pass'


@pytest.mark.parametrize('mapping', [ThreadMap(threads=768), ThreadMap(private_axis=1),
                                    ThreadMap(shuffle_axis=0), ThreadMap(cache=(12, 16, 32)),
                                    ThreadMap(cache=(8, 16, 31))])
def test_unsupported_maps_fail_before_cuda_generation(mapping):
    with pytest.raises(IllegalSchedule):
        derive_routes(mapping=mapping)


def test_y_neighbours_never_use_the_normal_y_component():
    shared = [r for r in derive_routes() if r.route == 'shared-y-neighbour']
    assert shared
    assert all(r.source[-1] in 'xz' and r.offset[1] in (-1, 1) for r in shared)


@pytest.mark.parametrize('axes', [(), (0,), (1,), (2,), (0, 1, 2)])
def test_cpml_specialization_preserves_field_routes(axes):
    routes = derive_routes(yee_operator(cpml_axes=axes))
    actual = {r for r in routes if not r.output.startswith('psi')}
    expected = {r for r in derive_routes() if not r.output.startswith('psi')
                and not r.source.startswith('psi')}
    assert {r for r in actual if not r.source.startswith('psi')} == expected


def test_phase_declarations_cannot_be_relabelled():
    operator = yee_operator()
    tensors = tuple(dataclasses.replace(t, phase=1-t.phase) for t in operator.tensors)
    with pytest.raises(IllegalSchedule):
        compile_column(dataclasses.replace(operator, tensors=tensors))


@pytest.mark.parametrize('steps', [4, 16, 35])
def test_minimal_shared_provider_boxes_cover_the_traced_reads(steps):
    proof = prove_compact_providers(steps=steps)
    assert proof['words_per_component'] == 7*15*31
    assert proof['checked_shared_reads'] > 50000


def test_provider_geometry_differs_between_phases():
    # At version0 the low ghost is slot(7,15,31). H's Y halo is needed,
    # while E's incoming Z/X halos survive into the next H phase instead.
    assert compact_publication((0, 15, 0), 0, 'H') is not None
    assert compact_publication((0, 15, 0), 0, 'E') is None
    assert compact_publication((7, 0, 31), 0, 'E') is not None
    assert compact_publication((7, 0, 31), 0, 'H') is None


def test_unknown_family_is_rejected():
    with pytest.raises(IllegalSchedule):
        compact_publication((0, 0, 0), 0, 'psi')


@pytest.mark.parametrize('steps', [4, 16, 35])
def test_two_z_thread_groups_exchange_only_their_private_chunk_edges(steps):
    mapping = ThreadMap(threads=1024, z_groups=2)
    plan = compile_column(mapping=mapping)
    assert plan['private_field_floats_per_thread'] == 24
    assert plan['barriers_per_step'] == 2
    proof = prove_z_providers(mapping, steps=steps)
    assert proof['checked_z_edge_reads'] > 0
    assert proof['shared_z_words_per_component'] == 2*15*31
    assert prove_routes(mapping, steps=steps)['result'] == 'pass'
