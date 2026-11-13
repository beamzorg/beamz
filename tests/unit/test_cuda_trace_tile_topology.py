"""Subwarp routing must exactly cover the cell trace's cross-warp cut."""
import pytest
import re

from scripts.cuda_space_time_ir import IllegalSchedule
from scripts.cuda_trace_tile_topology import TileTopology,allocate_providers,verify_topology
from scripts.cuda_trace_tile_topology import prove_shared_e_effects
from scripts.cuda_phase_schedule_ir import ScheduleHazard
from scripts.build_cuda_trace_tile import replace_last_bounds
from scripts.generate_cuda_native_cell import generate_native_cell
from scripts.cuda_trace_tile_bindings import TileReadBinding
from scripts.cuda_stencil_trace import Read


@pytest.mark.parametrize('cache,groups',[
    ((8,16,32),1),((8,16,32),2),((16,16,16),2),((16,16,16),4),
    ((16,8,16),4),((8,8,8),4),
])
def test_provider_cover_is_exact_and_disjoint(cache,groups):
    proof=verify_topology(TileTopology(cache,groups))
    assert proof['disjoint_minimal_cover']
    assert proof['shared_words']==proof['unique_required_providers']


def test_two_y_rows_share_a_warp_without_cross_row_x_shuffles():
    mapping=TileTopology()
    assert mapping.route((0,0,15),(0,0,1))=='shuffle'
    assert mapping.source((0,0,15),(0,0,1))==(0,0,0)
    assert mapping.route((0,0,0),(0,1,0))=='shuffle'
    assert mapping.route((0,1,0),(0,1,0))=='shared'
    assert mapping.route((0,0,0),(0,-1,0))=='shared'


def test_z_frontiers_are_component_deduplicated():
    topology=TileTopology();regions,words=allocate_providers(topology)
    assert words==11264
    hx=[r for r in regions if r.field=='Hx']
    assert len(hx)==2 and hx[1].excluded_y_residue==1
    assert hx[0].index(topology,(3,1,0)) is not None
    assert hx[1].index(topology,(3,1,0)) is None
    assert hx[1].index(topology,(3,0,0)) is not None


@pytest.mark.parametrize('cache,groups',[
    ((16,16,12),2),((16,3,16),2),((16,16,16),3),((16,16,16),8),
])
def test_unsupported_topology_is_rejected(cache,groups):
    with pytest.raises(IllegalSchedule):TileTopology(cache,groups).validate()


def test_shared_e_sources_require_both_retirement_and_publication():
    assert prove_shared_e_effects()['result']=='pass'
    with pytest.raises(ScheduleHazard):prove_shared_e_effects(early_ghost=True)
    with pytest.raises(ScheduleHazard):prove_shared_e_effects(omit_publication=True)


def test_runtime_binding_handles_nested_launch_bound_expressions():
    text='__global__ __launch_bounds__((Fast?1024:512),1) '
    assert replace_last_bounds(text,'TraceThreads')=='__global__ __launch_bounds__(TraceThreads,1) '


def test_bound_point_ssa_does_not_change_the_generic_numerical_functor():
    ordinary=generate_native_cell()
    bound=generate_native_cell(read_binding=TileReadBinding())
    assert ordinary['source']==bound['source']
    assert set(bound['point_blocks'])=={'Hx','Hy','Hz','Ex','Ey','Ez'}
    assert all('read(' not in code for code in bound['point_blocks'].values())


def test_cell_binding_rejects_a_wrong_field_version():
    with pytest.raises(IllegalSchedule):TileReadBinding()('Hx',Read('Ey',1,(0,1,0)))
    with pytest.raises(IllegalSchedule):TileReadBinding()('Ex',Read('Hx',0,(0,-1,0)))


def test_shared_e_placement_changes_storage_without_generic_read_dispatch():
    text=TileReadBinding(shared_e=True)('Hx',Read('Ey',0,(0,1,0)))
    assert 'frame[' in text and not re.search(r'\be\[',text)
    assert 'Slot(zs+' in text
