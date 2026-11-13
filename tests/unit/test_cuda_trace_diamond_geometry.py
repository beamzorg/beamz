"""Retimed diamond coordinates must preserve every physical cell and version."""
import pytest

from scripts.cuda_trace_diamond_geometry import DiamondGeometry,verify_geometry,verify_common_cell_gauge
from scripts.cuda_space_time_ir import IllegalSchedule


@pytest.mark.parametrize('orientation,cache',[('xy',(4,8,4)),('yz',(4,4,8))])
@pytest.mark.parametrize('bias',[24,25])
def test_packed_geometry_covers_both_owner_and_physical_domain_parities(bias,orientation,cache):
    proof=verify_geometry(DiamondGeometry(cache,orientation),versions=(1,2,3,4,5,7,8,9),
                          origins=((0,0,0),(3,7,0)),bias_y=bias)
    assert proof['result']=='pass' and proof['checked_versioned_reads']>10000


@pytest.mark.parametrize('orientation,cache',[('xy',(4,8,4)),('yz',(4,4,8))])
def test_dropping_component_retiming_breaks_the_physical_lattice(orientation,cache):
    with pytest.raises(IllegalSchedule):
        verify_geometry(DiamondGeometry(cache,orientation),versions=(1,),origins=((0,0,0),),drop_retiming=True)


def test_full_shared_storage_fits_without_cached_cpml_profiles():
    geometry=DiamondGeometry()
    assert geometry.field_points==4096
    assert 24*geometry.field_points+76==98380
    assert 98380<101376


def test_physical_cell_storage_gauge_preserves_versioned_neighbors_and_owned_coordinates():
    geometry=DiamondGeometry((4,4,8),'yz','physical')
    assert verify_geometry(geometry,versions=(1,2,3,4,5,7,8,9),origins=((0,0,0),(3,7,0)),bias_y=25)['result']=='pass'
    assert verify_common_cell_gauge(geometry)['checked_owned_components']>3000
    with pytest.raises(IllegalSchedule):
        verify_common_cell_gauge(geometry,drop_field_ownership=True)


def test_cpml_free_owned_bounds_admit_all_cached_cells_including_initial_ghosts():
    import itertools
    from scripts.cuda_trace_diamond_regions import physical_bounds
    geometry=DiamondGeometry((4,4,8),'yz','physical');depth=5;origin=(90,135,62);shape=(136,137,139)
    low,high=physical_bounds(origin,tuple(c-1 for c in geometry.cache),depth,shape[1],orientation='yz')
    assert all(lo>=12 and hi<size-12 for lo,hi,size in zip(low,high,shape))
    for n,point,field in itertools.product(range(depth+1),itertools.product(*(range(c) for c in geometry.packed_shape)),('Hx','Hy','Hz','Ex','Ey','Ez')):
        p=geometry.physical(field,point,n,origin,shape[1])
        assert all(lo-1<=value<=hi+1 and 0<=value<size for lo,value,hi,size in zip(low,p,high,shape))
    # Merely admitting owned interior points does not justify removing ghost
    # support checks: with no margin the incoming X plane is out of support.
    edge=(90,135,depth)
    assert physical_bounds(edge,tuple(c-1 for c in geometry.cache),depth,shape[1],orientation='yz')[0][2]==0
    assert any(geometry.physical('Hx',point,depth,edge,shape[1])[2]<0 for point in itertools.product(*(range(c) for c in geometry.packed_shape)))
