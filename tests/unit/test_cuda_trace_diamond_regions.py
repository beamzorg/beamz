"""Conservative material bounds and exact nonempty-owner certificates."""
import itertools
import pytest

from scripts.cuda_trace_diamond_geometry import DiamondGeometry
from scripts.cuda_trace_diamond_regions import physical_bounds,has_update


@pytest.mark.parametrize('orientation,cache',[('xy',(4,8,4)),('yz',(4,4,8))])
@pytest.mark.parametrize('bias',[7,8])
@pytest.mark.parametrize('r',[0,1])
def test_regions_against_independent_packed_cell_enumeration(bias,r,orientation,cache):
    geometry=DiamondGeometry(cache,orientation);owned=tuple(c-1 for c in cache);depth=5
    field='Hy' if r else 'Hx';shape=(6,8,9)
    points=list(itertools.product(*(range(c) for c in geometry.packed_shape)))
    for tile in itertools.product(range(3),range(4),range(5)):
        origin=tuple(q*b for q,b in zip(tile,owned));low,high=physical_bounds(origin,owned,depth,bias,r,orientation=orientation)
        hit=False
        for n in range(1,depth+1):
            for point in points:
                if not geometry.owns(field,point,n,origin,bias):continue
                physical=geometry.physical(field,point,n,origin,bias)
                assert all(a<=p<=b for a,p,b in zip(low,physical,high))
                hit|=all(0<=p<size for p,size in zip(physical,shape))
        assert has_update(origin,owned,depth,bias,shape,r,orientation=orientation)==hit
