"""Field-retimed diamond cache geometry derived from the cell dependence graph.

This models a different global ownership basis, not a balanced-box tuning knob.
The determinant-two lattice is packed without fictitious cells. GPU ownership,
coalesced I/O, CPML classification and publication lowering remain separate work.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import itertools
import math

from scripts.cuda_trace_affine_ownership import DIAMOND_XY_BASIS,synthesize
from scripts.cuda_space_time_ir import IllegalSchedule,yee_operator
from scripts.cuda_space_time_ir import FIELDS


@dataclass(frozen=True)
class DiamondGeometry:
    cache: tuple[int,int,int]=(16,32,16)
    orientation: str='xy'
    storage_gauge: str='retimed'

    @property
    def axes(self):
        # (identity logical axis, plus logical axis, minus logical axis,
        # identity physical axis, rotated physical axis). Physical Y is shared.
        if self.orientation=='xy':return (0,1,2,0,2)
        if self.orientation=='yz':return (2,0,1,2,0)
        raise IllegalSchedule('Supported diamond orientations are xy and yz')

    @property
    def packed_shape(self):
        shape=list(self.cache);shape[self.axes[2]]//=2
        return tuple(shape)

    @property
    def basis(self):
        return DIAMOND_XY_BASIS if self.orientation=='xy' else ((1,1,0),(1,-1,0),(0,0,1))


    def validate(self):
        if self.storage_gauge not in ('retimed','physical'):raise IllegalSchedule('Unknown field storage gauge')
        if self.storage_gauge=='physical' and self.orientation!='yz':raise IllegalSchedule('Physical-cell gauge currently reviewed for contiguous-X YZ diamonds')
        if len(self.cache)!=3 or any(type(c) is not int or c<2 or c&(c-1) for c in self.cache):
            raise IllegalSchedule('Even power-of-two transformed cache extents required')

    @cached_property
    def plan(self):
        return synthesize(self.basis)

    @property
    def field_points(self):return math.prod(self.cache)//2

    def retiming(self,field):
        value=self.plan['tensor_retiming'][field]
        minus=self.axes[2]
        if any(v for a,v in enumerate(value) if a!=minus) or value[minus] not in (0,1):
            raise IllegalSchedule('Reviewed two-class retiming required')
        return value[minus]

    def raw(self,field,point,origin=(0,0,0),bias_y=0):
        identity,plus,minus,_,_=self.axes
        raw=list(point)
        if self.storage_gauge=='physical':
            parity=(origin[plus]+origin[minus]-bias_y+point[plus])&1
            raw[minus]=(2*point[minus]+parity+self.retiming(field))%self.cache[minus]
        else:
            parity=(origin[plus]+origin[minus]-bias_y+point[plus]-self.retiming(field))&1
            raw[minus]=2*point[minus]+parity
        return tuple(raw)

    def physical(self,field,point,version,origin=(0,0,0),bias_y=0):
        raw=self.raw(field,point,origin,bias_y);low=-version-1
        values=tuple(o+low+(s-low)%c for o,s,c in zip(origin,raw,self.cache))
        identity,plus,minus,physical_identity,physical_rotated=self.axes
        p,m=values[plus],values[minus]-self.retiming(field)-bias_y
        if (p+m)&1:raise IllegalSchedule('Fictitious half-integer Yee cell')
        result=[0,0,0];result[physical_identity]=values[identity]
        result[1]=(p-m)//2;result[physical_rotated]=(p+m)//2
        return tuple(result)

    def owns(self,field,point,version,origin=(0,0,0),bias_y=0):
        raw=self.raw(field,point,origin,bias_y);head=tuple((-version-1)%c for c in self.cache)
        return all(s!=h for s,h in zip(raw,head))

    def read_point(self,output,read,point,origin=(0,0,0),bias_y=0,*,drop_retiming=False):
        raw=self.raw(output,point,origin,bias_y)
        delta=0 if drop_retiming else self.retiming(read.field)-self.retiming(output)
        shift=[sum(a*b for a,b in zip(row,read.offset)) for row in self.basis]
        minus=self.axes[2];shift[minus]+=delta
        target=tuple((q+d)%c for q,d,c in zip(raw,shift,self.cache))
        packed=list(target)
        if self.storage_gauge=='physical':packed[minus]=(packed[minus]-self.retiming(read.field))%self.cache[minus]
        packed[minus]//=2;packed=tuple(packed)
        if self.raw(read.field,packed,origin,bias_y)!=target:
            raise IllegalSchedule('Read mapped to the wrong field lattice parity')
        return packed


def verify_geometry(geometry=DiamondGeometry(),*,versions=(1,2,15,16,17,31,32,33,35),origins=((0,0,0),(15,31,0)),bias_y=136,drop_retiming=False):
    geometry.validate();operator=yee_operator(cpml_axes=(0,1,2));checks=0;owned=0
    points=list(itertools.product(*(range(c) for c in geometry.packed_shape)))
    # Cache the synthesized graph proof once; no domain-sized Python IR is built.
    for origin in origins:
        for version in versions:
            for update in operator.updates:
                field=update.output.field;seen=set()
                for point in points:
                    physical=geometry.physical(field,point,version,origin,bias_y)
                    if physical in seen:raise IllegalSchedule('Duplicate packed physical cell')
                    seen.add(physical)
                    if not geometry.owns(field,point,version,origin,bias_y):continue
                    owned+=1
                    for read in update.reads:
                        source=geometry.read_point(field,read,point,origin,bias_y,drop_retiming=drop_retiming)
                        value=geometry.physical(read.field,source,version-1+read.version,origin,bias_y)
                        expected=tuple(p+d for p,d in zip(physical,read.offset))
                        if value!=expected:raise IllegalSchedule('Diamond read has the wrong physical version/location')
                        checks+=1
    return dict(schema='beamz.retimed-diamond-cache.v1',cache=geometry.cache,
                orientation=geometry.orientation,storage_gauge=geometry.storage_gauge,field_points=geometry.field_points,full_six_field_shared_bytes=24*geometry.field_points,
                versions=versions,origins=origins,bias_y=bias_y,owned_tensor_updates=owned,
                checked_versioned_reads=checks,retiming=geometry.plan['tensor_retiming'],
                result='pass',GPU_executed=False,
                scope='Packed lattice and versioned neighbor addresses; no native host plan, CUDA visibility, coalesced I/O or performance claim')


def verify_common_cell_gauge(geometry,*,versions=range(1,10),bias_y=25,drop_field_ownership=False):
    """Owned components share a physical cell; ghost components may not."""
    if geometry.orientation!='yz' or geometry.storage_gauge!='physical':
        raise IllegalSchedule('Common-cell proof requires physical storage gauge')
    checks=0
    for parity,version,point,field in itertools.product((0,1),versions,
            itertools.product(*(range(c) for c in geometry.packed_shape)),FIELDS):
        origin=(0,parity,0);low=-version-1
        owned_field='Hx' if drop_field_ownership else field
        if not geometry.owns(owned_field,point,version,origin,bias_y):continue
        plus=point[0];minus=2*point[1]+((origin[0]+origin[1]-bias_y+plus)&1);identity=point[2]
        a,b,c=(o+low+(q-low)%width for o,q,width in zip(origin,(plus,minus,identity),geometry.cache))
        expected=((a+b-bias_y)//2,(a-b+bias_y)//2,c)
        if geometry.physical(field,point,version,origin,bias_y)!=expected:
            raise IllegalSchedule('Retimed field does not share the common owned physical cell')
        checks+=1
    return dict(passed=True,checked_owned_components=checks,scope='Independent physical-cell inverse with field-specific ownership; ghost co-location not admitted')


class DiamondReadBinding:
    """Typed read to a packed shared-field address; no runtime field dispatch."""
    def __init__(self,geometry=DiamondGeometry()):self.geometry=geometry

    def __call__(self,output,read):
        phase=int(output.startswith('E'));source_phase=int(read.field.startswith('E'))
        if read.version!=(0 if source_phase else phase):raise IllegalSchedule('Wrong diamond field version')
        field=FIELDS.index(read.field)
        shift=[sum(a*b for a,b in zip(row,read.offset)) for row in self.geometry.basis]
        if self.geometry.storage_gauge=='retimed':shift[self.geometry.axes[2]]+=self.geometry.retiming(read.field)-self.geometry.retiming(output)
        if self.geometry.orientation=='xy':
            dz,dp,dm=shift
            return (f'frame[{field}*DiamondPoints+((((qz+({dz}))&(CZ-1))*CY+'
                    f'((qp+({dp}))&(CY-1)))*DiamondHalfX+(((m+({dm}))&(CX-1))>>1))]')
        a,b,c=shift
        return (f'frame[{field}*DiamondPoints+(((((q0+({a}))&(CZ-1))*DiamondHalfY+'
                f'(((q1+({b}))&(CY-1))>>1))*CX)+((q2+({c}))&(CX-1)))]')
