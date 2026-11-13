"""Source-plane-oriented warp ownership with independent coalesced field I/O.

World X is the source normal. A warp spans two Z rows and16 Y cells, with
private X chunks per lane. Source additions then use full half-warps rather
than one or two X lanes. The Yee trace determines cross-warp providers.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import itertools
import math

from scripts.cuda_space_time_ir import FIELDS,IllegalSchedule,yee_operator


@dataclass(frozen=True)
class SourceWarpTopology:
    cache: tuple[int,int,int]=(8,16,32)
    x_slots: int=8

    def validate(self):
        if self.cache!=(8,16,32) or self.x_slots not in (4,8,16):
            raise IllegalSchedule('Reviewed source cache and power-of-two X chunk required')

    @property
    def slots(self):return self.x_slots
    @property
    def threads(self):return math.prod(self.cache)//self.slots
    @property
    def points(self):return math.prod(self.cache)

    def point(self,thread,slot):
        warp,lane=divmod(thread,32);groups=self.cache[2]//self.slots
        return (warp//groups)*2+lane//16,lane%16,(warp%groups)*self.slots+slot

    def owner(self,point):
        z,y,x=point;groups=self.cache[2]//self.slots
        return ((z//2)*groups+x//self.slots)*32+(z%2)*16+y,x%self.slots

    def source(self,read,point):
        return tuple((p+d)%c for p,d,c in zip(point,read.offset,self.cache))

    @cached_property
    def providers(self):
        self.validate();points={f:set() for f in FIELDS};operator=yee_operator(cpml_axes=(0,1,2))
        # Y lanes have identical inter-warp routes; independent proof below
        # expands every lane and exact source point.
        for update,z,x in itertools.product(operator.updates,range(self.cache[0]),range(self.cache[2])):
            if update.output.field not in FIELDS:continue
            point=(z,0,x);thread,_=self.owner(point)
            for read in update.reads:
                if read.field not in FIELDS:continue
                source=self.source(read,point);source_thread,_=self.owner(source)
                if source_thread//32!=thread//32:points[read.field].add((source[0],source[2]))
        slots={};offsets={};words=0
        for field in FIELDS:
            slots[field]=tuple(sorted({(z%2)*self.slots+x%self.slots for z,x in points[field]}))
            expected={(z,x) for z in range(self.cache[0]) for x in range(self.cache[2])
                      if (z%2)*self.slots+x%self.slots in slots[field]}
            if points[field]!=expected:raise IllegalSchedule('Nonperiodic source provider cover')
            offsets[field]=words;words+=16*len(points[field])
        return dict(slots=slots,field_offsets=offsets,words=words,points=points)

    def provider_index(self,field,point):
        thread,slot=self.owner(point);combined=(thread%32//16)*self.slots+slot
        slots=self.providers['slots'][field]
        if combined not in slots:return -1
        return self.providers['field_offsets'][field]+((thread//32)*len(slots)+slots.index(combined))*16+thread%16

    def route(self,read,slot,z_lane):
        point=self.point(z_lane*16,slot);thread,_=self.owner(point)
        source=self.source(read,point);other,source_slot=self.owner(source)
        route='shared' if other//32!=thread//32 else 'shuffle' if other%32!=thread%32 else 'register'
        return route,source_slot

    def prove(self):
        self.validate();checks=0;counts=dict(register=0,shuffle=0,shared=0)
        operator=yee_operator(cpml_axes=(0,1,2))
        for update,thread,slot in itertools.product(operator.updates,range(self.threads),range(self.slots)):
            if update.output.field not in FIELDS:continue
            point=self.point(thread,slot)
            if self.owner(point)!=(thread,slot):raise IllegalSchedule('Source ownership alias')
            for read in update.reads:
                if read.field not in FIELDS:continue
                source=self.source(read,point);other,other_slot=self.owner(source)
                route,source_slot=self.route(read,slot,thread%32//16)
                expected='shared' if other//32!=thread//32 else 'shuffle' if other%32!=thread%32 else 'register'
                if route!=expected or source_slot!=other_slot:raise IllegalSchedule('Incorrect source register/shuffle route')
                if route=='shared' and self.provider_index(read.field,source)<0:raise IllegalSchedule('Missing source provider')
                counts[route]+=1;checks+=1
        indices={self.provider_index(f,(z,y,x)) for f in FIELDS for z,x in self.providers['points'][f] for y in range(16)}
        if indices!=set(range(self.providers['words'])):raise IllegalSchedule('Source provider storage aliases or holes')
        return dict(schema='beamz.source-warp-topology.v1',cache=self.cache,threads=self.threads,slots=self.slots,
                    provider_words=self.providers['words'],provider_bytes=4*self.providers['words'],
                    startup_io_scratch_words=3*self.points,
                    shared_frame_bytes=4*max(3*self.points,self.providers['words'])+76,
                    private_field_registers=6*self.slots,checked_reads=checks,routes=counts,passed=True,
                    source_normal='physical X',warp_axes='16Y by2Z; private X chunk',
                    scope='All physical field neighbor routes and injective shared provider cover; independent emitted and CUDA validation still required')
