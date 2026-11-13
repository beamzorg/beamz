"""Derive warp ownership and shared providers from retimed Yee field reads.

Physical X stays in a warp. Each warp owns a small rectangle in the packed
YZ diamond lattice; the other two coordinates remain in per-thread registers.
The provider cover is checked against both field lattice parities.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import itertools

from scripts.cuda_trace_diamond_geometry import DiamondGeometry
from scripts.cuda_space_time_ir import FIELDS,IllegalSchedule,yee_operator


@dataclass(frozen=True)
class DiamondWarpTopology:
    plus: int=4
    minus: int=2
    geometry: DiamondGeometry=DiamondGeometry((16,16,32),'yz')
    parity_providers: bool=False
    shared_field: str|None=None

    def validate(self):
        self.geometry.validate()
        if self.shared_field is not None and (self.shared_field not in FIELDS or not self.parity_providers or self.geometry.storage_gauge!='physical'):
            raise IllegalSchedule('Dense shared field requires physical gauge and parity providers')
        z,m,x=self.geometry.packed_shape
        if self.geometry.orientation!='yz' or x!=32 or z%self.plus or m%self.minus:
            raise IllegalSchedule('Warp rectangles must divide the contiguous-X diamond lattice')
        if self.threads>1024 or self.threads<32:
            raise IllegalSchedule('Invalid CUDA block size')

    @property
    def slots(self):return self.plus*self.minus
    @property
    def threads(self):return self.geometry.field_points//self.slots
    @property
    def warps(self):return self.threads//32

    def owner(self,point):
        p,m,x=point;minus_blocks=self.geometry.packed_shape[1]//self.minus
        return ((p//self.plus)*minus_blocks+m//self.minus)*32+x,(p%self.plus)*self.minus+m%self.minus

    def point(self,thread,slot):
        warp,lane=divmod(thread,32);minus_blocks=self.geometry.packed_shape[1]//self.minus
        return (warp//minus_blocks)*self.plus+slot//self.minus,(warp%minus_blocks)*self.minus+slot%self.minus,lane

    @cached_property
    def provider_layouts(self):
        self.validate();operator=yee_operator(cpml_axes=(0,1,2));layouts=[]
        for parity in (0,1):
            rows={f:set() for f in FIELDS}
            for update,p,m in itertools.product(operator.updates,
                    range(self.geometry.packed_shape[0]),range(self.geometry.packed_shape[1])):
                if update.output.field not in FIELDS:continue
                point=(p,m,16);outthread,_=self.owner(point)
                for read in update.reads:
                    if read.field not in FIELDS:continue
                    source=self.geometry.read_point(update.output.field,read,point,(0,parity,0),136)
                    thread,_=self.owner(source)
                    if thread//32!=outthread//32:rows[read.field].add(source[:2])
            if self.shared_field is not None:rows[self.shared_field]=set()
            layouts.append(self.allocate_rows(rows))
        return tuple(layouts)

    def allocate_rows(self,rows):
        slots={};offsets={};words=0
        for field in FIELDS:
            slots[field]=tuple(sorted({(p%self.plus)*self.minus+m%self.minus for p,m in rows[field]}))
            expected={(p,m) for p in range(self.geometry.packed_shape[0])
                      for m in range(self.geometry.packed_shape[1])
                      if (p%self.plus)*self.minus+m%self.minus in slots[field]}
            if rows[field]!=expected:raise IllegalSchedule('Provider cover is not periodic in the warp rectangle')
            offsets[field]=words;words+=32*len(rows[field])
        return dict(slots=slots,rows={f:tuple(sorted(value)) for f,value in rows.items()},
                    field_offsets=offsets,words=words)

    @cached_property
    def providers(self):
        return self.allocate_rows({f:set(self.provider_layouts[0]['rows'][f])|set(self.provider_layouts[1]['rows'][f]) for f in FIELDS})

    @property
    def allocated_words(self):
        words=max(p['words'] for p in self.provider_layouts) if self.parity_providers else self.providers['words']
        return words+(self.geometry.field_points if self.shared_field is not None else 0)

    @property
    def shared_field_offset(self):return max(p['words'] for p in self.provider_layouts)

    def provider_index(self,field,point,parity=None):
        if field==self.shared_field:
            p,m,x=point
            return self.shared_field_offset+(p*self.geometry.packed_shape[1]+m)*self.geometry.packed_shape[2]+x
        layout=self.provider_layouts[parity] if self.parity_providers and parity is not None else self.providers
        thread,slot=self.owner(point);slots=layout['slots'][field]
        if slot not in slots:return -1
        return layout['field_offsets'][field]+(
            (thread//32)*len(slots)+slots.index(slot))*32+thread%32

    def route(self,output,read,slot,parity):
        point=self.point(16,slot)
        source=self.geometry.read_point(output,read,point,(0,parity,0),136)
        thread,source_slot=self.owner(source)
        return ('shared' if thread//32 else 'shuffle' if thread%32!=16 else 'register',source_slot,read.offset[2])

    def prove(self):
        checks=0;counts=dict(register=0,shuffle=0,shared=0)
        operator=yee_operator(cpml_axes=(0,1,2));used=set()
        for parity,update,thread,slot in itertools.product((0,1),operator.updates,range(0,self.threads,32),range(self.slots)):
            if update.output.field not in FIELDS:continue
            point=self.point(thread,slot)
            if self.owner(point)!=(thread,slot):raise IllegalSchedule('Private owner is not invertible')
            for read in update.reads:
                if read.field not in FIELDS:continue
                source=self.geometry.read_point(update.output.field,read,point,(0,parity,0),136)
                source_thread,source_slot=self.owner(source)
                route,actual_slot,dx=self.route(update.output.field,read,slot,parity)
                expected='shared' if source_thread//32!=thread//32 else 'shuffle' if source_thread%32 else 'register'
                if route!=expected or actual_slot!=source_slot:raise IllegalSchedule('Nonuniform per-warp transport route')
                if route=='shared':
                    index=self.provider_index(read.field,source,parity)
                    if index<0:raise IllegalSchedule('Missing cross-warp shared provider')
                    used.add(index)
                counts[route]+=1;checks+=1
        for parity in (0,1):
            layout=self.provider_layouts[parity] if self.parity_providers else self.providers
            indices={self.provider_index(f,(p,m,x),parity) for f in FIELDS for p,m in layout['rows'][f] for x in range(32)}
            if self.shared_field is not None:
                dense={self.provider_index(self.shared_field,p,parity) for p in itertools.product(*(range(c) for c in self.geometry.packed_shape))}
                if indices&dense or dense!=set(range(self.shared_field_offset,self.allocated_words)):raise IllegalSchedule('Dense shared field aliases providers')
            if indices!=set(range(layout['words'])):raise IllegalSchedule('Shared provider alias or hole')
        return dict(schema='beamz.diamond-warp-storage.v1',plus=self.plus,minus=self.minus,
                    threads=self.threads,slots=self.slots,private_field_registers=(6-int(self.shared_field is not None))*self.slots,shared_field=self.shared_field,shared_field_offset=self.shared_field_offset if self.shared_field is not None else None,
                    shared_words=self.allocated_words,shared_bytes=4*self.allocated_words+76,parity_providers=self.parity_providers,
                    provider_slots=self.providers['slots'],parity_layouts=[{k:v for k,v in p.items() if k!='rows'} for p in self.provider_layouts] if self.parity_providers else None,checked_routes=checks,routes=counts,passed=True,
                    scope='Exact warp/slot route and provider cover; arithmetic and CUDA synchronization require separate checks')
