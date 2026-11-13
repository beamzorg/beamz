"""Derive rectangular private Z/Y warp tiles from the versioned cell trace.

This keeps physical X in complete warps. Unlike the column lowering, each warp
owns several Y rows and Z planes, so only the perimeter needs shared providers.
Field placement may move one complete component to shared memory to bound the
private register requirement. Geometry is the ordinary, unretimed moving box.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import itertools

from scripts.cuda_space_time_ir import FIELDS,IllegalSchedule,yee_operator


@dataclass(frozen=True)
class BoxWarpStorage:
    cache:tuple[int,int,int]=(16,16,32)
    private_z:int=4
    private_y:int=4
    shared_field:str|tuple[str,...]|None='Hx'
    profile_slots:int=12
    provider_home:bool=False
    slot_major:bool=False

    @property
    def dense_fields(self):return () if self.shared_field is None else (self.shared_field,) if isinstance(self.shared_field,str) else tuple(self.shared_field)

    def dense_offset(self,field):return self.providers['words']+self.dense_fields.index(field)*self.points

    def shared_home(self,field,slot):return field in self.dense_fields or (self.provider_home and slot in self.providers['slots'][field])

    @property
    def slots(self):return self.private_z*self.private_y
    @property
    def points(self):return self.cache[0]*self.cache[1]*self.cache[2]
    @property
    def threads(self):return self.points//self.slots

    def validate(self):
        if self.private_z<1 or self.private_y<1 or any(c<2 or c&(c-1) for c in self.cache) or self.cache[2]!=32 or self.cache[0]%self.private_z or self.cache[1]%self.private_y:
            raise IllegalSchedule('Integral private rectangles in a power-of-two contiguous-X box required')
        if self.threads<32 or self.threads>1024 or any(f not in FIELDS for f in self.dense_fields) or len(set(self.dense_fields))!=len(self.dense_fields) or not 0<=self.profile_slots<=12:raise IllegalSchedule('Invalid field placement or CTA shape')

    def point(self,tid,slot):
        warp,lane=divmod(tid,32);yb=self.cache[1]//self.private_y
        return (warp//yb)*self.private_z+slot//self.private_y,(warp%yb)*self.private_y+slot%self.private_y,lane

    def owner(self,p):
        z,y,x=p;yb=self.cache[1]//self.private_y
        return ((z//self.private_z)*yb+y//self.private_y)*32+x,(z%self.private_z)*self.private_y+y%self.private_y

    def source(self,p,offset):return tuple((a+d)%n for a,d,n in zip(p,offset,self.cache,strict=True))

    @cached_property
    def providers(self):
        self.validate();rows={f:set() for f in FIELDS}
        for update,z,y in itertools.product(yee_operator(cpml_axes=(0,1,2)).updates,range(self.cache[0]),range(self.cache[1])):
            if update.output.field not in FIELDS:continue
            p=(z,y,0);thread,_=self.owner(p)
            for read in update.reads:
                if read.field not in FIELDS or read.field in self.dense_fields:continue
                source=self.source(p,read.offset);source_thread,_=self.owner(source)
                if source_thread//32!=thread//32:rows[read.field].add(source[:2])
        slots={};offsets={};words=0
        for field in FIELDS:
            slots[field]=tuple(sorted({(z%self.private_z)*self.private_y+y%self.private_y for z,y in rows[field]}))
            expected={(z,y) for z in range(self.cache[0]) for y in range(self.cache[1])
                      if (z%self.private_z)*self.private_y+y%self.private_y in slots[field]}
            if rows[field]!=expected:raise IllegalSchedule('Nonperiodic rectangular provider cover')
            offsets[field]=words;words+=len(rows[field])*32
        return dict(slots=slots,offsets=offsets,words=words)

    @property
    def shared_words(self):return self.providers['words']+self.points*len(self.dense_fields)

    def provider_index(self,f,p):
        tid,slot=self.owner(p);slots=self.providers['slots'][f]
        if slot not in slots:return -1
        if self.slot_major:return self.providers['offsets'][f]+slots.index(slot)*self.threads+tid
        return self.providers['offsets'][f]+((tid//32)*len(slots)+slots.index(slot))*32+tid%32

    def prove(self):
        seen=set();checks=0;routes=dict(register=0,shuffle=0,shared=0)
        for tid,slot in itertools.product(range(self.threads),range(self.slots)):
            p=self.point(tid,slot)
            if self.owner(p)!=(tid,slot) or p in seen:raise IllegalSchedule('Private box map is not bijective')
            seen.add(p)
        for update,tid,slot in itertools.product(yee_operator(cpml_axes=(0,1,2)).updates,range(self.threads),range(self.slots)):
            if update.output.field not in FIELDS:continue
            p=self.point(tid,slot)
            for read in update.reads:
                if read.field not in FIELDS:continue
                source=self.source(p,read.offset);src,_=self.owner(source)
                _,source_slot=self.owner(source)
                route='shared' if self.shared_home(read.field,source_slot) or src//32!=tid//32 else 'shuffle' if src!=tid else 'register'
                if route=='shared' and read.field not in self.dense_fields and self.provider_index(read.field,source)<0:raise IllegalSchedule('Missing cross-warp provider')
                routes[route]+=1;checks+=1
        indices={self.provider_index(f,self.point(t,s)) for f in FIELDS for t in range(self.threads)
                 for s in self.providers['slots'][f]}
        if indices!=set(range(self.providers['words'])):raise IllegalSchedule('Shared providers alias or leave holes')
        return dict(schema='beamz.rectangular-private-warp-storage.v1',cache=self.cache,private_z=self.private_z,private_y=self.private_y,
                    threads=self.threads,slots=self.slots,field_points=self.points,
                    private_field_register_requirement=sum(not self.shared_home(f,s) for f in FIELDS for s in range(self.slots)),
                    provider_home=self.provider_home,slot_major=self.slot_major,
                    shared_field=self.shared_field,provider_words=self.providers['words'],shared_words=self.shared_words,
                    shared_bytes=4*self.shared_words+76,cached_shared_bytes=4*self.shared_words+76+self.profile_slots*72*4,profile_slots=self.profile_slots,
                    checked_routes=checks,routes=routes,passed=True,
                    scope='Exact register/shuffle/shared field cut, provider injectivity and placement requirement; compiler resources and GPU semantics separate')


class BoxReadBinding:
    def __init__(self,storage,slot):self.storage=storage;self.slot=slot;self.shuffles={}
    def __call__(self,output,read):
        t=self.storage;point=t.point(0,self.slot);source=t.source(point,read.offset);thread,slot=t.owner(source)
        field=FIELDS.index(read.field);family='e' if field>=3 else 'h'
        if read.version!=(0 if field>=3 else int(output.startswith('E'))):raise IllegalSchedule('Wrong version in box read')
        if read.field in t.dense_fields:
            dz,dy,dx=read.offset
            extra=t.dense_fields.index(read.field)*t.points
            return f'frame[BoxDenseOffset+{extra}+Slot(zs+({dz}),ys+({dy}),lane+({dx}))]'
        if t.provider_home and slot in t.providers['slots'][read.field]:
            yb=t.cache[1]//t.private_y;zb=t.cache[0]//t.private_z;zg,yg=divmod(thread//32,yb)
            if zg==zb-1 and zb>1:zg=-1
            if yg==yb-1 and yb>1:yg=-1
            return f'frame[BoxProviderIndex<{field},{slot},{zg},{yg}>(warp,((lane+({read.offset[2]}))&31))]'
        value=f'{family}[{field%3}][{slot}]'
        if thread//32==0:
            if thread%32:
                name=f'box_shuffle_{field}_{self.slot}';self.shuffles[name]=(value,read.offset[2]);return name
            return value
        yb=t.cache[1]//t.private_y;zb=t.cache[0]//t.private_z;zg,yg=divmod(thread//32,yb)
        if zg==zb-1 and zb>1:zg=-1
        if yg==yb-1 and yb>1:yg=-1
        return f'frame[BoxProviderIndex<{field},{slot},{zg},{yg}>(warp,lane)]'
