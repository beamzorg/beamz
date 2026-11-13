"""Derive CUDA lane routes and shared providers from a symbolic Yee cell DAG.

Unlike the earlier column binding, X may occupy a subwarp. A warp can therefore
carry several Y rows: some Y reads use shuffles and only the remainder require
shared memory. Provider sets are derived from actual thread ownership, then
covered by disjoint affine regions. Space stays symbolic in emitted kernels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json

from scripts.cuda_space_time_ir import FIELDS, IllegalSchedule, validate_operator, yee_operator
from scripts.cuda_phase_schedule_ir import Event,check_schedule


@dataclass(frozen=True)
class TileTopology:
    cache: tuple[int, int, int] = (16, 16, 16)
    z_groups: int = 4

    @property
    def threads(self):
        return self.cache[1]*self.cache[2]*self.z_groups

    @property
    def slots(self):
        return self.cache[0]//self.z_groups

    @property
    def rows_per_warp(self):
        return 32//self.cache[2]

    def validate(self):
        if (len(self.cache)!=3 or any(type(n) is not int or n<2 or n&(n-1) for n in self.cache)
                or self.cache[2] not in (8,16,32) or type(self.z_groups) is not int
                or self.z_groups<1 or self.cache[0]%self.z_groups
                or self.cache[1]%self.rows_per_warp or self.threads>1024):
            raise IllegalSchedule('Power-of-two cache, complete subwarp rows and integral Z groups required')

    def owner(self, point):
        z,y,x=point;_,cy,cx=self.cache
        return (z//self.slots)*cy*cx+y*cx+x,z%self.slots

    def source(self, point, offset):
        return tuple((p+d)%n for p,d,n in zip(point,offset,self.cache,strict=True))

    def route(self, point, offset):
        source=self.source(point,offset)
        target_tid,_=self.owner(point);source_tid,_=self.owner(source)
        if target_tid==source_tid:return 'register'
        if target_tid//32==source_tid//32:return 'shuffle'
        return 'shared'


@dataclass(frozen=True)
class ProviderRegion:
    field: str
    axis: int
    offset: int
    words: int
    residue: int
    excluded_y_residue: int | None = None

    def index(self, topology, point):
        z,y,x=point;cz,cy,cx=topology.cache;r=topology.rows_per_warp
        if self.axis==1:
            if y%r!=self.residue:return None
            return self.offset+(z*(cy//r)+y//r)*cx+x
        if z%topology.slots!=self.residue:return None
        if self.excluded_y_residue is not None:
            if y%r==self.excluded_y_residue:return None
            packed_y=y-y//r-(self.excluded_y_residue==0)
            extent_y=cy-cy//r
        else:packed_y,extent_y=y,cy
        return self.offset+(z//topology.slots*extent_y+packed_y)*cx+x


def field_offsets(operator=None):
    operator=yee_operator(cpml_axes=(0,1,2)) if operator is None else operator
    validate_operator(operator)
    result={f:set() for f in FIELDS}
    for update in operator.updates:
        for read in update.reads:
            if read.field in result and any(read.offset):
                if sum(abs(d) for d in read.offset)!=1:
                    raise IllegalSchedule('Topology lowering currently admits nearest axial reads')
                result[read.field].add(tuple(read.offset))
    return result


def allocate_providers(topology=TileTopology(), operator=None):
    topology.validate();offsets=field_offsets(operator);regions=[];cursor=0
    cz,cy,cx=topology.cache;r=topology.rows_per_warp
    for field in FIELDS:
        ys={d[1] for d in offsets[field] if d[1]};zs={d[0] for d in offsets[field] if d[0]}
        if len(ys)>1 or len(zs)>1:raise IllegalSchedule('One direction per field/axis required')
        y_residue=None
        if ys:
            y_residue=0 if next(iter(ys))>0 else r-1
            words=cz*(cy//r)*cx
            regions.append(ProviderRegion(field,1,cursor,words,y_residue));cursor+=words
        if zs and topology.z_groups>1:
            words=topology.z_groups*(cy-(cy//r if ys else 0))*cx
            if words:
                regions.append(ProviderRegion(field,0,cursor,words,
                    0 if next(iter(zs))>0 else topology.slots-1,y_residue));cursor+=words
    return tuple(regions),cursor


def verify_topology(topology=TileTopology(), *, operator=None):
    """Compare generated affine regions with an independent enumerated route cut."""
    topology.validate();offsets=field_offsets(operator);regions,words=allocate_providers(topology,operator)
    points=list(itertools.product(*(range(n) for n in topology.cache)))
    assert len({topology.owner(p) for p in points})==len(points)
    required=set();routes=Counter()
    for field,reads in offsets.items():
        for point,delta in itertools.product(points,reads):
            route=topology.route(point,delta);routes[route]+=1
            if route=='shared':required.add((field,topology.source(point,delta)))
    allocated={};addresses=set()
    for region in regions:
        for point in points:
            index=region.index(topology,point)
            if index is None:continue
            key=region.field,point
            if key in allocated or index in addresses:raise IllegalSchedule('Provider alias')
            allocated[key]=index;addresses.add(index)
    if set(allocated)!=required or addresses!=set(range(words)):
        raise IllegalSchedule('Shared provider cover differs from traced cross-warp reads')
    return dict(schema='beamz.tile-topology.v1',topology=asdict(topology),threads=topology.threads,
                private_field_floats_per_thread=6*topology.slots,rows_per_warp=topology.rows_per_warp,
                owned_zyx=[n-1 for n in topology.cache],regions=[asdict(r) for r in regions],
                shared_words=words,shared_bytes=words*4,route_counts=dict(routes),
                unique_required_providers=len(required),disjoint_minimal_cover=True,
                ownership_checks=len(points),GPU_executed=False)


def emit_provider_helpers(cache=(16,16,16)):
    """Emit region ranks from the planner, with a template Z-group parameter."""
    cz,cy,cx=cache
    # Discover region ordering from a concrete legal topology; expressions for
    # region size are affine in Groups. The independent checker covers each map.
    sample=TileTopology(cache,2);sample.validate();regions,_=allocate_providers(sample)
    lines=['template<int Groups>struct TraceTileStorage{',
           f' static constexpr int Slots={cz}/Groups,Rows={32//cx};']
    expressions=[]
    for i,region in enumerate(regions):
        size=(f'{cz}*({cy}/Rows)*{cx}' if region.axis==1 else
              f'Groups*({cy}'+(f'-{cy}/Rows' if region.excluded_y_residue is not None else '')+f')*{cx}')
        expressions.append(size)
        lines.append(f' static constexpr int O{i}='+('+'.join(expressions[:-1]) or '0')+';')
    lines+=[' static constexpr int Words='+'+'.join(expressions)+';',
            ' template<bool Electric>__device__ __forceinline__ static void Store(float*frame,int z,int y,int x,float v0,float v1,float v2){']
    for family in ('H','E'):
        lines.append(f'  if constexpr(Electric=={str(family=="E").lower()})'+'{')
        for i,region in enumerate(regions):
            if not region.field.startswith(family):continue
            cond,index=_cpp_region(region,i,cy,cx)
            c='xyz'.index(region.field[-1]);lines.append(f'   if({cond})frame[{index}]=v{c};')
        lines.append('  }')
    lines+=[' }',' template<bool Electric>__device__ __forceinline__ static float Read(const float*frame,int c,int z,int y,int x){',
            f'  z&={cz-1};y&={cy-1};x&={cx-1};']
    for family in ('H','E'):
        lines.append(f'  if constexpr(Electric=={str(family=="E").lower()})'+'{')
        for i,region in enumerate(regions):
            if not region.field.startswith(family):continue
            cond,index=_cpp_region(region,i,cy,cx)
            c='xyz'.index(region.field[-1]);lines.append(f'   if(c=={c}&&({cond}))return frame[{index}];')
        lines.append('  }')
    lines+=['  return __int_as_float(0x7fffffff); // Impossible for an admitted traced route.',
            ' }','};']
    return '\n'.join(lines)


def _cpp_region(region,index,cy,cx):
    # E uses positive neighbours, H negative: source edge follows phase.
    electric=region.field.startswith('E')
    if region.axis==1:
        residue='0' if electric else 'Rows-1'
        return f'y%Rows=={residue}',f'O{index}+(z*({cy}/Rows)+y/Rows)*{cx}+x'
    residue='0' if electric else 'Slots-1'
    cond=f'z%Slots=={residue}'
    if region.excluded_y_residue is not None:
        yr='0' if electric else 'Rows-1';cond+=f'&&y%Rows!={yr}'
        y=f'(y-y/Rows{"-1" if electric else ""})';extent=f'({cy}-{cy}/Rows)'
    else:y,extent='y',str(cy)
    return cond,f'O{index}+(z/Slots*{extent}+{y})*{cx}+x'


def prove_shared_e_effects(cache=(16,16,16),*,early_ghost=False,omit_publication=False):
    """A shared-E overwrite must follow H retirement and avoid E's own readers."""
    points=set(itertools.product(*(range(n) for n in cache)))
    e_all=frozenset(('E',*p) for p in points);h_shared=frozenset(('H',*p) for p in points)
    events=[]
    for n in (1,2):
        head=tuple((-n-1)%c for c in cache)
        ghost=frozenset(('E',*p) for p in points if any(a==b for a,b in zip(p,head)))
        owned=e_all-ghost
        events.append(Event(f'H{n} compute',e_all))
        if early_ghost:events.append(Event(f'E{n} EARLY ghosts',writes=ghost))
        events.append(Event(f'H{n} frontier',writes=h_shared))
        events.append(Event(f'B1.{n}',barrier=True))
        events.append(Event(f'E{n} ghosts',writes=ghost))
        events.append(Event(f'E{n} point-local compute',owned|h_shared,owned))
        if not omit_publication:events.append(Event(f'B2.{n}',barrier=True))
    return check_schedule(events)
