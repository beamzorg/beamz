"""Emit exact parity-specific provider ranks and typed warp-neighbor addresses."""
from __future__ import annotations

from scripts.cuda_space_time_ir import FIELDS,IllegalSchedule


def generate_helpers(t):
    if not t.parity_providers:raise IllegalSchedule('Parity-specific provider allocation required')
    bp,bm=t.plus,t.minus;mb=t.geometry.packed_shape[1]//bm;pb=t.geometry.packed_shape[0]//bp
    lines=[f'''constexpr int DiamondPrivateSlots={t.slots},DiamondPrivateThreads={t.threads};
constexpr int DiamondProviderWords={t.allocated_words};
constexpr size_t DiamondPrivateSharedBytes=4*DiamondProviderWords+76;
template<int Slot>__device__ __forceinline__ int DiamondPrivateQ(int tid){{
 const int warp=tid>>5,lane=tid&31,plus=(warp/{mb})*{bp}+Slot/{bm},minus=(warp%{mb})*{bm}+Slot%{bm};
 return (plus*DiamondHalfY+minus)*CX+lane;
}}
template<int Field>__device__ __forceinline__ int DiamondProviderIndex(int plus,int minus,int x,int parity){{
 const int slot=(plus%{bp})*{bm}+minus%{bm},warp=(plus/{bp})*{mb}+minus/{bm};int rank=-1;
''']
    for parity,layout in enumerate(t.provider_layouts):
        lines.append(f' if(parity=={parity})'+'{')
        for field,name in enumerate(FIELDS):
            slots=layout['slots'][name];offset=layout['field_offsets'][name]
            lines.append(f'  if constexpr(Field=={field})'+'{switch(slot){'+''.join(f'case {s}:rank={i};break;' for i,s in enumerate(slots))+f'}}if(rank<0)return -1;return {offset}+(warp*{len(slots)}+rank)*32+x;'+'}')
        lines.append(' }')
    lines+=[' return -1;','}',f'''template<int Field,int Slot,int Parity,int PlusDelta=0,int MinusDelta=0>
__device__ __forceinline__ int DiamondProviderIndexTyped(int warp,int x){{
 const int source_warp=(((warp/{mb}+PlusDelta)&{pb-1})*{mb})+((warp%{mb}+MinusDelta)&{mb-1});
''']
    for parity,layout in enumerate(t.provider_layouts):
        for field,name in enumerate(FIELDS):
            slots=layout['slots'][name];offset=layout['field_offsets'][name]
            if not slots:continue
            rank=':'.join(f'Slot=={s}?{i}' for i,s in enumerate(slots))+':-1'
            lines.append(f' if constexpr(Field=={field}&&Parity=={parity})'+'{'+f'constexpr int rank={rank};static_assert(rank>=0,"Typed read needs an allocated provider");return {offset}+(source_warp*{len(slots)}+rank)*32+x;'+'}')
    lines+=[' return -1;','}','']
    if t.shared_field is not None:lines.insert(0,f'constexpr int DiamondSharedFieldOffset={t.shared_field_offset};')
    return '\n'.join(lines)


def store_lines(t,field,slot,value,indent='   '):
    name=FIELDS[field]
    if name==t.shared_field:return []
    if t.parity_providers:
        return [f'{indent}if(parity=={p})frame[DiamondProviderIndexTyped<{field},{slot},{p}>(warp,lane)]={value};'
                for p,layout in enumerate(t.provider_layouts) if slot in layout['slots'][name]]
    if slot in t.providers['slots'][name]:
        return [f'{indent}frame[DiamondProviderIndex<{field}>(q/(DiamondHalfY*CX),(q/CX)&(DiamondHalfY-1),lane)]={value};']
    return []


def shared_expression(t,output,read,slot,parity):
    point=t.point(16,slot);source=t.geometry.read_point(output,read,point,(0,parity,0),136)
    thread,source_slot=t.owner(source);warp=thread//32
    mb=t.geometry.packed_shape[1]//t.minus;pb=t.geometry.packed_shape[0]//t.plus
    plus,minus=divmod(warp,mb)
    if pb>1 and plus==pb-1:plus=-1
    if mb>1 and minus==mb-1:minus=-1
    field=FIELDS.index(read.field)
    if source_slot not in t.provider_layouts[parity]['slots'][read.field]:raise IllegalSchedule('Typed parity provider missing')
    return f'frame[DiamondProviderIndexTyped<{field},{source_slot},{parity},{plus},{minus}>(warp,((lane+({read.offset[2]}))&31))]'
