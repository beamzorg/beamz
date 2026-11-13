"""Dependency- and alias-checked scheduling of traced publication effects.

This pass only reorders already-produced values and immutable guard SSA values
within one publication region. Buffer names (or the supplied alias mapping)
identify canonical physical allocations, not merely different pointer names.
It does not reassociate arithmetic, move barriers, or reorder source/DFT effects.
The native lowering supplies the allocation and trail-component span facts.
These are plain stores: consumers observe the completed publication region
through the enclosing kernel/rank dependency, not an intermediate release flag.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping,Sequence
from scripts.cuda_space_time_ir import FIELDS


@dataclass(frozen=True)
class StoreEffect:
    name:str
    buffer:str
    word_interval:tuple[int,int]|None
    guard:str
    value:str
    predicate:str|None=None
    region:str='publication'
    ordered:bool=False


def storage_disjoint(a:StoreEffect,b:StoreEffect,aliases:Mapping[str,str])->bool:
    if aliases.get(a.buffer,a.buffer)!=aliases.get(b.buffer,b.buffer):return True
    if a.word_interval is None or b.word_interval is None:return False
    lo=max(a.word_interval[0],b.word_interval[0]);hi=min(a.word_interval[1],b.word_interval[1])
    return lo>=hi


def group_guards(effects:Sequence[StoreEffect],*,aliases:Mapping[str,str]|None=None,dependencies:Sequence[tuple[str,str]]=())->tuple[StoreEffect,...]:
    effects=tuple(effects);aliases={} if aliases is None else aliases
    if len({e.name for e in effects})!=len(effects):raise ValueError('Unique effect identifiers required')
    if any(e.word_interval is not None and not (0<=e.word_interval[0]<e.word_interval[1]) for e in effects):
        raise ValueError('Nonempty, nonnegative allocation intervals required')
    if len({e.region for e in effects})>1:raise ValueError('Cannot cross publication/barrier regions')
    if any(e.ordered for e in effects):raise ValueError('Ordered source/DFT/volatile effects cannot be grouped')
    guards=list(dict.fromkeys(e.guard for e in effects))
    result=tuple(e for guard in guards for e in effects if e.guard==guard)
    old={e.name:i for i,e in enumerate(effects)};new={e.name:i for i,e in enumerate(result)}
    for before,after in dependencies:
        if before not in old or after not in old:raise ValueError('Unknown dependency effect')
        if old[before]>=old[after]:raise ValueError('Input violates a dependency')
        if new[before]>=new[after]:raise ValueError('Grouping violates a value/effect dependency')
    for i,a in enumerate(effects):
        for b in effects[i+1:]:
            if new[a.name]>new[b.name] and not storage_disjoint(a,b,aliases):
                raise ValueError('Grouping may reorder overlapping stores: '+a.name+'/'+b.name)
    return result


def native_publication(surface_points:int=735)->tuple[StoreEffect,...]:
    if surface_points<1:raise ValueError('Positive component span required')
    # Original native loop order: Hc/Ec trail effects, then Hc/Ec terminal
    # effects, for c=0,1,2. Each trail family owns one disjoint span.
    effects=[]
    for component in range(3):
        for field in (component,component+3):
            name=FIELDS[field]
            effects.append(StoreEffect('trail_'+name,'trails',(field*surface_points,(field+1)*surface_points),'surface',name))
        for field in (component,component+3):
            name=FIELDS[field]
            effects.append(StoreEffect('final_'+name,'field_'+name,None,'terminal',name,predicate='contains_'+name))
    return tuple(effects)


def prove_native_publication():
    old=native_publication();new=group_guards(old)
    before_transitions=sum(i==0 or old[i-1].guard!=e.guard for i,e in enumerate(old))
    after_transitions=sum(i==0 or new[i-1].guard!=e.guard for i,e in enumerate(new))
    assert len(new)==12 and before_transitions==6 and after_transitions==2
    # A finite bit-pattern interpreter checks observable memory, including
    # signed zero/subnormal/NaN/infinity encodings, with no FP arithmetic.
    patterns=[0,0x80000000,1,0x007fffff,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc12345,0xffffffff]
    checks=0
    for surface in (0,1,30,31,464,465,650,651,734):
        for active_surface in (False,True):
            for active_terminal in (False,True):
                for support in range(64):
                    for pattern in patterns:
                        def execute(sequence):
                            memory={}
                            for effect in sequence:
                                if not (active_surface if effect.guard=='surface' else active_terminal):continue
                                if effect.predicate and not (support&(1<<FIELDS.index(effect.value))):continue
                                address=effect.word_interval[0]+surface if effect.word_interval else 0
                                memory[effect.buffer,address]=pattern^FIELDS.index(effect.value)
                            return memory
                        assert execute(old)==execute(new);checks+=1
    return dict(schema='beamz.trace-effect-grouping-proof.v1',effect_count=12,guard_groups_before=6,guard_groups_after=2,
                finite_memory_bitpattern_checks=checks,component_support_masks_checked=64,inner_component_predicates_retained=True,
                source_order=[e.name for e in old],scheduled_order=[e.name for e in new],
                allocation_fact='Native trail arena is separate from all field allocations',
                restrictions=['One publication region','Already-produced immutable values','No ordered source/DFT effects','Dependencies and possibly overlapping stores retain order'])
