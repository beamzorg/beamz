"""Canonical six-field version proof for phase-local registers/shared rebinding.

H owns private registers while reading E from shared. E owns private registers
while reading H from shared. Ghost imports scatter directly into their final
shared family, and the next H register state is reloaded before shared reuse.
This models arithmetic operands/placement, not FP32 arithmetic or GPU ordering.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def simulate(minimum=(13,17,21),owned=(7,15,31),depth=4,swizzle=False,negative=None):
    shape=[tuple(n+d for n,d in zip(minimum,delta)) for delta in SUPPORT_DELTA]
    resident=tuple(b+1 for b in owned);cz,cy,cx=resident;pad=math.prod(resident)
    assert cy==16 and cx==32 and cz in (8,16)
    maximum=tuple(n+1 for n in minimum);counts=tuple((n+depth+b-1)//b for n,b in zip(maximum,owned))
    logical=list(itertools.product(*(range(n) for n in resident)))
    raw=lambda s:(s[0]*cy+s[1])*cx+s[2]
    def shared(s,wrong=False):
        z,y,x=(v%n for v,n in zip(s,resident))
        if swizzle and not wrong:x^=y|((z&1)<<4)
        return (z*cy+y)*cx+x
    assert len({shared(s) for s in logical})==pad
    valid=lambda f,p:all(0<=x<n for x,n in zip(p,shape[f]))
    linear=lambda f,p:(p[0]*shape[f][1]+p[1])*shape[f][2]+p[2]
    token=lambda f,p,n:(n,f,p) if valid(f,p) else None
    output=[[None]*math.prod(s) for s in shape];psi=[[0]*len(a) for a in output]
    history={};dead=object();reads=0;ghost_reads=0;rebindings=0;early_late=0;tasks=0;phase_live_values=0
    operator=yee_operator(cpml_axes=(0,1,2))
    operands={u.output.field:tuple(r for r in sorted(u.reads) if r.field in FIELDS) for u in operator.updates if u.output.field in FIELDS}
    def imported(f,p,n):
        nonlocal ghost_reads
        if not valid(f,p):return None
        q=tuple((x+n)//b for x,b in zip(p,owned));expected=token(f,p,n)
        value=output[f][linear(f,p)] if n==depth else history.get((q,n,f,p),dead)
        assert value==expected,('Imported wrong field/point/version',f,p,n,q,value)
        ghost_reads+=1;return value
    for q in sorted(itertools.product(*(range(c) for c in counts)),key=lambda q:(sum(q),q)):
        origin=tuple(a*b for a,b in zip(q,owned));first=max(1,*(o-n for o,n in zip(origin,minimum)));last=min(depth,*(o+b-1 for o,b in zip(origin,owned)));tasks+=1
        if first>last:continue
        early_late+=first>1;h=[[dead]*pad for _ in range(3)];frame=[[dead]*pad for _ in range(3)]
        for s in logical:
            if first==1:
                p=tuple(o+(v if v!=d-1 else -1) for o,v,d in zip(origin,s,resident))
                for c in range(3):h[c][raw(s)]=token(c,p,0);frame[c][shared(s)]=token(c+3,p,0)
            else:
                prev=first-1;ghost=tuple((-first)%d for d in resident);d=tuple((v-g)%n for v,g,n in zip(s,ghost,resident));p=tuple(o-first+x for o,x in zip(origin,d))
                for c in range(3):
                    h[c][raw(s)]=imported(c,p,prev) if any(x==0 for x in d) else None
                    frame[c][shared(s)]=imported(c+3,p,prev) if any(x==0 for x in d) else None
        for n in range(first,last+1):
            ghost=tuple((-n-1)%d for d in resident)
            def physical(s):return tuple(o-n-1+(v-g)%d for o,v,g,d in zip(origin,s,ghost,resident))
            def is_owned(s):return all(v!=g for v,g in zip(s,ghost))
            own=[s for s in logical if is_owned(s)];halo=[s for s in logical if not is_owned(s)]
            if negative=='overwrite-old-E-before-H':
                for s in halo:
                    for c in range(3):frame[c][shared(s)]=token(c,physical(s),n)
            # H phase has no live E register array.
            phase_live_values=max(phase_live_values,3*pad)
            for s in own:
                p=physical(s);i=raw(s)
                for f in range(3):
                    if not valid(f,p):h[f][i]=None;continue
                    for operand in operands[FIELDS[f]]:
                        sf=FIELDS.index(operand.field);sp=tuple(x+dx for x,dx in zip(p,operand.offset))
                        if not valid(sf,sp):continue
                        adjacent=tuple(v+dx for v,dx in zip(s,operand.offset))
                        value=h[sf][i] if sf<3 else frame[sf-3][shared(adjacent)]
                        expected=token(sf,sp,n-1+operand.version)
                        assert value==expected,('H operand',q,n,FIELDS[f],str(operand),s,sp,value,expected);reads+=1
                    assert psi[f][linear(f,p)]==n-1,('H psi order',q,n,f,p);psi[f][linear(f,p)]=n;h[f][i]=token(f,p,n)
            # R0 retires old-E stencil readers. Each old-E center has exactly
            # one thread/slot reader, so swap it with private H immediately
            # before that thread overwrites its shared center. One bank only.
            written=set()
            for s in own:
                p=physical(s)
                for c in range(3):
                    value=frame[c][shared(s)]
                    if valid(c+3,p):assert value==token(c+3,p,n-1),('E old-center capture',q,n,s,c,value)
                    frame[c][shared(s)]=h[c][raw(s)];written.add((c,shared(s)))
                    h[c][raw(s)]=value
            for s in halo:
                p=physical(s)
                for c in range(3):
                    slot=shared(s,negative=='wrong-ghost-swizzle');frame[c][slot]=imported(c,p,n);written.add((c,slot))
            assert len(written)==3*pad,'Incomplete H shared-family writes'
            e=h # Same physical register bank, now interpreted as E values.
            for s in own:
                p=physical(s);i=raw(s)
                for f in range(3,6):
                    if not valid(f,p):e[f-3][i]=None;continue
                    for operand in operands[FIELDS[f]]:
                        sf=FIELDS.index(operand.field);sp=tuple(x+dx for x,dx in zip(p,operand.offset))
                        if not valid(sf,sp):continue
                        adjacent=tuple(v+dx for v,dx in zip(s,operand.offset))
                        value=e[sf-3][i] if sf>=3 else frame[sf][shared(adjacent)]
                        expected=token(sf,sp,n-1+operand.version)
                        assert value==expected,('E operand',q,n,FIELDS[f],str(operand),s,sp,value,expected);reads+=1
                    assert psi[f][linear(f,p)]==n-1,('E psi order',q,n,f,p);psi[f][linear(f,p)]=n;e[f-3][i]=token(f,p,n)
            # Export H/E while H shared and E registers still live.
            for s in own:
                p=physical(s);d=tuple((v-g)%n for v,g,n in zip(s,ghost,resident));surface=any(x==b for x,b in zip(d,owned))
                if not surface and n!=depth:continue
                for f in range(6):
                    if not valid(f,p):continue
                    value=frame[f][shared(s)] if f<3 else e[f-3][raw(s)];assert value==token(f,p,n),('Export',q,n,f,p,value)
                    if n==depth:output[f][linear(f,p)]=value
                    else:history[q,n,f,p]=value
            if n<last:
                # Ghost E registers are dead; reuse them for current H before
                # B3, so later E-ghost scatter can overwrite those shared slots.
                for s in halo:
                    for c in range(3):
                        if negative=='skip-H-ghost-rebind':continue
                        h[c][raw(s)]=frame[c][shared(s)];rebindings+=1
                # B3 retires H stencil/ghost-reload readers. Swap each owned
                # E register with its H shared center before overwriting it.
                written=set()
                for s in own:
                    for c in range(3):
                        next_h=frame[c][shared(s)];frame[c][shared(s)]=e[c][raw(s)];h[c][raw(s)]=next_h;rebindings+=1;written.add((c,shared(s)))
                for s in halo:
                    p=physical(s)
                    for c in range(3):frame[c][shared(s)]=None if negative=='skip-E-ghost-scatter' else imported(c+3,p,n);written.add((c,shared(s)))
                assert len(written)==3*pad,'Incomplete E shared-family writes'
            del e # End the E interpretation; h names that same physical bank.
    assert all(v==token(f,p,depth) for f,s in enumerate(shape) for p in itertools.product(*(range(n) for n in s)) for v in [output[f][linear(f,p)]]),'Incomplete final Yee fields'
    assert all(v==depth for a in psi for v in a),'Incomplete point-local auxiliary histories'
    return dict(minimum=minimum,owned=owned,depth=depth,swizzle=swizzle,tasks=tasks,late_tasks=early_late,canonical_reads=reads,ghost_reads=ghost_reads,shared_to_private_H_rebindings=rebindings,max_phase_field_values=phase_live_values,single_physical_register_bank=True,simultaneous_H_E_arithmetic_registers=False)


def report():
    cases=[]
    for owned in ((7,15,31),(15,15,31)):
        for depth in (4,16):
            for swizzle in (False,True):cases.append(simulate(owned=owned,depth=depth,swizzle=swizzle))
    negatives={}
    for name in ('overwrite-old-E-before-H','wrong-ghost-swizzle','skip-H-ghost-rebind','skip-E-ghost-scatter'):
        try:simulate(depth=4,swizzle=True,negative=name)
        except AssertionError as e:negatives[name]=str(e)
        else:raise AssertionError('Ineffective phase-register negative:'+name)
    return dict(passed=True,schema='beamz.phase-register-rebinding-model.v1',cases=cases,negative_controls=negatives,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Canonical six-field/version and all point-local auxiliary history reads; shared-family ownership/poison and swizzle; not numerical/CUDA-memory visibility or throughput evidence')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),canonical_reads=sum(c['canonical_reads'] for c in r['cases']),ghost_reads=sum(c['ghost_reads'] for c in r['cases']),negative_controls=r['negative_controls'])))
