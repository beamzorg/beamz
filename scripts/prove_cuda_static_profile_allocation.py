"""Trace dead E planes, allocate independent profiles, and poison field aliases.

The numerical/version oracle is the existing fixed-logical model with explicit
storage-only instrumentation. The exact transformation and both source hashes
are retained so this proof cannot silently certify another dataflow model.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from scripts.build_cuda_partial_plane_pipeline import once
from scripts.cuda_space_time_ir import FIELDS,yee_operator


def dead_planes():
    dims=(8,16,32);readset={f:set() for f in FIELDS[3:]}
    for d in itertools.product(*(range(1,n) for n in dims)):
        for update in yee_operator(cpml_axes=(0,1,2)).updates:
            if update.output.field not in FIELDS:continue
            for r in update.reads:
                if r.field in readset:readset[r.field].add(tuple(v+dv-1 for v,dv in zip(d,r.offset)))
    planes=[]
    for f,axis in zip(FIELDS[3:],(2,1,0)):
        reserved={d for d in itertools.product(*(range(n) for n in dims)) if d[axis]==dims[axis]-1}
        assert not reserved&readset[f],('Profile plane has a canonical field reader',f)
        planes.append(dict(field=f,normal_axis=axis,words=len(reserved),canonical_read_positions=len(readset[f]),read_overlap=0))
    slots=[]
    for i in range(864):
        if i<128:f=3;d=(i//16,i%16,31)
        elif i<384:k=i-128;f=4;d=(k//32,15,k%32)
        else:k=i-384;f=5;d=(7,k//32,k%32)
        assert d[(2,1,0)[f-3]]==dims[(2,1,0)[f-3]]-1
        assert d not in readset[FIELDS[f]]
        slots.append((f,d))
    assert len(set(slots))==864,'Independent profile words alias'
    return dict(reserved_planes=planes,reserved_words=sum(p['words'] for p in planes),independent_profile_words=864,unique_profile_slots=len(set(slots)))


def instrument():
    parent=Path(__file__).with_name('prove_cuda_static_logical_cells.py');original=parent.read_text();text=original
    text=once(text,"    valid=lambda f,p:","    reserved=lambda f,d:f>=3 and d[(2,1,0)[f-3]]==owned[(2,1,0)[f-3]]\n    valid=lambda f,p:")
    text=once(text,'        for n in range(first,last+1):','''        profiles={}
        for d in logical:
            for f in range(3,6):
                if reserved(f,d):profiles[f,d]=('immutable-profile',f,d);frame[f][address(d)]=profiles[f,d]
        for n in range(first,last+1):''')
    text=once(text,'                    for f in range(phase*3,phase*3+3):frame[f][address(d)]=temporary[f%3][address(d)]','''                    for f in range(phase*3,phase*3+3):
                        if reserved(f,d) and negative!='clobber-owned-profile-slot':continue
                        frame[f][address(d)]=temporary[f%3][address(d)]''')
    text=once(text,"                        for f in range(phase*3,phase*3+3):frame[f][address(d,negative=='wrong-ghost-map')]=imported(f,p,n)",'''                        for f in range(phase*3,phase*3+3):
                            if reserved(f,d) and negative!='clobber-ghost-profile-slot':continue
                            frame[f][address(d)]=imported(f,p,n)''')
    text=once(text,'            for d in own:\n                if n!=depth','''            for (f,d),bits in profiles.items():assert frame[f][address(d)]==bits,('Profile slot overwritten by field store',q,n,FIELDS[f],d,negative)
            for d in own:
                if n!=depth''')
    text=once(text,"                    actual=frame[f][address(d)];assert actual==token(f,p,n),('Static export',q,n,f,p,actual)","                    actual=temporary[f-3][address(d)] if f>=3 and negative!='export-E-from-profile-slot' else frame[f][address(d)];assert actual==token(f,p,n),('Static export',q,n,f,p,actual)")
    namespace={'__name__':'beamz_static_profile_storage_oracle','__file__':str(parent)};exec(compile(text,str(parent)+':profile-instrumented','exec'),namespace)
    return namespace['simulate'],original,text,parent


def report():
    allocation=dead_planes();simulate,original,instrumented,parent=instrument()
    cases=[simulate(depth=d,swizzle=s) for d in (4,16) for s in (False,True)]
    cases.append(simulate(minimum=(31,35,39),depth=16))
    negatives={}
    for name in ('clobber-owned-profile-slot','clobber-ghost-profile-slot','export-E-from-profile-slot'):
        try:simulate(depth=4,negative=name)
        except AssertionError as error:negatives[name]=str(error)
        else:raise AssertionError('Ineffective profile allocation negative:'+name)
    assert parent.read_text()==original,'Model source changed during execution'
    return dict(schema='beamz.static-logical-profile-allocation.v1',passed=True,allocation=allocation,cases=cases,negative_controls=negatives,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_source_sha256=hashlib.sha256(original.encode()).hexdigest(),instrumented_source_sha256=hashlib.sha256(instrumented.encode()).hexdigest(),instrumented_source=instrumented,
                scope='Canonical field readsets prove896Ewords neverreadasfields; all864independentprofilesfitdistinctslots. Poisoned reservedplanes surviveallinit/owned/ghostphasewrites whileallfields/auxversions and Eprivate terminal/trail exports remain exact. Not GPU/FP/performance evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),canonical_reads=sum(c['canonical_reads'] for c in r['cases']),allocation=r['allocation'],negative_controls=r['negative_controls'])))
